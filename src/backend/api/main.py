from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Security, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import asyncio
from prometheus_client import Counter, Histogram, start_http_server
from .auth import (
    Token, User, Role,
    create_tokens, authenticate_user, get_current_active_user,
    check_rate_limit, get_role_permissions
)
from .oauth import OAuthHandler, handle_oauth_callback

# Initialize FastAPI app
app = FastAPI(
    title="Autism Detection AI API",
    description="API for autism detection using fMRI data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize metrics
PREDICTION_COUNTER = Counter(
    'autism_predictions_total',
    'Total number of predictions made'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time taken for predictions'
)
QC_FAILURE_COUNTER = Counter(
    'qc_failures_total',
    'Number of quality control failures'
)

# Global state
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_VERSION = "1.0.0"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionCache:
    """Simple cache for predictions to enable A/B testing."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def add(self, file_hash: str, prediction: Dict[str, Any], version: str):
        """Add prediction to cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
            del self.cache[oldest[0]]
        
        self.cache[file_hash] = {
            'prediction': prediction,
            'version': version,
            'timestamp': datetime.now().isoformat()
        }
    
    def get(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get prediction from cache."""
        return self.cache.get(file_hash)


# Initialize prediction cache
prediction_cache = PredictionCache()


@app.on_event("startup")
async def startup_event():
    """Initialize model and start metrics server."""
    global MODEL
    
    try:
        # Load model
        model_path = Path(__file__).parent.parent / 'models' / 'best_model.pt'
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Initialize model
        MODEL = AutismClassifier(
            num_regions=checkpoint['num_regions'],
            hidden_dim=checkpoint['hidden_dim'],
            embedding_dim=checkpoint['embedding_dim'],
            num_heads=checkpoint['num_heads']
        )
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()
        
        logger.info(f"Model loaded successfully on {DEVICE}")
        
        # Start metrics server
        start_http_server(8000)
        logger.info("Metrics server started on port 8000")
    
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


async def process_prediction(
    data: np.ndarray,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Process prediction asynchronously."""
    try:
        # Preprocess data
        processed_data = preprocess_fmri_data(data)
        
        # Calculate QC metrics
        qc_metrics = calculate_qc_metrics(processed_data)
        
        # Check QC thresholds
        if qc_metrics['temporal_snr'] < 50 or qc_metrics['motion_metrics']['fd_mean'] > 0.5:
            QC_FAILURE_COUNTER.inc()
            raise HTTPException(
                status_code=400,
                detail="Data quality below acceptable threshold"
            )
        
        # Convert to tensor
        data_tensor = torch.from_numpy(processed_data).float().to(DEVICE)
        
        # Get prediction
        with torch.no_grad(), PREDICTION_LATENCY.time():
            output = MODEL(data_tensor)
            probability = torch.sigmoid(output).item()
        
        # Calculate network metrics
        network_metrics = calculate_network_metrics(processed_data)
        
        # Prepare response
        prediction = {
            'probability': probability,
            'prediction': 'ASD' if probability > 0.5 else 'Control',
            'confidence': abs(probability - 0.5) * 2,
            'qc_metrics': qc_metrics,
            'network_metrics': network_metrics,
            'model_version': MODEL_VERSION,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        
        # Add background task for cache update
        background_tasks.add_task(
            prediction_cache.add,
            hashlib.sha256(data.tobytes()).hexdigest(),
            prediction,
            MODEL_VERSION
        )
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """Login to get access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get permissions based on role
    permissions = get_role_permissions(user.role)
    
    # Create tokens
    return create_tokens(user.username, permissions)


@app.post("/refresh")
async def refresh_token(
    current_user: User = Depends(get_current_active_user)
) -> Token:
    """Refresh access token."""
    permissions = get_role_permissions(current_user.role)
    return create_tokens(current_user.username, permissions)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: User = Security(
        get_current_active_user,
        scopes=["predict"]
    )
) -> JSONResponse:
    """
    Make prediction from fMRI data.
    
    Args:
        file: NIfTI file containing fMRI data
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
    
    Returns:
        Prediction results including probability and metrics
    """
    # Check rate limit
    check_rate_limit(current_user)
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp:
            tmp.write(await file.read())
            tmp.flush()
            
            # Load NIfTI file
            img = nib.load(tmp.name)
            data = img.get_fdata()
        
        # Get prediction
        prediction = await process_prediction(data, background_tasks)
        
        # Add user info to prediction
        prediction['user'] = current_user.username
        prediction['role'] = current_user.role
        
        return JSONResponse(content=prediction)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Check API health."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def get_metrics(
    current_user: User = Security(
        get_current_active_user,
        scopes=["metrics"]
    )
) -> Dict[str, Any]:
    """Get prediction metrics."""
    return {
        "total_predictions": PREDICTION_COUNTER._value.get(),
        "qc_failures": QC_FAILURE_COUNTER._value.get(),
        "average_latency": PREDICTION_LATENCY.observe(),
        "model_version": MODEL_VERSION,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/users/me")
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user info."""
    return current_user


@app.get("/users/me/items")
async def read_own_items(
    current_user: User = Security(
        get_current_active_user,
        scopes=["predict"]
    )
) -> Dict[str, List[str]]:
    """Get user's predictions."""
    return {
        "predictions": [
            p["id"] for p in prediction_cache.cache.values()
            if p.get("user") == current_user.username
        ]
    }


@app.get("/auth/{provider}/login")
async def oauth_login(provider: str):
    """Redirect to OAuth provider login."""
    try:
        handler = OAuthHandler(provider)
        return RedirectResponse(url=handler.get_authorization_url())
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"OAuth provider error: {str(e)}"
        )


@app.get("/auth/{provider}/callback")
async def oauth_callback(provider: str, code: str, request: Request):
    """Handle OAuth callback."""
    try:
        # Get tokens
        tokens = await handle_oauth_callback(provider, code)
        
        # In production, you should redirect to frontend with tokens
        return JSONResponse(content=tokens.dict())
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"OAuth callback failed: {str(e)}"
        )


@app.get("/auth/providers")
async def list_providers():
    """List available OAuth providers."""
    return {
        "providers": [
            {
                "name": "Google",
                "id": "google",
                "icon": "https://www.google.com/favicon.ico",
                "login_url": "/auth/google/login"
            },
            {
                "name": "Microsoft",
                "id": "microsoft",
                "icon": "https://www.microsoft.com/favicon.ico",
                "login_url": "/auth/microsoft/login"
            },
            {
                "name": "GitHub",
                "id": "github",
                "icon": "https://github.com/favicon.ico",
                "login_url": "/auth/github/login"
            }
        ]
    }
