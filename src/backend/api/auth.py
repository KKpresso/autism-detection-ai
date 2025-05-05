from datetime import datetime, timedelta
from typing import Optional, Dict, List
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError
import secrets
from enum import Enum
import logging
from prometheus_client import Counter


# Security configurations
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Initialize password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "predict": "Make predictions",
        "admin": "Administrative access",
        "metrics": "View metrics",
    }
)

# Initialize metrics
AUTH_FAILURE_COUNTER = Counter(
    'auth_failures_total',
    'Total number of authentication failures'
)


class Role(str, Enum):
    """User roles."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"


class TokenType(str, Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"


class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: str
    role: Role
    disabled: bool = False
    rate_limit: int = 100  # requests per minute


class Token(BaseModel):
    """Token model."""
    access_token: str
    refresh_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: List[str] = []
    type: TokenType


# Mock database - Replace with actual database in production
USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "role": Role.ADMIN,
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
        "rate_limit": 1000
    }
}

# Rate limiting storage
rate_limits: Dict[str, List[datetime]] = {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Get password hash."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[User]:
    """Get user from database."""
    if username in USERS_DB:
        user_dict = USERS_DB[username]
        return User(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, USERS_DB[username]["hashed_password"]):
        AUTH_FAILURE_COUNTER.inc()
        return None
    return user


def create_token(
    data: dict,
    token_type: TokenType,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({
        "exp": expire,
        "type": token_type,
        "iat": datetime.utcnow()
    })
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_tokens(username: str, scopes: List[str]) -> Token:
    """Create access and refresh tokens."""
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_token(
        data={"sub": username, "scopes": scopes},
        token_type=TokenType.ACCESS,
        expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_token(
        data={"sub": username},
        token_type=TokenType.REFRESH,
        expires_delta=refresh_token_expires
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    """Get current user from token."""
    authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != TokenType.ACCESS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": authenticate_value},
            )
        
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(
            username=username,
            scopes=token_scopes,
            type=TokenType.ACCESS
        )
    
    except (JWTError, ValidationError):
        AUTH_FAILURE_COUNTER.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )
    
    user = get_user(username=token_data.username)
    if user is None:
        AUTH_FAILURE_COUNTER.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )
    
    # Check for required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            AUTH_FAILURE_COUNTER.inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    return user


async def get_current_active_user(
    current_user: User = Security(get_current_user, scopes=[])
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def check_rate_limit(user: User):
    """Check rate limit for user."""
    now = datetime.now()
    user_requests = rate_limits.get(user.username, [])
    
    # Remove old requests
    user_requests = [t for t in user_requests if t > now - timedelta(minutes=1)]
    
    # Check rate limit
    if len(user_requests) >= user.rate_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Add new request
    user_requests.append(now)
    rate_limits[user.username] = user_requests


def get_role_permissions(role: Role) -> List[str]:
    """Get permissions for role."""
    permissions = {
        Role.ADMIN: ["predict", "admin", "metrics"],
        Role.RESEARCHER: ["predict", "metrics"],
        Role.CLINICIAN: ["predict"]
    }
    return permissions.get(role, [])
