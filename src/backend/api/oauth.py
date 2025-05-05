from typing import Dict, Optional, Any
import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel
import json
from pathlib import Path
import logging
from datetime import datetime
from .auth import User, Role, create_tokens, Token

# Initialize logging
logger = logging.getLogger(__name__)

# Load OAuth configs
try:
    config_path = Path(__file__).parent.parent.parent / "config" / "oauth_config.json"
    with open(config_path) as f:
        OAUTH_CONFIG = json.load(f)
except Exception as e:
    logger.error(f"Failed to load OAuth config: {e}")
    OAUTH_CONFIG = {
        "google": {
            "client_id": "",
            "client_secret": "",
            "redirect_uri": "http://localhost:8000/auth/google/callback",
            "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
            "scope": "openid email profile"
        },
        "microsoft": {
            "client_id": "",
            "client_secret": "",
            "redirect_uri": "http://localhost:8000/auth/microsoft/callback",
            "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "userinfo_url": "https://graph.microsoft.com/v1.0/me",
            "scope": "openid email profile User.Read"
        },
        "github": {
            "client_id": "",
            "client_secret": "",
            "redirect_uri": "http://localhost:8000/auth/github/callback",
            "auth_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "userinfo_url": "https://api.github.com/user",
            "scope": "read:user user:email"
        }
    }


class OAuthConfig(BaseModel):
    """OAuth provider configuration."""
    client_id: str
    client_secret: str
    redirect_uri: str
    auth_url: str
    token_url: str
    userinfo_url: str
    scope: str


class OAuthError(Exception):
    """OAuth error."""
    pass


class OAuthHandler:
    """OAuth handler for different providers."""
    
    def __init__(self, provider: str):
        """Initialize OAuth handler."""
        if provider not in OAUTH_CONFIG:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        self.provider = provider
        self.config = OAuthConfig(**OAUTH_CONFIG[provider])
    
    def get_authorization_url(self) -> str:
        """Get authorization URL for OAuth provider."""
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": self.config.scope,
            "response_type": "code",
            "access_type": "offline",  # For refresh token
            "prompt": "consent"  # Force consent screen
        }
        
        # Add provider-specific parameters
        if self.provider == "microsoft":
            params["response_mode"] = "query"
        
        # Build URL
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.config.auth_url}?{query}"
    
    async def get_access_token(self, code: str) -> Dict[str, Any]:
        """Get access token from OAuth provider."""
        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "redirect_uri": self.config.redirect_uri,
            "code": code,
            "grant_type": "authorization_code"
        }
        
        headers = {"Accept": "application/json"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.token_url,
                data=data,
                headers=headers
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to get access token: {response.text}")
            
            return response.json()
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from OAuth provider."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.config.userinfo_url,
                headers=headers
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to get user info: {response.text}")
            
            return response.json()
    
    def map_user_info(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Map provider user info to our user model."""
        if self.provider == "google":
            return {
                "username": user_info["email"].split("@")[0],
                "email": user_info["email"],
                "full_name": user_info["name"],
                "picture": user_info.get("picture")
            }
        elif self.provider == "microsoft":
            return {
                "username": user_info["userPrincipalName"].split("@")[0],
                "email": user_info["userPrincipalName"],
                "full_name": user_info["displayName"],
                "picture": None  # Microsoft Graph API needs additional permissions
            }
        elif self.provider == "github":
            return {
                "username": user_info["login"],
                "email": user_info["email"],
                "full_name": user_info["name"] or user_info["login"],
                "picture": user_info["avatar_url"]
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


async def handle_oauth_callback(
    provider: str,
    code: str,
    default_role: Role = Role.CLINICIAN
) -> Token:
    """Handle OAuth callback and return tokens."""
    try:
        # Initialize handler
        handler = OAuthHandler(provider)
        
        # Get access token
        token_info = await handler.get_access_token(code)
        access_token = token_info["access_token"]
        
        # Get user info
        user_info = await handler.get_user_info(access_token)
        mapped_info = handler.map_user_info(user_info)
        
        # Create or update user
        from .auth import USERS_DB, get_password_hash
        
        username = mapped_info["username"]
        if username not in USERS_DB:
            USERS_DB[username] = {
                **mapped_info,
                "role": default_role,
                "disabled": False,
                "rate_limit": 100,
                "hashed_password": get_password_hash(access_token[:32]),  # Temporary password
                "oauth_provider": provider,
                "oauth_id": user_info.get("sub") or user_info.get("id"),
                "created_at": datetime.now().isoformat()
            }
        
        # Create tokens
        user = User(**USERS_DB[username])
        permissions = ["predict"]  # Default permission
        if user.role == Role.ADMIN:
            permissions.extend(["admin", "metrics"])
        elif user.role == Role.RESEARCHER:
            permissions.append("metrics")
        
        return create_tokens(username, permissions)
    
    except Exception as e:
        logger.error(f"OAuth error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"OAuth authentication failed: {str(e)}"
        )
