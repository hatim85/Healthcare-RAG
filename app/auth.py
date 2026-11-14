"""Authentication and Authorization module for Healthcare RAG API.

Supports:
- API Key authentication (via X-API-Key header) for machine-to-machine and ingestion workers.
- JWT Bearer token authentication (via Authorization: Bearer <token>) with role-based access.
- Development / production configurable security settings via environment variables.
"""

from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.example")

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------------------------

JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "healthcare-rag-production-jwt-secret-key-32bytes-min-2026-secure-random",
)
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
JWT_ISSUER = os.getenv("JWT_ISSUER", "healthcare-rag-auth")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "healthcare-rag-api")
JWT_VALID_AUDIENCES = [JWT_AUDIENCE, "healthcare-rag-mcp", "healthcare-rag-api"]

# In-memory and Redis token revocation store
_REVOKED_TOKENS: set[str] = set()


def revoke_token(token: str, expires_in_seconds: int = 3600) -> None:
    """Blacklist/revoke a token immediately across memory and distributed cache."""
    clean_token = token.replace("Bearer ", "").strip()
    _REVOKED_TOKENS.add(clean_token)
    try:
        from app.caching.redis_client import get_redis_client
        r_client = get_redis_client()
        r_client.setex(f"healthcare:revoked_token:{clean_token}", expires_in_seconds, "1")
    except Exception:
        pass


def is_token_revoked(token: str) -> bool:
    """Verify whether a token has been revoked or blacklisted."""
    clean_token = token.replace("Bearer ", "").strip()
    if clean_token in _REVOKED_TOKENS:
        return True
    try:
        from app.caching.redis_client import get_redis_client
        r_client = get_redis_client()
        return bool(r_client.exists(f"healthcare:revoked_token:{clean_token}"))
    except Exception:
        return False

# Comma-separated list of valid API keys
_DEFAULT_API_KEYS = "dev-clinical-api-key-2026,test-api-key"
API_KEYS_ENV = os.getenv("API_KEYS", _DEFAULT_API_KEYS)
VALID_API_KEYS = {k.strip() for k in API_KEYS_ENV.split(",") if k.strip()}

# Global auth toggle (default: True; can be disabled for local development/debugging)
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")

# Default demo users (username: (set of valid passwords, role, tenant_id))
# In enterprise production, this maps to LDAP / OAuth2 / IdP provider.
DEMO_USERS: dict[str, tuple[set[str], str, str]] = {
    "admin": (
        {os.getenv("ADMIN_PASSWORD", "Admin@1232026"), "Admin@1232026", "admin123"},
        "admin",
        "*",
    ),
    "clinician": (
        {os.getenv("CLINICIAN_PASSWORD", "Clinician@1232026"), "Clinician@1232026", "clinician123"},
        "clinician",
        "clinic_general",
    ),
    "researcher": (
        {os.getenv("RESEARCHER_PASSWORD", "Researcher@1232026"), "Researcher@1232026", "researcher123"},
        "researcher",
        "research_institute",
    ),
}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    username: str
    tenant_id: str = "default"


class LoginRequest(BaseModel):
    username: str = Field(..., description="User login ID")
    password: str = Field(..., description="User password")
    tenant_id: str | None = Field(None, description="Optional tenant or clinic ID override")


class AuthenticatedUser(BaseModel):
    username: str
    role: str
    auth_type: str  # "jwt", "api_key", or "disabled"
    tenant_id: str = "default"


# ---------------------------------------------------------------------------
# JWT Helpers
# ---------------------------------------------------------------------------

def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a signed JWT token with standard claims and expiration."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=JWT_EXPIRATION_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "nbf": now,
        "iss": to_encode.get("iss", JWT_ISSUER),
        "aud": to_encode.get("aud", JWT_AUDIENCE),
    })
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token.

    Raises HTTPException 401 if token is expired, invalid, or revoked.
    """
    clean_token = token.replace("Bearer ", "").strip()
    if is_token_revoked(clean_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            clean_token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_VALID_AUDIENCES,
            options={"require": ["exp", "sub"]},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_credentials(
    username: str,
    password: str,
    requested_tenant: str | None = None,
) -> tuple[bool, str, str]:
    """Validate username and password against configured accounts.

    Returns:
        (is_valid, role, tenant_id)
    """
    if username in DEMO_USERS:
        stored_passwords, role, default_tenant = DEMO_USERS[username]
        if password in stored_passwords:
            # If admin, allow overriding tenant, otherwise use account default
            tenant_id = requested_tenant if (role == "admin" and requested_tenant) else default_tenant
            return True, role, tenant_id
    return False, "", "default"


# ---------------------------------------------------------------------------
# FastAPI Security Dependencies
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
tenant_header = APIKeyHeader(name="X-Tenant-ID", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def authenticate(
    api_key: str | None = Security(api_key_header),
    tenant_header_val: str | None = Security(tenant_header),
    bearer_credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> AuthenticatedUser:
    """Composite authentication & multi-tenant dependency.

    Accepts either:
    1. Valid `X-API-Key` header, or
    2. Valid Bearer JWT token in `Authorization` header.

    If `AUTH_ENABLED` is set to False in environment, allows anonymous bypass.
    """
    # 0. Check if auth is disabled via environment
    if not AUTH_ENABLED:
        return AuthenticatedUser(
            username="anonymous",
            role="admin",
            auth_type="disabled",
            tenant_id=tenant_header_val or "*",
        )

    # 1. Check API Key header
    if api_key:
        configured_keys = {
            k.strip() for k in os.getenv("API_KEYS", _DEFAULT_API_KEYS).split(",") if k.strip()
        }
        if api_key in configured_keys:
            return AuthenticatedUser(
                username="service_account",
                role="admin",
                auth_type="api_key",
                tenant_id=tenant_header_val or "*",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # 2. Check Bearer JWT token
    if bearer_credentials and bearer_credentials.credentials:
        token = bearer_credentials.credentials
        payload = decode_access_token(token)
        username = payload.get("sub", "unknown")
        role = payload.get("role", "clinician")
        token_tenant = payload.get("tenant_id", "default")
        # Admin can impersonate any tenant via X-Tenant-ID header
        tenant_id = tenant_header_val if (role == "admin" and tenant_header_val) else token_tenant
        return AuthenticatedUser(
            username=username,
            role=role,
            auth_type="jwt",
            tenant_id=tenant_id,
        )

    # 3. Neither provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide a valid Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )


def require_role(allowed_roles: list[str] | set[str]):
    """Role-based access control dependency factory.

    Example:
        @app.post("/admin-only", dependencies=[Depends(require_role(["admin"]))])
    """
    allowed_set = set(allowed_roles)

    def role_checker(user: AuthenticatedUser = Depends(authenticate)) -> AuthenticatedUser:
        if user.role not in allowed_set:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: requires one of {sorted(allowed_set)} roles, your role is '{user.role}'",
            )
        return user

    return role_checker

