"""Production-grade Authentication Middleware for Healthcare MCP Server.

Enforces HIPAA Security Rule Access Controls (§ 164.312(a)(1)):
Restricts access to clinical MCP endpoints (/mcp, /mcp/sse) to authorized callers
holding a valid Bearer JWT token, Admin API key, or dedicated LiteLLM service credentials.
Provides fine-grained role context and tenant isolation via ContextVars.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import logging
import os
from typing import Any

import anyio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.auth import VALID_API_KEYS, decode_access_token, is_token_revoked
from app.observability.audit_logger import log_audit

logger = logging.getLogger("mcp_auth")

# Request-scoped ContextVars for fine-grained authorization within MCP tools
current_mcp_user: ContextVar[str] = ContextVar("current_mcp_user", default="anonymous")
current_mcp_role: ContextVar[str] = ContextVar("current_mcp_role", default="unauthenticated")
current_mcp_tenant: ContextVar[str] = ContextVar("current_mcp_tenant", default="default")


class MCPAuthenticationMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware validating Bearer tokens and API keys for MCP endpoints."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # 1. Path rewrite to avoid Starlette 307 Temporary Redirect loop on /mcp
        path = request.scope.get("path", "")
        if path == "/mcp":
            request.scope["path"] = "/mcp/"
            if "raw_path" in request.scope:
                request.scope["raw_path"] = b"/mcp/"
            path = "/mcp/"

        # Only protect the MCP protocol endpoints (/mcp, /mcp/), not /mcp-web
        if path.startswith("/mcp") and not path.startswith("/mcp-web"):
            # Allow CORS preflight requests without authentication
            if request.method == "OPTIONS":
                return await call_next(request)

            auth_enabled = os.getenv(
                "MCP_AUTH_ENABLED",
                os.getenv("AUTH_ENABLED", "true"),
            ).lower() in ("true", "1", "yes")

            if not auth_enabled:
                token_u = current_mcp_user.set("dev_bypass")
                token_r = current_mcp_role.set("admin")
                token_t = current_mcp_tenant.set("*")
                try:
                    return await call_next(request)
                finally:
                    current_mcp_user.reset(token_u)
                    current_mcp_role.reset(token_r)
                    current_mcp_tenant.reset(token_t)

            # Extract credentials from Authorization header or X-API-Key
            auth_header = request.headers.get("authorization", "").strip()
            api_key_header = request.headers.get("x-api-key", "").strip()

            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
            elif auth_header and not auth_header.startswith("Basic "):
                token = auth_header

            key_candidate = token or api_key_header
            client_ip = request.client.host if request.client else "unknown"

            # Check for token revocation first
            if token and is_token_revoked(token):
                log_audit(
                    user_id="revoked_token_client",
                    role="unauthenticated",
                    tenant_id="default",
                    action="mcp.auth_denied",
                    client_ip=client_ip,
                    status="REVOKED",
                    details={"path": path, "reason": "Bearer token is blacklisted/revoked"},
                )
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32000,
                            "message": "Unauthorized: Provided token has been revoked.",
                        },
                    },
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Static credentials matching
            litellm_secret = os.getenv("LITELLM_MCP_SECRET_KEY", "").strip()
            admin_key = os.getenv("ADMIN_API_KEY", "").strip()
            mcp_token = os.getenv("HEALTHCARE_MCP_TOKEN", "").strip()
            allowed_static_keys = set(VALID_API_KEYS)
            if admin_key:
                allowed_static_keys.add(admin_key)
            if litellm_secret:
                allowed_static_keys.add(litellm_secret)
            if mcp_token:
                allowed_static_keys.add(mcp_token)

            authenticated = False
            user_id = "anonymous"
            role = "unauthenticated"
            tenant_id = "default"

            # Check dedicated service keys
            if key_candidate and litellm_secret and key_candidate == litellm_secret:
                authenticated = True
                user_id = "litellm_gateway"
                role = "gateway"
                tenant_id = request.headers.get("x-tenant-id", "*")
            elif key_candidate and admin_key and key_candidate == admin_key:
                authenticated = True
                user_id = "admin_service"
                role = "admin"
                tenant_id = request.headers.get("x-tenant-id", "*")
            elif key_candidate and key_candidate in allowed_static_keys:
                authenticated = True
                user_id = "mcp_service_client"
                role = "clinician"
                tenant_id = request.headers.get("x-tenant-id", "default")
            elif token:
                # Validate signed JWT token
                try:
                    payload = decode_access_token(token)
                    authenticated = True
                    user_id = payload.get("sub", "jwt_user")
                    role = payload.get("role", "clinician")
                    token_tenant = payload.get("tenant_id", "default")
                    # Admin can override tenant via X-Tenant-ID header
                    req_tenant = request.headers.get("x-tenant-id")
                    tenant_id = req_tenant if (role in ("admin", "gateway") and req_tenant) else token_tenant
                except Exception as exc:
                    logger.debug(f"MCP JWT validation failed: {exc}")

            if not authenticated:
                log_audit(
                    user_id="unknown_mcp_client",
                    role="unauthenticated",
                    tenant_id="default",
                    action="mcp.auth_denied",
                    client_ip=client_ip,
                    status="DENIED",
                    details={"path": path, "reason": "Missing or invalid token"},
                )
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32000,
                            "message": (
                                "Unauthorized: Valid Bearer token or X-API-Key required "
                                "to access Healthcare Clinical MCP tools."
                            ),
                        },
                    },
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer, ApiKey"},
                )

            # Record authentication context in request.state and ContextVars
            request.state.mcp_user = user_id
            request.state.mcp_role = role
            request.state.mcp_tenant = tenant_id

            token_user = current_mcp_user.set(user_id)
            token_role = current_mcp_role.set(role)
            token_tenant = current_mcp_tenant.set(tenant_id)

            try:
                return await call_next(request)
            except (anyio.ClosedResourceError, asyncio.CancelledError):
                logger.debug("Client closed MCP stream connection abruptly on %s", path)
                return Response(status_code=204)
            finally:
                current_mcp_user.reset(token_user)
                current_mcp_role.reset(token_role)
                current_mcp_tenant.reset(token_tenant)

        try:
            return await call_next(request)
        except (anyio.ClosedResourceError, asyncio.CancelledError):
            logger.debug("Client closed connection abruptly on %s", path)
            return Response(status_code=204)
