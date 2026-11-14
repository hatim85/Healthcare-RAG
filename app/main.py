"""Production FastAPI Service for Healthcare RAG and Document Ingestion.

Features:
- Asynchronous large PDF ingestion with page-by-page streaming, PaddleOCR fallback, and checkpoints.
- Real-time job progress tracking and resumption.
- Clinical question answering with end-to-end guardrails and 7-dimension observability.
- Health checks for Qdrant, SQLite, and caching layers.
- Multi-provider MCP client integration (Langfuse, Qdrant, Firecrawl, Groq, LiteLLM, Redis).
"""

from contextlib import asynccontextmanager, AsyncExitStack
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import threading
from typing import Any
import uuid

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.auth import (
    AuthenticatedUser,
    LoginRequest,
    Token,
    authenticate,
    create_access_token,
    verify_credentials,
    JWT_EXPIRATION_MINUTES,
)

from app.generation.answer_pipeline import AnswerPipeline
from app.generation.circuit_breaker import CircuitBreakerOpenException
from app.ingestion.batched_indexer import BatchedClinicalIndexer
from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.job_manager import IngestionJobManager
from app.middleware.rate_limiter import (
    ask_limiter,
    rate_limit,
    upload_limiter,
)
from app.observability.audit_logger import log_audit
from app.retrieval.qdrant_client import get_qdrant_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("healthcare_rag_api")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Shared singletons
registry = DocumentRegistry()
indexer = BatchedClinicalIndexer(registry=registry)
job_manager = IngestionJobManager(registry=registry, indexer=indexer)

# Thread-safe lazy/warmed initialization for AnswerPipeline
_pipeline_lock = threading.Lock()
_answer_pipeline: AnswerPipeline | None = None


def get_pipeline() -> AnswerPipeline:
    global _answer_pipeline
    if _answer_pipeline is None:
        with _pipeline_lock:
            if _answer_pipeline is None:
                logger.info("Initializing AnswerPipeline singleton...")
                _answer_pipeline = AnswerPipeline()
    return _answer_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Healthcare RAG API service...")
    if os.getenv("TESTING") != "1" and os.getenv("WARM_PIPELINE", "0") == "1":
        get_pipeline()

    async with AsyncExitStack() as stack:
        # 1. Start Healthcare MCP Server Streamable HTTP session manager
        try:
            from app.mcp_server import mcp_server
            await stack.enter_async_context(mcp_server.session_manager.run())
            logger.info("Healthcare MCP Streamable HTTP session manager active")
        except Exception as exc:
            logger.warning(f"Could not start MCP Streamable HTTP session manager: {exc}")

        # 2. Start multi-provider MCP clients (Langfuse, Qdrant, Firecrawl, LiteLLM, Redis)
        mcp_results: dict[str, bool] = {}
        if os.getenv("MCP_CLIENTS_ENABLED", "true").lower() in ("true", "1", "yes"):
            try:
                from app.mcp_clients.api import startup_mcp_clients
                mcp_results = await startup_mcp_clients()
                connected = [n for n, ok in mcp_results.items() if ok]
                if connected:
                    logger.info(f"MCP external providers ready: {', '.join(connected)}")
            except (Exception, BaseException) as exc:
                logger.warning(f"MCP client startup failed (non-fatal): {exc}")

        yield

        # Shutdown MCP clients
        try:
            from app.mcp_clients.api import shutdown_mcp_clients
            await shutdown_mcp_clients()
        except Exception as exc:
            logger.debug(f"MCP client shutdown error: {exc}")

        logger.info("Shutting down Healthcare RAG API worker pool...")
        job_manager.shutdown(wait=False)


app = FastAPI(
    title="Healthcare RAG & Large PDF Ingestion API",
    description=(
        "Production-grade Clinical RAG system featuring streaming page-by-page "
        "PDF ingestion, selective PaddleOCR fallback, SQLite checkpointing, "
        "deterministic chunk IDs, and verifiable clinical question answering."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Production Authentication Middleware for Healthcare MCP Server endpoints
from app.mcp_auth import MCPAuthenticationMiddleware

app.add_middleware(MCPAuthenticationMiddleware)

# Mount external MCP providers API router
from app.mcp_clients.api import router as mcp_providers_router

app.include_router(mcp_providers_router)

# Mount internal Healthcare MCP Server (Streamable HTTP + SSE fallback)
from mcp.server.transport_security import TransportSecuritySettings
from app.mcp_server import mcp_server
from app.mcp_web import app as mcp_web_app

_mcp_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
app.mount("/mcp/sse", mcp_server.sse_app(transport_security=_mcp_security))
app.mount("/mcp", mcp_server.streamable_http_app(streamable_http_path="/", transport_security=_mcp_security))
app.mount("/mcp-web", mcp_web_app)


@app.exception_handler(CircuitBreakerOpenException)
async def circuit_breaker_exception_handler(request: Request, exc: CircuitBreakerOpenException):
    """Fail-fast error response when downstream LLM gateway or vector DB circuit trips."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "service": exc.service_name,
            "retry_after": round(exc.retry_after, 1),
        },
        headers={"Retry-After": str(int(max(1, exc.retry_after)))},
    )


import anyio

@app.exception_handler(anyio.ClosedResourceError)
async def closed_resource_exception_handler(request: Request, exc: anyio.ClosedResourceError):
    """Gracefully handle closed MCP session streams during client disconnects."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"detail": "Session closed."},
    )




# ---------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------

class DocumentPathRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to a local PDF file.")
    force_reindex: bool = Field(False, description="Reindex even if document is already completed.")


class IngestionJobResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: str
    message: str


class BatchIngestRequest(BaseModel):
    file_paths: list[str] | None = Field(None, description="List of file paths to ingest.")
    directory_path: str | None = Field(None, description="Directory path containing PDF files.")
    recursive: bool = Field(True, description="Scan subdirectories recursively.")
    force_reindex: bool = Field(False, description="Reindex even if document is already completed.")


class BatchIngestResponse(BaseModel):
    total_queued: int
    jobs: list[IngestionJobResponse]


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: str
    current_page: int
    total_pages: int
    progress_percent: float
    indexed_chunks: int
    error: str | None = None
    created_at: str
    updated_at: str


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    file_path: str
    content_hash: str
    version: int
    status: str
    total_pages: int
    last_completed_page: int
    indexed_chunks: int
    extraction_stats: dict[str, Any]
    error_message: str | None = None
    created_at: str
    updated_at: str


class AskRequest(BaseModel):
    question: str = Field(..., description="Clinical question to answer based on guidelines.")
    conversation_history: list[dict[str, str]] | None = Field(
        None, description="Prior conversation turns [{'role': 'user'|'assistant', 'content': '...'}]"
    )
    session_id: str | None = Field(None, description="Client session ID for request tracing.")
    verbose: bool = Field(False, description="Print execution steps to server console.")
    stream: bool = Field(False, description="Stream answer tokens incrementally via Server-Sent Events (SSE).")


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    sqlite_registry: str
    database: str = "sqlite"
    redis: str = "ok"
    active_jobs: int
    timestamp: str


# ---------------------------------------------------------
# Authentication Endpoint
# ---------------------------------------------------------

@app.post(
    "/auth/token",
    response_model=Token,
    summary="Obtain a JWT access token using user credentials",
)
def login_for_access_token(credentials: LoginRequest, request: Request):
    """Authenticate with username and password to receive a signed JWT access token.

    Default accounts available for testing:
    - admin / admin123 (role: admin, tenant: * or custom)
    - clinician / clinician123 (role: clinician, tenant: clinic_general)
    - researcher / researcher123 (role: researcher, tenant: research_institute)
    """
    client_ip = request.client.host if request.client else "unknown"
    is_valid, role, tenant_id = verify_credentials(
        credentials.username, credentials.password, credentials.tenant_id
    )
    if not is_valid:
        log_audit(
            user_id=credentials.username,
            role="unauthenticated",
            tenant_id=credentials.tenant_id or "default",
            action="auth.login",
            client_ip=client_ip,
            status="DENIED",
            details={"reason": "Invalid credentials"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    log_audit(
        user_id=credentials.username,
        role=role,
        tenant_id=tenant_id,
        action="auth.login",
        client_ip=client_ip,
        status="SUCCESS",
        details={"assigned_role": role, "tenant_id": tenant_id},
    )

    access_token = create_access_token(
        data={"sub": credentials.username, "role": role, "tenant_id": tenant_id}
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_MINUTES * 60,
        role=role,
        username=credentials.username,
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------
# Document Ingestion Endpoints
# ---------------------------------------------------------

@app.post(
    "/documents/upload",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF file for background ingestion",
    dependencies=[Depends(rate_limit(upload_limiter))],
)
async def upload_document(
    file: UploadFile = File(...),
    force_reindex: bool = Query(False, description="Reindex if file exists."),
    user: AuthenticatedUser = Depends(authenticate),
):
    """Upload a PDF file. The file is saved to disk and queued for asynchronous ingestion."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Save to disk
    dest_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        )

    try:
        job = job_manager.submit_job(dest_path, force_reindex=force_reindex, tenant_id=user.tenant_id)
        log_audit(
            user_id=user.username,
            role=user.role,
            tenant_id=user.tenant_id,
            action="document.upload",
            resource_id=job.document_id,
            details={"filename": file.filename, "job_id": job.job_id, "size_bytes": dest_path.stat().st_size},
        )
        return IngestionJobResponse(
            job_id=job.job_id,
            document_id=job.document_id,
            filename=job.filename,
            status=job.status,
            message="Document accepted and queued for background ingestion.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue ingestion job: {exc}",
        )


@app.post(
    "/documents",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Register a local PDF file for background ingestion",
    dependencies=[Depends(rate_limit(upload_limiter))],
)
def register_document_path(
    req: DocumentPathRequest,
    user: AuthenticatedUser = Depends(authenticate),
):
    """Register a local file system PDF path for asynchronous ingestion."""
    path = Path(req.file_path).resolve()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Valid PDF file not found at: {req.file_path}",
        )

    try:
        job = job_manager.submit_job(path, force_reindex=req.force_reindex, tenant_id=user.tenant_id)
        return IngestionJobResponse(
            job_id=job.job_id,
            document_id=job.document_id,
            filename=job.filename,
            status=job.status,
            message="Document registered and queued for background ingestion.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue document: {exc}",
        )


@app.post(
    "/documents/batch",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch ingest multiple PDF files or a directory",
    dependencies=[Depends(rate_limit(upload_limiter))],
)
def batch_ingest_documents(
    req: BatchIngestRequest,
    user: AuthenticatedUser = Depends(authenticate),
):
    """Queue large document collections incrementally without blocking the API."""
    jobs = []
    if req.directory_path:
        try:
            jobs = job_manager.submit_directory(
                req.directory_path,
                recursive=req.recursive,
                force_reindex=req.force_reindex,
                tenant_id=user.tenant_id,
            )
        except NotADirectoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    elif req.file_paths:
        jobs = job_manager.submit_batch(
            req.file_paths,
            force_reindex=req.force_reindex,
            tenant_id=user.tenant_id,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either directory_path or file_paths must be provided.",
        )

    responses = [
        IngestionJobResponse(
            job_id=j.job_id,
            document_id=j.document_id,
            filename=j.filename,
            status=j.status,
            message="Queued for ingestion",
        )
        for j in jobs
    ]

    return BatchIngestResponse(total_queued=len(jobs), jobs=responses)


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Check ingestion progress for a job",
)
def get_job_status(
    job_id: str,
    user: AuthenticatedUser = Depends(authenticate),
):
    """Retrieve the real-time status and page progress of an ingestion job."""
    job = job_manager.get_job(job_id)
    if not job or (user.tenant_id != "*" and job.tenant_id != user.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return JobStatusResponse(**job.to_dict())


@app.get(
    "/jobs",
    response_model=list[JobStatusResponse],
    summary="List all recent ingestion jobs",
)
def list_jobs(
    user: AuthenticatedUser = Depends(authenticate),
):
    """List all queued, running, and completed ingestion jobs."""
    jobs = job_manager.list_jobs()
    if user.tenant_id != "*":
        jobs = [j for j in jobs if j.tenant_id == user.tenant_id]
    return [JobStatusResponse(**j.to_dict()) for j in jobs]


@app.get(
    "/documents/{document_id}",
    response_model=DocumentMetadataResponse,
    summary="Inspect indexed document metadata",
)
def get_document_metadata(
    document_id: str,
    user: AuthenticatedUser = Depends(authenticate),
):
    """Inspect stored document metadata, checkpoints, and extraction stats."""
    doc = registry.get_document(document_id, tenant_id=user.tenant_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )
    return DocumentMetadataResponse(**doc.to_dict())


@app.get(
    "/documents",
    response_model=list[DocumentMetadataResponse],
    summary="List all registered documents",
)
def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: AuthenticatedUser = Depends(authenticate),
):
    """List all documents registered in the system."""
    docs = registry.list_documents(limit=limit, offset=offset, tenant_id=user.tenant_id)
    return [DocumentMetadataResponse(**d.to_dict()) for d in docs]


@app.post(
    "/documents/{document_id}/resume",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume interrupted document ingestion from checkpoint",
    dependencies=[Depends(rate_limit(upload_limiter))],
)
def resume_document(
    document_id: str,
    user: AuthenticatedUser = Depends(authenticate),
):
    """Resume an interrupted or failed document ingestion from its last checkpointed page."""
    try:
        job = job_manager.resume_document(document_id, tenant_id=user.tenant_id)
        return IngestionJobResponse(
            job_id=job.job_id,
            document_id=job.document_id,
            filename=job.filename,
            status=job.status,
            message="Ingestion resumed from checkpoint in background.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume document: {exc}",
        )


# ---------------------------------------------------------
# Question Answering Endpoints
# ---------------------------------------------------------

@app.post(
    "/ask",
    summary="Ask a clinical question grounded in indexed guidelines",
    dependencies=[Depends(rate_limit(ask_limiter))],
)
@app.post(
    "/api/ask",
    summary="Ask a clinical question (API alias)",
    dependencies=[Depends(rate_limit(ask_limiter))],
)
def ask_question(
    req: AskRequest,
    user: AuthenticatedUser = Depends(authenticate),
):
    """Answer a clinical question using the full RAG pipeline with safety rails,

    hybrid search, reranking, citation verification, and observability.
    If `stream: true` is passed, returns a Server-Sent Events (SSE) token stream.
    """
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    log_audit(
        user_id=user.username,
        role=user.role,
        tenant_id=user.tenant_id,
        action="query.ask",
        resource_id=req.session_id or "adhoc",
        details={"question_length": len(req.question), "stream": req.stream},
    )

    pipeline = get_pipeline()

    if req.stream:
        return StreamingResponse(
            pipeline.run_stream(
                question=req.question,
                conversation_history=req.conversation_history,
                session_id=req.session_id,
                verbose=req.verbose,
                tenant_id=user.tenant_id,
                user_id=user.username,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = pipeline.run(
            question=req.question,
            conversation_history=req.conversation_history,
            session_id=req.session_id,
            verbose=req.verbose,
            tenant_id=user.tenant_id,
            user_id=user.username,
        )
        return result
    except CircuitBreakerOpenException:
        raise
    except Exception as exc:
        logger.error(f"Error processing question: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing clinical question pipeline: {exc}",
        )


@app.post(
    "/ask/stream",
    summary="Ask a clinical question with token streaming via SSE",
    dependencies=[Depends(rate_limit(ask_limiter))],
)
def ask_question_stream(
    req: AskRequest,
    user: AuthenticatedUser = Depends(authenticate),
):
    """Answer a clinical question streaming tokens in real time via Server-Sent Events (SSE).

    Yields events:
    - `metadata`: Sources, model route, and context token allocations
    - `token`: Incremental answer tokens from LLM
    - `done`: Final verified response with complete citation and claim checks
    - `withheld` / `blocked`: If safety rails intervene
    """
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    log_audit(
        user_id=user.username,
        role=user.role,
        tenant_id=user.tenant_id,
        action="query.stream",
        resource_id=req.session_id or "adhoc",
        details={"question_length": len(req.question), "stream": True},
    )

    pipeline = get_pipeline()
    return StreamingResponse(
        pipeline.run_stream(
            question=req.question,
            conversation_history=req.conversation_history,
            session_id=req.session_id,
            verbose=req.verbose,
            tenant_id=user.tenant_id,
            user_id=user.username,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------
# Health Check Endpoints
# ---------------------------------------------------------

@app.get("/health/liveness", summary="Kubernetes liveness probe")
def liveness():
    """Liveness probe: verifies process is alive and responsive."""
    return {"status": "alive"}


@app.get("/health/readiness", summary="Kubernetes readiness probe")
def readiness():
    """Readiness probe: verifies core database and cache are ready to accept traffic."""
    try:
        registry.list_documents(limit=1)
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not ready: {exc}",
        )


@app.get("/health", response_model=HealthResponse, summary="System health check")
@app.get("/api/health", response_model=HealthResponse, summary="System health check (API alias)")
def health_check():
    """Verify application health and external connectivity across all subsystems."""
    # 1. Check Qdrant
    qdrant_status = "ok"
    try:
        client = get_qdrant_client()
        client.get_collections()
        client.close()
    except Exception as exc:
        qdrant_status = f"unhealthy: {exc}"

    # 2. Check Database registry
    sqlite_status = "ok"
    try:
        registry.list_documents(limit=1)
    except Exception as exc:
        sqlite_status = f"unhealthy: {exc}"

    db_engine = "sqlite" if getattr(registry, "is_sqlite", True) else "postgresql"

    # 3. Check Redis cache/broker
    redis_status = "ok"
    try:
        from app.caching.redis_client import get_redis_client
        r_client = get_redis_client()
        if not r_client.ping():
            redis_status = "unhealthy: ping returned false"
    except Exception as exc:
        redis_status = f"unhealthy: {exc}"

    overall_status = "healthy" if qdrant_status == "ok" and sqlite_status == "ok" else "degraded"

    return HealthResponse(
        status=overall_status,
        qdrant=qdrant_status,
        sqlite_registry=sqlite_status,
        database=db_engine,
        redis=redis_status,
        active_jobs=len([j for j in job_manager.list_jobs() if j.status in ("PENDING", "RUNNING")]),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# Mount built frontend static files if available
_frontend_dist = Path("frontend/dist")
if _frontend_dist.is_dir():
    from starlette.staticfiles import StaticFiles
    app.mount("/dashboard", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend_dist")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

