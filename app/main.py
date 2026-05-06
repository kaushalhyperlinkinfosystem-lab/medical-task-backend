import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes.api import llm_service, router

app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


@app.exception_handler(413)
async def request_entity_too_large(_request: Request, _exc: Exception):
    return JSONResponse(
        status_code=413,
        content={"success": False, "error": "Uploaded file exceeds the maximum allowed size."},
    )


@app.exception_handler(415)
async def unsupported_media_type(_request: Request, _exc: Exception):
    return JSONResponse(
        status_code=415,
        content={"success": False, "error": "Unsupported file type."},
    )


@app.on_event("shutdown")
async def _shutdown():
    await llm_service.aclose()


app.include_router(router)


@app.get("/", tags=["status"])
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "llm_provider": settings.LLM_PROVIDER or "unconfigured",
        "model": settings.LLM_MODEL or "default",
    }
