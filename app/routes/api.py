import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.schemas.models import AnalysisResponse, BloodReportInput, TextReportInput
from app.services.llm_service import MedicalLLMService
from app.services.report_analysis_service import (
    analyze_blood_markers,
    analyze_report_text,
    analyze_uploaded_report,
    get_llm_recommendations,
)

router = APIRouter()
llm_service = MedicalLLMService()

SUPPORTED_LANGUAGES = [
    {"code": "en", "label": "English"},
    {"code": "hi", "label": "Hindi"},
    {"code": "gu", "label": "Gujarati"},
]

_VALID_LANGUAGES = {"en", "hi", "gu"}

_ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}
_ALLOWED_BLOOD_UPLOAD_TYPES = _ALLOWED_IMAGE_TYPES | {
    "application/pdf",
    "text/plain",
    "text/csv",
}


def _validated_language(language: Optional[str]) -> str:
    lang = (language or "en").strip().lower()
    if lang not in _VALID_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{lang}'. Allowed values: en, hi, gu.",
        )
    return lang


def _validate_file(
    file: UploadFile,
    allowed_types: set[str],
    type_label: str,
) -> None:
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Size check via content-length header when available (actual bytes checked in service)
    content_length = file.size
    if content_length is not None and content_length > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum allowed size is {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    # MIME type check
    ct = (file.content_type or "").split(";")[0].strip().lower()
    filename = (file.filename or "").lower()

    # Fall back to extension-based guessing when browser sends generic MIME
    if ct in ("application/octet-stream", "") or not ct:
        if filename.endswith(".pdf"):
            ct = "application/pdf"
        elif filename.endswith(".txt"):
            ct = "text/plain"
        elif filename.endswith(".csv"):
            ct = "text/csv"
        elif any(filename.endswith(ext) for ext in (".jpg", ".jpeg")):
            ct = "image/jpeg"
        elif filename.endswith(".png"):
            ct = "image/png"
        elif filename.endswith(".webp"):
            ct = "image/webp"

    if ct not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=(
                f"This endpoint only accepts {type_label} files. "
                f"Received content type: '{ct or 'unknown'}'. "
                f"Allowed types: {', '.join(sorted(allowed_types))}."
            ),
        )


def _stamped(response: AnalysisResponse, start: float) -> AnalysisResponse:
    response.meta.processing_time_ms = int((time.perf_counter() - start) * 1000)
    response.meta.api_version = settings.API_VERSION
    return response


async def _analyze_upload(
    module: str,
    file: UploadFile,
    language: str,
    include_raw_text: bool,
) -> AnalysisResponse:
    start = time.perf_counter()
    result = await analyze_uploaded_report(module, file, language=language)
    if not include_raw_text:
        result.extracted_content.raw_text = ""
    return _stamped(result, start)


@router.post("/analyze/xray", response_model=AnalysisResponse, tags=["analyze"])
async def analyze_xray(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    include_raw_text: bool = Form(False),
):
    lang = _validated_language(language)
    _validate_file(file, _ALLOWED_IMAGE_TYPES, "image (JPEG, PNG, WEBP)")
    return await _analyze_upload("xray", file, lang, include_raw_text)


@router.post("/analyze/pdf", response_model=AnalysisResponse, tags=["analyze"])
async def analyze_pdf(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    include_raw_text: bool = Form(False),
):
    lang = _validated_language(language)
    _validate_file(file, {"application/pdf"}, "PDF")
    return await _analyze_upload("pdf", file, lang, include_raw_text)


@router.post("/analyze/blood-report", response_model=AnalysisResponse, tags=["analyze"])
async def analyze_blood_report(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    include_raw_text: bool = Form(False),
):
    lang = _validated_language(language)
    _validate_file(file, _ALLOWED_BLOOD_UPLOAD_TYPES, "PDF, plain text, CSV, or image")
    return await _analyze_upload("blood", file, lang, include_raw_text)


@router.post("/analyze/blood-values", response_model=AnalysisResponse, tags=["analyze"])
async def analyze_blood(blood_data: BloodReportInput):
    start = time.perf_counter()
    result = await analyze_blood_markers(blood_data)
    return _stamped(result, start)


@router.post("/analyze/report-text", response_model=AnalysisResponse, tags=["analyze"])
async def analyze_text_report(payload: TextReportInput):
    start = time.perf_counter()
    result = await analyze_report_text(payload)
    return _stamped(result, start)


@router.get("/llm-recommendations", tags=["diagnostics"])
async def llm_recommendations():
    return get_llm_recommendations()


@router.get("/health", tags=["diagnostics"])
async def health():
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "llm_configured": llm_service.is_configured,
        "llm_provider": llm_service.provider_name,
        "local_xray_model_configured": bool(settings.LOCAL_XRAY_MODEL_PATH),
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@router.get("/capabilities", tags=["diagnostics"])
async def capabilities():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "modules": [
            {
                "id": "xray",
                "name": "X-Ray Analysis",
                "accepted_types": sorted(_ALLOWED_IMAGE_TYPES),
                "inputs": ["image upload"],
                "outputs": ["summary", "key findings", "possible causes", "next steps", "safety notes"],
            },
            {
                "id": "pdf",
                "name": "PDF Report Analysis",
                "accepted_types": ["application/pdf"],
                "inputs": ["pdf upload", "pasted report text"],
                "outputs": ["summary", "extracted data", "key findings", "plain-language explanation", "next steps"],
            },
            {
                "id": "blood",
                "name": "Blood Report Analysis",
                "accepted_types": sorted(_ALLOWED_BLOOD_UPLOAD_TYPES),
                "inputs": ["lab report upload", "manual blood values"],
                "outputs": ["abnormal marker highlights", "simple explanation", "recommendations", "future risks"],
            },
        ],
        "supported_languages": SUPPORTED_LANGUAGES,
        "max_upload_mb": settings.MAX_UPLOAD_SIZE_MB,
        "patient_notice": (
            "This tool is designed for patient education and report understanding only. "
            "Final interpretation must come from a qualified clinician."
        ),
    }


@router.get("/health/llm", tags=["diagnostics"])
async def llm_health():
    return llm_service.diagnostic_snapshot()
