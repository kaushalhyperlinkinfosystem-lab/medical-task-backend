from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

_VALID_LANGUAGES = {"en", "hi", "gu"}


class BloodMarkerInput(BaseModel):
    name: str
    value: float
    unit: Optional[str] = None
    reference_range: Optional[str] = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Marker name must not be empty.")
        return v

    @field_validator("value")
    @classmethod
    def value_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Marker value must be zero or positive.")
        return v


class BloodReportInput(BaseModel):
    patient_id: Optional[str] = None
    report_name: Optional[str] = None
    markers: List[BloodMarkerInput] = Field(default_factory=list)
    hemoglobin: Optional[float] = None
    wbc: Optional[float] = None
    platelets: Optional[float] = None
    glucose: Optional[float] = None
    rbc: Optional[float] = None
    hematocrit: Optional[float] = None
    language: Optional[str] = "en"

    @field_validator("language")
    @classmethod
    def language_must_be_supported(cls, v: Optional[str]) -> str:
        lang = (v or "en").strip().lower()
        if lang not in _VALID_LANGUAGES:
            raise ValueError(f"Unsupported language '{lang}'. Allowed: en, hi, gu.")
        return lang

    @field_validator("hemoglobin", "wbc", "platelets", "glucose", "rbc", "hematocrit", mode="before")
    @classmethod
    def scalar_values_positive(cls, v: Any) -> Any:
        if v is not None and float(v) < 0:
            raise ValueError("Blood values must be zero or positive.")
        return v

    def normalized_markers(self) -> List[BloodMarkerInput]:
        if self.markers:
            return self.markers

        defaults = {
            "hemoglobin": ("g/dL", self.hemoglobin),
            "wbc": ("/uL", self.wbc),
            "platelets": ("/uL", self.platelets),
            "glucose": ("mg/dL", self.glucose),
            "rbc": ("M/uL", self.rbc),
            "hematocrit": ("%", self.hematocrit),
        }

        return [
            BloodMarkerInput(name=name, value=value, unit=unit)
            for name, (unit, value) in defaults.items()
            if value is not None
        ]


class TextReportInput(BaseModel):
    report_text: str
    report_name: Optional[str] = None
    language: Optional[str] = "en"

    @field_validator("language")
    @classmethod
    def language_must_be_supported(cls, v: Optional[str]) -> str:
        lang = (v or "en").strip().lower()
        if lang not in _VALID_LANGUAGES:
            raise ValueError(f"Unsupported language '{lang}'. Allowed: en, hi, gu.")
        return lang

    @field_validator("report_text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 20:
            raise ValueError("Report text must be at least 20 characters long.")
        return stripped


class ExtractedSection(BaseModel):
    title: str
    content: str


class ExtractedContent(BaseModel):
    raw_text: str = ""
    sections: List[ExtractedSection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_warnings: List[str] = Field(default_factory=list)


class KeyFinding(BaseModel):
    title: str
    detail: str
    severity: str = "info"
    evidence: Optional[str] = None


class IntegrityResult(BaseModel):
    status: str
    completeness_score: float
    missing_fields: List[str] = Field(default_factory=list)
    suspicious_values: List[str] = Field(default_factory=list)
    formatting_issues: List[str] = Field(default_factory=list)
    extraction_inconsistencies: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ConfidenceNote(BaseModel):
    area: str
    level: str
    note: str


class LLMExecution(BaseModel):
    provider: str
    primary_model: str
    mode: str
    live_model_used: bool
    fallback_model: Optional[str] = None


class ResponseMetadata(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    processing_time_ms: Optional[int] = None
    api_version: str = "1.0.0"


class PatientReportSummary(BaseModel):
    main_findings: List[str] = Field(default_factory=list)
    is_everything_normal: bool
    normal_status_text: str
    problems_concerns: List[str] = Field(default_factory=list)
    important_values_observations: List[str] = Field(default_factory=list)
    simple_explanation: str = ""
    suggested_next_steps: List[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    success: bool
    module: str
    file_name: str
    file_type: str
    extracted_content: ExtractedContent
    summary: str
    key_findings: List[KeyFinding] = Field(default_factory=list)
    integrity: IntegrityResult
    recommendations: List[str] = Field(default_factory=list)
    confidence_notes: List[ConfidenceNote] = Field(default_factory=list)
    llm: LLMExecution
    detected_issues: List[str] = Field(default_factory=list)
    explanation: str = ""
    possible_causes: List[str] = Field(default_factory=list)
    effects: List[str] = Field(default_factory=list)
    future_risks: List[str] = Field(default_factory=list)
    general_advice: List[str] = Field(default_factory=list)
    patient_report_summary: PatientReportSummary
    error: Optional[str] = None
    language: str = "en"
    disclaimer: str = (
        "This output supports review and triage only. Final interpretation must be "
        "performed by a licensed clinician."
    )
    meta: ResponseMetadata = Field(default_factory=ResponseMetadata)
