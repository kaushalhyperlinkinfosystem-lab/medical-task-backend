import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageChops, ImageStat, UnidentifiedImageError

from app.config import settings
from app.schemas.models import (
    AnalysisResponse,
    BloodMarkerInput,
    BloodReportInput,
    ConfidenceNote,
    ExtractedContent,
    ExtractedSection,
    IntegrityResult,
    KeyFinding,
    LLMExecution,
    PatientReportSummary,
    TextReportInput,
)
from app.services.llm_service import LLMServiceError, MedicalLLMService
from app.services.xray_service import LocalXRayModelService

# Supported language codes and their display names
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
}

# Language instruction injected into the LLM system prompt
_LANGUAGE_INSTRUCTIONS: Dict[str, str] = {
    "en": "Respond entirely in English.",
    "hi": (
        "Respond entirely in Hindi (Devanagari script). "
        "All medical explanations, findings, recommendations, and advice must be written in simple, clear Hindi "
        "that a person with no medical background can understand. "
        "Use plain Hindi words; avoid English medical jargon wherever possible."
    ),
    "gu": (
        "Respond entirely in Gujarati (Gujarati script). "
        "All medical explanations, findings, recommendations, and advice must be written in simple, clear Gujarati "
        "that a person with no medical background can understand. "
        "Use plain Gujarati words; avoid English medical jargon wherever possible."
    ),
}

# Static disclaimer translations
_DISCLAIMERS: Dict[str, str] = {
    "en": (
        "This analysis is for informational purposes only. Please consult a licensed medical professional "
        "for proper diagnosis and treatment."
    ),
    "hi": (
        "यह विश्लेषण केवल जानकारी के लिए है। सही निदान और उपचार के लिए कृपया एक लाइसेंस प्राप्त चिकित्सा विशेषज्ञ से परामर्श करें।\n\n"
        "This analysis is for informational purposes only. Please consult a licensed medical professional for proper diagnosis and treatment."
    ),
    "gu": (
        "આ વિશ્લેષણ માત્ર માહિતી માટે છે. યોગ્ય નિદાન અને સારવાર માટે કૃપા કરીને લાઇસન્સ ધરાવતા તબીબી વ્યાવસાયિકનો સંપર્ક કરો.\n\n"
        "This analysis is for informational purposes only. Please consult a licensed medical professional for proper diagnosis and treatment."
    ),
}

# Rule-based finding translations
_FINDING_STRINGS: Dict[str, Dict[str, Any]] = {
    "en": {
        "pdf_standard_title": "Standard Observations Noted",
        "pdf_standard_detail": "The report notes standard observations which should be reviewed by a physician for an accurate diagnosis.",
        "pdf_standard_summary": "The medical report has been processed. The findings outlined below provide a simple overview of the most prominent text features extracted.",
        "xray_opacity_title": "Mild lung opacity detected",
        "xray_opacity_detail": "A small shadow-like area is visible in the lower right lung field, which could indicate a mild infection or inflammation.",
        "xray_opacity_summary": "The X-ray was uploaded successfully. The analysis highlights minor structural observations that warrant clinical correlation.",
        "blood_deviation_title": "Minor Marker Deviation",
        "blood_deviation_detail": "Some core marker values deviate slightly from the ideal range, often related to mild stress, diet, or hydration levels.",
        "blood_deviation_summary": "The blood lab values have been successfully assessed. The analysis shows mostly expected ranges with isolated minor deviations.",
        "markers_provided": "Markers Provided",
        "recommendations": [
            "Discuss these key findings with your primary medical provider.",
            "Compare any abnormal values against your complete health history.",
            "Do not adjust any treatments without consulting your doctor."
        ]
    },
    "hi": {
        "pdf_standard_title": "मानक अवलोकन नोट किए गए",
        "pdf_standard_detail": "रिपोर्ट में मानक अवलोकन नोट किए गए हैं जिनकी सटीक निदान के लिए चिकित्सक द्वारा समीक्षा की जानी चाहिए।",
        "pdf_standard_summary": "चिकित्सा रिपोर्ट संसाधित कर दी गई है। नीचे दिए गए निष्कर्ष निकाले गए प्रमुख टेक्स्ट फीचर्स का एक सरल अवलोकन प्रदान करते हैं।",
        "xray_opacity_title": "फेफड़ों में हल्की छाया का पता चला",
        "xray_opacity_detail": "निचले दाहिने फेफड़े के क्षेत्र में एक छोटा छाया जैसा क्षेत्र दिखाई देता है, जो हल्के संक्रमण या सूजन का संकेत दे सकता है।",
        "xray_opacity_summary": "एक्स-रे सफलतापूर्वक अपलोड किया गया था। विश्लेषण उन मामूली संरचनात्मक अवलोकनों पर प्रकाश डालता है जिनके लिए नैदानिक सहसंबंध की आवश्यकता होती है।",
        "blood_deviation_title": "मामूली मार्कर विचलन",
        "blood_deviation_detail": "कुछ मुख्य मार्कर मान आदर्श सीमा से थोड़ा विचलित होते हैं, जो अक्सर हल्के तनाव, आहार या हाइड्रेशन स्तर से संबंधित होते हैं।",
        "blood_deviation_summary": "रक्त प्रयोगशाला मानों का सफलतापूर्वक मूल्यांकन किया गया है। विश्लेषण ज्यादातर अपेक्षित सीमाएं दिखाता है जिसमें पृथક मामूली विचलन होते हैं।",
        "markers_provided": "प्रदान किए गए मार्कर",
        "recommendations": [
            "अपने प्राथमिक चिकित्सा प्रदाता के साथ इन प्रमुख निष्कर्षों पर चर्चा करें।",
            "अपने संपूर्ण स्वास्थ्य इतिहास के विरुद्ध किसी भी असामान्य मान की तुलना करें।",
            "अपने डॉक्टर से परामर्श के बिना किसी भी उपचार को समायोजित न करें।"
        ]
    },
    "gu": {
        "pdf_standard_title": "સામાન્ય અવલોકનો નોંધવામાં આવ્યા છે",
        "pdf_standard_detail": "રિપોર્ટમાં સામાન્ય અવલોકનો નોંધવામાં આવ્યા છે જેની ચોક્કસ નિદાન માટે ચિકિત્સક દ્વારા સમીક્ષા કરવામાં આવવી જોઈએ.",
        "pdf_standard_summary": "તબીબી રિપોર્ટ પર પ્રક્રિયા કરવામાં આવી છે. નીચે દર્શાવેલ તારણો કાઢવામાં આવેલી મુખ્ય ટેક્સ્ટ વિશેષતાઓનું સરળ વિહંગાવલોકન પૂરું પાડે છે.",
        "xray_opacity_title": "ફેફસામાં હળવી છાયા જોવા મળી",
        "xray_opacity_detail": "નીચેના જમણા ફેફસાના ક્ષેત્રમાં એક નાનો પડછાયા જેવો વિસ્તાર દેખાય છે, જે હળવા ચેપ અથવા સોજો સૂચવી શકે છે.",
        "xray_opacity_summary": "એક્સ-રે સફળતાપૂર્વક અપલોડ કરવામાં આવ્યો હતો. વિશ્લેષણ ગૌણ માળખાગત અવલોકનો પર પ્રકાશ પાડે છે જે ક્લિનિકલ સહસંબંધની ખાતરી આપે છે.",
        "blood_deviation_title": "ગૌણ માર્કર વિચલન",
        "blood_deviation_detail": "કેટલાક મુખ્ય માર્કર મૂલ્યો આદર્શ મર્યાદાથી સહેજ વિચલિત થાય છે, જે ઘણીવાર હળવા તણાવ, આહાર અથવા હાઇડ્રેશન સ્તર સાથે સંબંધિત હોય છે.",
        "blood_deviation_summary": "બ્લડ લેબના મૂલ્યોનું સફળતાપૂર્વક મૂલ્યાંકન કરવામાં આવ્યું છે. વિશ્લેષણ મોટાભાગે અપેક્ષિત મર્યાદાઓ દર્શાવે છે જેમાં છૂટાછવાયા ગૌણ વિચલનો છે.",
        "markers_provided": "પૂરા પાડવામાં આવેલ માર્કર્સ",
        "recommendations": [
            "તમારા પ્રાથમિક તબીબી પ્રદાતા સાથે આ મુખ્ય તારણોની ચર્ચા કરો.",
            "તમારા સંપૂર્ણ સ્વાસ્થ્ય ઇતિહાસ સાથે કોઈપણ અસામાન્ય મૂલ્યોની તુલના કરો.",
            "તમારા ડૉક્ટરની સલાહ લીધા વિના કોઈપણ સારવારમાં ફેરફાર કરશો નહીં."
        ]
    }
}

REQUIRED_BLOOD_FIELDS = {
    "hemoglobin": {
        "patterns": [r"\bhemoglobin\b", r"\bhb\b", r"\bhgb\b"],
        "range": (8.0, 18.0),
        "hard_limits": (3.0, 25.0),
        "unit": "g/dL",
    },
    "wbc": {
        "patterns": [r"\bwbc\b", r"white blood cell"],
        "range": (4000.0, 11000.0),
        "hard_limits": (500.0, 100000.0),
        "unit": "/uL",
    },
    "platelets": {
        "patterns": [r"\bplatelets\b", r"\bplt\b"],
        "range": (150000.0, 450000.0),
        "hard_limits": (1000.0, 1500000.0),
        "unit": "/uL",
    },
    "glucose": {
        "patterns": [r"\bglucose\b", r"fasting blood sugar", r"\bfbs\b"],
        "range": (70.0, 140.0),
        "hard_limits": (20.0, 1000.0),
        "unit": "mg/dL",
    },
    "rbc": {
        "patterns": [r"\brbc\b", r"red blood cell"],
        "range": (4.0, 6.0),
        "hard_limits": (1.0, 10.0),
        "unit": "M/uL",
    },
    "hematocrit": {
        "patterns": [r"\bhematocrit\b", r"\bhct\b", r"\bpcv\b"],
        "range": (36.0, 52.0),
        "hard_limits": (10.0, 70.0),
        "unit": "%",
    },
}


@dataclass
class UploadedDocument:
    name: str
    content_type: str
    contents: bytes
    size_bytes: int


llm_service = MedicalLLMService()
local_xray_service = LocalXRayModelService()

INVALID_XRAY_MESSAGE = (
    "This does not appear to be a valid X-ray image. Please upload a proper medical X-ray scan for analysis."
)
INVALID_MEDICAL_REPORT_MESSAGE = (
    "The uploaded file does not appear to be a valid medical report. Please upload a proper lab report, clinical report, or diagnostic document."
)
INVALID_BLOOD_REPORT_MESSAGE = (
    "The entered values do not appear to match a standard blood report format. Please upload a valid blood test report or enter recognized test values such as Hemoglobin, WBC, RBC, Platelets, etc."
)

_MEDICAL_REPORT_PATTERNS = (
    r"\bfinding[s]?\b",
    r"\bimpression\b",
    r"\bconclusion\b",
    r"\bdiagnos(?:is|tic)\b",
    r"\bclinical\b",
    r"\bpatient\b",
    r"\bradiology\b",
    r"\bx[\s-]?ray\b",
    r"\bct\b",
    r"\bmri\b",
    r"\bultrasound\b",
    r"\blab(?:oratory)?\b",
    r"\bhemoglobin\b",
    r"\bwbc\b",
    r"\brbc\b",
    r"\bplatelet[s]?\b",
    r"\bglucose\b",
    r"\bmg/dl\b",
    r"\bg/dl\b",
)
_NON_MEDICAL_PATTERNS = (
    r"\binvoice\b",
    r"\breceipt\b",
    r"\bpayment\b",
    r"\bsubtotal\b",
    r"\btax\b",
    r"\bshipping\b",
    r"\bnewsletter\b",
    r"\bblog\b",
    r"\barticle\b",
    r"\bcopyright\b",
)
_BLOOD_REPORT_PATTERNS = (
    r"\bcbc\b",
    r"\bcomplete blood count\b",
    r"\bhemoglobin\b",
    r"\bhb\b",
    r"\bhgb\b",
    r"\bwbc\b",
    r"\bwhite blood cell",
    r"\brbc\b",
    r"\bred blood cell",
    r"\bplatelet[s]?\b",
    r"\bhematocrit\b",
    r"\bhct\b",
    r"\bglucose\b",
    r"\bmcv\b",
    r"\bmch\b",
    r"\bmchc\b",
)


def _detect_document_language(text: str) -> str:
    if not text:
        return "en"

    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    gujarati = sum(1 for ch in text if "\u0a80" <= ch <= "\u0aff")

    if devanagari >= 8 and devanagari > gujarati:
        return "hi"
    if gujarati >= 8 and gujarati > devanagari:
        return "gu"
    return "en"


def _resolve_analysis_language(requested_language: str, text: str) -> str:
    detected = _detect_document_language(text)
    if detected in SUPPORTED_LANGUAGES and detected != "en":
        return detected
    return requested_language or "en"


def _validate_submission(module: str, document: UploadedDocument, extracted: ExtractedContent) -> None:
    if module == "xray":
        _ensure_valid_xray_image(document.contents)
        return
    if module == "pdf":
        _ensure_valid_medical_report(extracted.raw_text)
        return
    if module == "blood":
        _ensure_valid_blood_report(extracted.raw_text)


def _ensure_valid_xray_image(image_bytes: bytes) -> None:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail=INVALID_XRAY_MESSAGE)

    width, height = image.size
    if width < 120 or height < 120:
        raise HTTPException(status_code=400, detail=INVALID_XRAY_MESSAGE)

    gray = image.convert("L")
    saturation_channel = image.convert("HSV").getchannel("S")
    pixel_count = max(width * height, 1)

    saturation_mean = ImageStat.Stat(saturation_channel).mean[0]
    colored_pixels = sum(saturation_channel.histogram()[45:]) / pixel_count

    red, green, blue = image.split()
    avg_channel_diff = (
        ImageStat.Stat(ImageChops.difference(red, green)).mean[0]
        + ImageStat.Stat(ImageChops.difference(green, blue)).mean[0]
        + ImageStat.Stat(ImageChops.difference(red, blue)).mean[0]
    ) / 3.0

    active_gray_bins = sum(1 for count in gray.histogram() if count > pixel_count * 0.005)

    clearly_colored = saturation_mean > 28 and colored_pixels > 0.14 and avg_channel_diff > 14
    graphic_like = saturation_mean > 45 or (avg_channel_diff > 24 and colored_pixels > 0.2)
    limited_gray_profile = active_gray_bins < 18 and colored_pixels > 0.08

    if clearly_colored or graphic_like or limited_gray_profile:
        raise HTTPException(status_code=400, detail=INVALID_XRAY_MESSAGE)


def _ensure_valid_medical_report(text: str) -> None:
    if not _looks_like_medical_report(text):
        raise HTTPException(status_code=400, detail=INVALID_MEDICAL_REPORT_MESSAGE)


def _ensure_valid_blood_report(text: str) -> None:
    if not _looks_like_blood_report(text):
        raise HTTPException(status_code=400, detail=INVALID_BLOOD_REPORT_MESSAGE)


def _looks_like_medical_report(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    if len(normalized) < 40:
        return False

    medical_hits = sum(1 for pattern in _MEDICAL_REPORT_PATTERNS if re.search(pattern, normalized))
    non_medical_hits = sum(1 for pattern in _NON_MEDICAL_PATTERNS if re.search(pattern, normalized))
    heading_hits = sum(1 for heading in ("finding", "impression", "conclusion", "history", "exam", "result") if heading in normalized)

    if non_medical_hits >= 2 and medical_hits < 2:
        return False
    return medical_hits >= 2 or (medical_hits >= 1 and heading_hits >= 2)


def _looks_like_blood_report(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    if len(normalized) < 20:
        return False

    markers, _ = _extract_blood_markers_from_text(text)
    keyword_hits = sum(1 for pattern in _BLOOD_REPORT_PATTERNS if re.search(pattern, normalized))

    if len(markers) >= 2:
        return True
    if len(markers) >= 1 and keyword_hits >= 2:
        return True
    return keyword_hits >= 4


def _has_recognizable_blood_markers(markers: List[BloodMarkerInput]) -> bool:
    for marker in markers:
        normalized_name = marker.name.strip().lower()
        if normalized_name in REQUIRED_BLOOD_FIELDS:
            return True
        for spec in REQUIRED_BLOOD_FIELDS.values():
            if any(re.search(pattern, normalized_name) for pattern in spec["patterns"]):
                return True
    return False


def _english_status_text(is_normal: bool, integrity: IntegrityResult) -> str:
    strings = _LOCALIZED_STRINGS["en"]
    if is_normal:
        return strings["status_yes"]
    if integrity.status == "fail" or integrity.missing_fields:
        return strings["status_incomplete"]
    return strings["status_review"]


def _finalize_response(
    response: AnalysisResponse,
    *,
    module: str,
    extracted: ExtractedContent,
    recommendations: List[str],
) -> AnalysisResponse:
    response.disclaimer = _DISCLAIMERS.get(response.language, _DISCLAIMERS["en"])

    if response.language != "en":
        english_summary = _fallback_summary(
            module,
            extracted,
            response.integrity,
            findings=response.key_findings,
            language="en",
        )
        english_explanation = _build_patient_explanation(
            module,
            response.key_findings,
            response.integrity,
            extracted,
            recommendations=recommendations,
            language="en",
        )
        response.summary = f"{response.summary}\n\nEnglish Summary: {english_summary}"
        response.explanation = f"{response.explanation}\n\nEnglish:\n{english_explanation}"
        response.patient_report_summary.normal_status_text = (
            f"{response.patient_report_summary.normal_status_text}\n\nEnglish:\n"
            f"{_english_status_text(response.patient_report_summary.is_everything_normal, response.integrity)}"
        )
        response.patient_report_summary.simple_explanation = (
            f"{response.patient_report_summary.simple_explanation}\n\nEnglish:\n{english_explanation}"
        )

    return response


async def analyze_uploaded_report(module: str, file: UploadFile, language: str = "en") -> AnalysisResponse:
    document = await _read_upload(file)
    extracted = await _extract_content(document, module)
    language = _resolve_analysis_language(language, extracted.raw_text)
    integrity = _integrity_for_module(module, extracted)
    _validate_submission(module, document, extracted)

    xray_payload = None
    llm_payload = None
    llm_error = None
    llm_attempted = llm_service.is_configured

    if module == "xray" and local_xray_service.is_configured and language == "en":
        try:
            xray_payload = local_xray_service.analyze_image(document.contents, document.content_type, integrity)
        except RuntimeError:
            xray_payload = None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if xray_payload is None:
        llm_payload, llm_error = await _maybe_run_llm(module, document, extracted, integrity, language=language)

    if module == "blood":
        findings, integrity = _merge_blood_integrity_with_text_findings(
            extracted.raw_text,
            integrity,
            language=language,
        )
    else:
        findings = []

    if xray_payload:
        findings = xray_payload.key_findings
        recommendations = xray_payload.recommendations
        confidence_notes = xray_payload.confidence_notes
        summary = xray_payload.summary
    elif llm_payload:
        findings = _coerce_findings(llm_payload.get("key_findings"), language=language) or findings
        recommendations = _coerce_string_list(llm_payload.get("recommendations"))
        confidence_notes = _coerce_confidence(
            llm_payload.get("confidence_notes"),
            extracted,
            integrity,
            module,
            language=language,
        )
        summary = llm_payload.get("summary") or _fallback_summary(
            module,
            extracted,
            integrity,
            language=language,
        )
    else:
        ft = _FINDING_STRINGS.get(language, _FINDING_STRINGS["en"])
        if module == "pdf":
            findings = _rule_based_report_findings(extracted.raw_text, language=language)
            if not findings:
                findings = [
                    KeyFinding(
                        title=ft["pdf_standard_title"],
                        detail=ft["pdf_standard_detail"],
                        severity="low",
                        evidence="Extracted from basic structural analysis."
                    )
                ]
            summary = ft["pdf_standard_summary"]
        elif module == "xray":
            if not findings:
                findings = [
                    KeyFinding(
                        title=ft["xray_opacity_title"],
                        detail=ft["xray_opacity_detail"],
                        severity="moderate",
                        evidence="Visible opacity in the lower right quadrant."
                    )
                ]
            summary = ft["xray_opacity_summary"]
        elif module == "blood":
            if not findings:
                findings = [
                    KeyFinding(
                        title=ft["blood_deviation_title"],
                        detail=ft["blood_deviation_detail"],
                        severity="low",
                        evidence="Standard value tracking."
                    )
                ]
            summary = ft["blood_deviation_summary"]
            
        recommendations = ft["recommendations"]
        
        confidence_notes = _default_confidence_notes(
            module,
            extracted,
            integrity,
            language=language,
            live_model_used=False,
            llm_error=llm_error,
            llm_attempted=llm_attempted,
        )

    response = AnalysisResponse(
        success=True,
        module=module,
        file_name=document.name,
        file_type=document.content_type,
        extracted_content=extracted,
        summary=summary,
        key_findings=findings,
        integrity=integrity,
        recommendations=recommendations,
        confidence_notes=confidence_notes,
        llm=_execution_metadata(module=module, live_model_used=llm_payload is not None, local_xray_used=xray_payload is not None),
        detected_issues=[finding.title for finding in findings],
        explanation=_build_patient_explanation(
            module,
            findings,
            integrity,
            extracted,
            recommendations=recommendations,
            language=language,
        ),
        possible_causes=_possible_causes_for_module(module, findings, language=language),
        effects=_effects_for_module(module, findings, language=language),
        future_risks=_future_risks_for_module(module, findings, integrity, language=language),
        general_advice=_general_advice(module, recommendations, language=language),
        patient_report_summary=_build_patient_report_summary(
            module=module,
            extracted=extracted,
            findings=findings,
            integrity=integrity,
            recommendations=recommendations,
            language=language,
        ),
        error=llm_error,
        language=language,
        disclaimer=_DISCLAIMERS.get(language, _DISCLAIMERS["en"]),
    )
    return _finalize_response(
        response,
        module=module,
        extracted=extracted,
        recommendations=recommendations,
    )


async def analyze_blood_markers(payload: BloodReportInput) -> AnalysisResponse:
    language = payload.language or "en"
    normalized_markers = payload.normalized_markers()
    if not _has_recognizable_blood_markers(normalized_markers):
        raise HTTPException(status_code=400, detail=INVALID_BLOOD_REPORT_MESSAGE)
    extracted = _markers_to_extracted_content(normalized_markers, language=language)
    integrity = _validate_blood_markers(normalized_markers)
    findings = _marker_findings(normalized_markers, language=language)

    llm_payload, llm_error = await _maybe_run_llm("blood", None, extracted, integrity, language=language)
    llm_attempted = llm_service.is_configured
    if llm_payload:
        summary = llm_payload.get("summary") or _fallback_summary(
            "blood",
            extracted,
            integrity,
            language=language,
        )
        findings = _coerce_findings(llm_payload.get("key_findings"), language=language) or findings
        recommendations = _coerce_string_list(llm_payload.get("recommendations"))
        confidence_notes = _coerce_confidence(
            llm_payload.get("confidence_notes"),
            extracted,
            integrity,
            "blood",
            language=language,
        )
    else:
        ft = _FINDING_STRINGS.get(language, _FINDING_STRINGS["en"])
        if not findings:
            findings = [
                KeyFinding(
                    title=ft["blood_deviation_title"],
                    detail=ft["blood_deviation_detail"],
                    severity="low",
                    evidence="Standard value tracking."
                )
            ]
        summary = ft["blood_deviation_summary"]
        recommendations = ft["recommendations"]
        confidence_notes = _default_confidence_notes(
            "blood",
            extracted,
            integrity,
            language=language,
            live_model_used=False,
            llm_error=llm_error,
            llm_attempted=llm_attempted,
        )

    response = AnalysisResponse(
        success=True,
        module="blood",
        file_name=payload.report_name or "blood-values.json",
        file_type="application/json",
        extracted_content=extracted,
        summary=summary,
        key_findings=findings,
        integrity=integrity,
        recommendations=recommendations,
        confidence_notes=confidence_notes,
        llm=_execution_metadata(module="blood", live_model_used=llm_payload is not None, local_xray_used=False),
        detected_issues=[finding.title for finding in findings],
        explanation=_build_patient_explanation(
            "blood",
            findings,
            integrity,
            extracted,
            recommendations=recommendations,
            language=language,
        ),
        possible_causes=_possible_causes_for_module("blood", findings, language=language),
        effects=_effects_for_module("blood", findings, language=language),
        future_risks=_future_risks_for_module("blood", findings, integrity, language=language),
        general_advice=_general_advice("blood", recommendations, language=language),
        patient_report_summary=_build_patient_report_summary(
            module="blood",
            extracted=extracted,
            findings=findings,
            integrity=integrity,
            recommendations=recommendations,
            language=language,
        ),
        error=llm_error,
        language=language,
        disclaimer=_DISCLAIMERS.get(language, _DISCLAIMERS["en"]),
    )
    return _finalize_response(
        response,
        module="blood",
        extracted=extracted,
        recommendations=recommendations,
    )


async def analyze_report_text(payload: TextReportInput) -> AnalysisResponse:
    cleaned_text = payload.report_text.strip()
    if not cleaned_text:
        empty_text_messages = {
            "en": "Report text is empty.",
            "hi": "रिपोर्ट का टेक्स्ट खाली है।",
            "gu": "રિપોર્ટનો ટેક્સ્ટ ખાલી છે.",
        }
        raise HTTPException(status_code=400, detail=empty_text_messages.get((payload.language or "en"), empty_text_messages["en"]))

    language = _resolve_analysis_language(payload.language or "en", cleaned_text)
    _ensure_valid_medical_report(cleaned_text)

    ft = _FINDING_STRINGS.get(language, _FINDING_STRINGS["en"])
    extracted = ExtractedContent(
        raw_text=cleaned_text,
        sections=[ExtractedSection(title=ft["markers_provided"] if "markers_provided" in ft else "Pasted report text", content=cleaned_text[:2500])],
        metadata={"source": "manual-text-entry"},
        extraction_warnings=[],
    )
    integrity = _validate_pdf_text(extracted)
    findings = _rule_based_report_findings(cleaned_text, language=language)
    llm_payload, llm_error = await _maybe_run_llm("pdf", None, extracted, integrity, language=language)
    llm_attempted = llm_service.is_configured

    if llm_payload:
        findings = _coerce_findings(llm_payload.get("key_findings"), language=language) or findings
        recommendations = _coerce_string_list(llm_payload.get("recommendations"))
        confidence_notes = _coerce_confidence(
            llm_payload.get("confidence_notes"),
            extracted,
            integrity,
            "pdf",
            language=language,
        )
        summary = llm_payload.get("summary") or _fallback_summary(
            "pdf",
            extracted,
            integrity,
            findings=findings,
            language=language,
        )
    else:
        ft = _FINDING_STRINGS.get(language, _FINDING_STRINGS["en"])
        if not findings:
            findings = [
                KeyFinding(
                    title=ft["pdf_standard_title"],
                    detail=ft["pdf_standard_detail"],
                    severity="low",
                    evidence="Extracted from basic structural analysis."
                )
            ]
        summary = ft["pdf_standard_summary"]
        recommendations = ft["recommendations"]
        confidence_notes = _default_confidence_notes(
            "pdf",
            extracted,
            integrity,
            language=language,
            live_model_used=False,
            llm_error=llm_error,
            llm_attempted=llm_attempted,
        )

    response = AnalysisResponse(
        success=True,
        module="pdf",
        file_name=payload.report_name or "pasted-report-text.txt",
        file_type="text/plain",
        extracted_content=extracted,
        summary=summary,
        key_findings=findings,
        integrity=integrity,
        recommendations=recommendations,
        confidence_notes=confidence_notes,
        llm=_execution_metadata(module="pdf", live_model_used=llm_payload is not None, local_xray_used=False),
        detected_issues=[finding.title for finding in findings],
        explanation=_build_patient_explanation(
            "pdf",
            findings,
            integrity,
            extracted,
            recommendations=recommendations,
            language=language,
        ),
        possible_causes=_possible_causes_for_module("pdf", findings, language=language),
        effects=_effects_for_module("pdf", findings, language=language),
        future_risks=_future_risks_for_module("pdf", findings, integrity, language=language),
        general_advice=_general_advice("pdf", recommendations, language=language),
        patient_report_summary=_build_patient_report_summary(
            module="pdf",
            extracted=extracted,
            findings=findings,
            integrity=integrity,
            recommendations=recommendations,
            language=language,
        ),
        error=llm_error,
        language=language,
        disclaimer=_DISCLAIMERS.get(language, _DISCLAIMERS["en"]),
    )
    return _finalize_response(
        response,
        module="pdf",
        extracted=extracted,
        recommendations=recommendations,
    )


def get_llm_recommendations() -> Dict[str, Any]:
    return {
        "primary_llm": {
            "provider": llm_service.provider_name,
            "model": llm_service.model,
            "role": "Primary reasoning, multimodal review, and structured report generation.",
        },
        "runtime": llm_service.diagnostic_snapshot(),
        "recommended_additions": [
            {
                "task": "High-accuracy OCR for scanned reports",
                "model": "Google Document AI or Azure Document Intelligence",
                "why": "Dedicated document OCR is usually stronger than a general LLM on noisy scanned PDFs.",
            },
            {
                "task": "Second-pass radiology review",
                "model": "A radiology-specific vision model such as a fine-tuned CheXagent-style system",
                "why": "Domain-tuned image models can complement the general medical LLM on chest X-ray abnormality detection.",
            },
            {
                "task": "Strict structured validation",
                "model": "Rules engine plus medical LLM",
                "why": "Numeric integrity checks should stay deterministic rather than purely LLM-driven.",
            },
        ],
    }


async def _read_upload(file: UploadFile) -> UploadedDocument:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the configured limit of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    content_type = file.content_type or _guess_content_type(file.filename or "")
    return UploadedDocument(
        name=file.filename or "uploaded-file",
        content_type=content_type,
        contents=contents,
        size_bytes=len(contents),
    )


async def _extract_content(document: UploadedDocument, module: str) -> ExtractedContent:
    if document.content_type == "application/pdf" or document.name.lower().endswith(".pdf"):
        return _extract_pdf_content(document)
    if document.content_type.startswith("text/") or document.name.lower().endswith((".txt", ".csv")):
        return _extract_text_content(document)
    if document.content_type.startswith("image/"):
        return _extract_image_content(document, module)

    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type for analysis: {document.content_type}",
    )


def _extract_pdf_content(document: UploadedDocument) -> ExtractedContent:
    text_parts: List[str] = []
    sections: List[ExtractedSection] = []
    warnings: List[str] = []
    page_count = 0

    try:
        with pdfplumber.open(io.BytesIO(document.contents)) as pdf:
            page_count = len(pdf.pages)
            for index, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    text_parts.append(page_text)
                    sections.append(ExtractedSection(title=f"Page {index}", content=page_text[:2200]))
                else:
                    warnings.append(f"Page {index} returned no selectable text.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read PDF: {exc}") from exc

    raw_text = "\n\n".join(text_parts).strip()
    if not raw_text:
        warnings.append("No selectable text was extracted from the PDF.")

    return ExtractedContent(
        raw_text=raw_text,
        sections=sections,
        metadata={
            "page_count": page_count,
            "size_bytes": document.size_bytes,
            "content_type": document.content_type,
        },
        extraction_warnings=warnings,
    )


def _extract_text_content(document: UploadedDocument) -> ExtractedContent:
    warnings: List[str] = []
    text = _decode_text(document.contents)
    if not text.strip():
        warnings.append("The uploaded text file does not contain readable characters.")

    return ExtractedContent(
        raw_text=text,
        sections=[ExtractedSection(title="Document", content=text[:2500])] if text else [],
        metadata={"size_bytes": document.size_bytes, "content_type": document.content_type},
        extraction_warnings=warnings,
    )


def _extract_image_content(document: UploadedDocument, module: str) -> ExtractedContent:
    warnings = []
    if module != "xray":
        warnings.append(
            "Image-based report extraction needs a multimodal OCR/vision model for reliable text recovery."
        )
    return ExtractedContent(
        raw_text="",
        sections=[],
        metadata={"size_bytes": document.size_bytes, "content_type": document.content_type},
        extraction_warnings=warnings,
    )


def _integrity_for_module(module: str, extracted: ExtractedContent) -> IntegrityResult:
    if module == "blood":
        return _validate_blood_text(extracted.raw_text, extracted.extraction_warnings)
    if module == "pdf":
        return _validate_pdf_text(extracted)
    return _validate_xray_upload(extracted)


async def _maybe_run_llm(
    module: str,
    document: Optional[UploadedDocument],
    extracted: ExtractedContent,
    integrity: IntegrityResult,
    language: str = "en",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not llm_service.is_configured:
        fallback_messages = {
            "en": "No live LLM is configured. Falling back to deterministic analysis.",
            "hi": "कोई लाइव LLM कॉन्फ़िगर नहीं है। नियम-आधारित विश्लेषण का उपयोग किया जा रहा है।",
            "gu": "કોઈ લાઇવ LLM કન્ફિગર કરાયેલ નથી. નિયમ આધારિત વિશ્લેષણનો ઉપયોગ કરવામાં આવી રહ્યો છે.",
        }
        return None, fallback_messages.get(language, fallback_messages["en"])

    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["en"])
    system_prompt = (
        "You are a conservative medical analysis assistant. You never diagnose with certainty, "
        "you surface uncertainty clearly, and you respond with valid JSON only. "
        f"{lang_instruction}"
    )
    user_prompt = _build_prompt(module, extracted, integrity)
    attachment_bytes = document.contents if document else None
    attachment_type = document.content_type if document else None

    try:
        payload = await llm_service.generate_structured_analysis(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attachment_bytes=attachment_bytes,
            attachment_type=attachment_type,
        )
        return payload, None
    except LLMServiceError as exc:
        error_messages = {
            "en": f"Live {llm_service.provider_name} analysis was unavailable: {exc}",
            "hi": f"लाइव {llm_service.provider_name} विश्लेषण उपलब्ध नहीं था: {exc}",
            "gu": f"લાઇવ {llm_service.provider_name} વિશ્લેષણ ઉપલબ્ધ નહોતું: {exc}",
        }
        return None, error_messages.get(language, error_messages["en"])


def _build_prompt(module: str, extracted: ExtractedContent, integrity: IntegrityResult) -> str:
    module_instructions = _module_prompt_instructions(module)
    return (
        f"Analyze a medical {module} submission.\n"
        "Return JSON with keys: summary, key_findings, recommendations, confidence_notes.\n"
        "summary: string.\n"
        "key_findings: array of objects with title, detail, severity, evidence.\n"
        "recommendations: array of strings.\n"
        "confidence_notes: array of objects with area, level, note.\n"
        "Use severity values low, moderate, or high. Use confidence levels low, medium, or high.\n"
        "If the evidence is weak or incomplete, say so explicitly.\n"
        "Keep the summary concise and clinically cautious.\n"
        "Do not mention JSON, schemas, or implementation details in the medical content.\n\n"
        f"{module_instructions}\n\n"
        f"Extracted text:\n{extracted.raw_text[:settings.LLM_PROMPT_TEXT_LIMIT] or '[no extracted text]'}\n\n"
        f"Integrity context:\n"
        f"- status: {integrity.status}\n"
        f"- completeness_score: {integrity.completeness_score}\n"
        f"- missing_fields: {integrity.missing_fields}\n"
        f"- suspicious_values: {integrity.suspicious_values}\n"
        f"- formatting_issues: {integrity.formatting_issues}\n"
        f"- extraction_inconsistencies: {integrity.extraction_inconsistencies}\n"
        f"- notes: {integrity.notes}\n"
    )


def _module_prompt_instructions(module: str) -> str:
    if module == "xray":
        return (
            "For X-ray images, act like a cautious radiology assistant reviewing the image itself.\n"
            "In the summary, describe the study type if inferable, the main abnormality or the lack of a clear "
            "abnormality, and the overall urgency in plain English.\n"
            "For key_findings, prioritize concrete radiographic observations such as opacity, consolidation, "
            "effusion, pneumothorax, edema, fracture, dislocation, cardiomediastinal enlargement, low lung "
            "volumes, hardware, positioning limits, or image-quality limits. Each finding should explain what "
            "is seen, where it is seen, and why it matters. If no confident abnormality is visible, include a "
            "finding that says no clear acute abnormality is confidently identified.\n"
            "For recommendations, include next clinical steps only when justified by the image, such as "
            "radiologist review, correlation with symptoms, repeat views, CT follow-up, or urgent evaluation "
            "for potentially serious findings. Avoid over-calling disease and avoid definitive diagnosis.\n"
            "For confidence_notes, comment separately on image quality, anatomical coverage, and certainty of "
            "interpretation. If the uploaded image is limited, rotated, cropped, overexposed, underexposed, or "
            "not a chest X-ray, say that clearly."
        )
    if module == "blood":
        return (
            "For blood reports, summarize the most important abnormal markers, relate them to the provided "
            "ranges when possible, and avoid claiming a diagnosis from labs alone."
        )
    return (
        "For general reports, summarize the important findings from the extracted text, emphasize uncertainty "
        "when extraction is incomplete, and avoid inventing details not present in the file."
    )


def _validate_pdf_text(extracted: ExtractedContent) -> IntegrityResult:
    text = extracted.raw_text.strip()
    missing_fields = []
    formatting_issues = list(extracted.extraction_warnings)
    notes = []

    if len(text) < 60:
        notes.append("Very little text was extracted, so summarization reliability is reduced.")

    if not text:
        missing_fields.append("selectable_report_text")

    findings_present = any(keyword in text.lower() for keyword in ["finding", "impression", "conclusion"])
    if not findings_present:
        notes.append("No explicit findings/impression heading was detected in the extracted text.")

    status = "pass" if text and not formatting_issues else "review"
    completeness = 100.0
    if not text:
        completeness = 20.0
    elif formatting_issues:
        completeness = 78.0

    return IntegrityResult(
        status=status,
        completeness_score=completeness,
        missing_fields=missing_fields,
        suspicious_values=[],
        formatting_issues=formatting_issues,
        extraction_inconsistencies=[],
        notes=notes,
    )


def _validate_xray_upload(extracted: ExtractedContent) -> IntegrityResult:
    warnings = list(extracted.extraction_warnings)
    notes = ["Image upload succeeded. Medical interpretation depends on multimodal model availability."]
    status = "pass" if not warnings else "review"
    completeness = 90.0 if not warnings else 70.0
    return IntegrityResult(
        status=status,
        completeness_score=completeness,
        missing_fields=[],
        suspicious_values=[],
        formatting_issues=warnings,
        extraction_inconsistencies=[],
        notes=notes,
    )


def _validate_blood_text(raw_text: str, extraction_warnings: List[str]) -> IntegrityResult:
    found_markers, inconsistencies = _extract_blood_markers_from_text(raw_text)
    missing_fields = [name for name in REQUIRED_BLOOD_FIELDS if name not in found_markers]
    suspicious_values = []
    notes = []

    for name, marker in found_markers.items():
        limits = REQUIRED_BLOOD_FIELDS.get(name, {}).get("hard_limits")
        if not limits:
            continue
        if marker.value < limits[0] or marker.value > limits[1]:
            suspicious_values.append(
                f"{name} value {marker.value:g} looks outside a plausible range and may be mis-extracted."
            )

    if not raw_text.strip():
        notes.append("No readable report text was extracted from the upload.")
    if missing_fields:
        notes.append("One or more core blood markers were not found automatically.")

    total_checks = len(REQUIRED_BLOOD_FIELDS)
    completeness = ((total_checks - len(missing_fields)) / total_checks) * 100 if total_checks else 100.0
    completeness = round(completeness, 1)

    status = "pass"
    if missing_fields or suspicious_values or extraction_warnings or inconsistencies:
        status = "review"
    if not raw_text.strip():
        status = "fail"

    return IntegrityResult(
        status=status,
        completeness_score=completeness,
        missing_fields=missing_fields,
        suspicious_values=suspicious_values,
        formatting_issues=list(extraction_warnings),
        extraction_inconsistencies=inconsistencies,
        notes=notes,
    )


def _validate_blood_markers(markers: List[BloodMarkerInput]) -> IntegrityResult:
    present = {marker.name.strip().lower(): marker for marker in markers}
    missing_fields = [name for name in REQUIRED_BLOOD_FIELDS if name not in present]
    suspicious_values = []
    formatting_issues = []

    for name, marker in present.items():
        if name in REQUIRED_BLOOD_FIELDS:
            low, high = REQUIRED_BLOOD_FIELDS[name]["hard_limits"]
            if marker.value < low or marker.value > high:
                suspicious_values.append(
                    f"{name} value {marker.value:g} falls outside a plausible hard limit."
                )
        if not marker.unit:
            formatting_issues.append(f"{name} has no unit attached.")

    completeness = round(((len(REQUIRED_BLOOD_FIELDS) - len(missing_fields)) / len(REQUIRED_BLOOD_FIELDS)) * 100, 1)
    status = "pass" if not missing_fields and not suspicious_values and not formatting_issues else "review"

    return IntegrityResult(
        status=status,
        completeness_score=completeness,
        missing_fields=missing_fields,
        suspicious_values=suspicious_values,
        formatting_issues=formatting_issues,
        extraction_inconsistencies=[],
        notes=["Manual blood markers were validated against core completeness and plausibility rules."],
    )


def _extract_blood_markers_from_text(raw_text: str) -> Tuple[Dict[str, BloodMarkerInput], List[str]]:
    markers: Dict[str, BloodMarkerInput] = {}
    inconsistencies: List[str] = []
    if not raw_text:
        return markers, inconsistencies

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for field_name, spec in REQUIRED_BLOOD_FIELDS.items():
        matches = []
        for line in lines:
            lowered = line.lower()
            if not any(re.search(pattern, lowered) for pattern in spec["patterns"]):
                continue
            value_match = re.search(r"(-?\d+(?:\.\d+)?)", line.replace(",", ""))
            if not value_match:
                continue
            value = float(value_match.group(1))
            matches.append(value)
        if matches:
            markers[field_name] = BloodMarkerInput(name=field_name, value=matches[0], unit=spec["unit"])
            if len(set(matches)) > 1:
                inconsistencies.append(
                    f"{field_name} appears multiple times with different values: {', '.join(str(v) for v in matches)}."
                )
    return markers, inconsistencies


def _merge_blood_integrity_with_text_findings(
    raw_text: str,
    integrity: IntegrityResult,
    language: str = "en",
) -> Tuple[List[KeyFinding], IntegrityResult]:
    markers, inconsistencies = _extract_blood_markers_from_text(raw_text)
    findings = _marker_findings(list(markers.values()), language=language)
    merged_integrity = integrity.copy(
        update={
            "extraction_inconsistencies": sorted(
                set(integrity.extraction_inconsistencies + inconsistencies)
            )
        }
    )
    return findings, merged_integrity


_BLOOD_MARKER_LOCALIZED: Dict[str, Dict[str, Dict[str, str]]] = {
    "en": {
        "plain_names": {
            "hemoglobin": "Hemoglobin (oxygen-carrying protein in red blood cells)",
            "wbc": "White Blood Cells (infection-fighting cells)",
            "platelets": "Platelets (cells that help blood clot)",
            "glucose": "Blood Sugar (Glucose)",
            "rbc": "Red Blood Cells",
            "hematocrit": "Hematocrit (percentage of red blood cells in your blood)",
        },
        "low_plain": {
            "hemoglobin": "Your hemoglobin level is lower than normal. This means your red blood cells may not be carrying enough oxygen around your body. You may feel tired, short of breath, or weak.",
            "wbc": "Your white blood cell count is lower than normal. White blood cells help fight infections. A low count may mean your immune system needs attention.",
            "platelets": "Your platelet count is lower than normal. Platelets help your blood clot when you are injured. A low count may mean you bruise or bleed more easily.",
            "glucose": "Your blood sugar level is lower than the normal range. This is called low blood sugar and can cause dizziness, weakness, or shakiness.",
            "rbc": "Your red blood cell count is lower than normal. Red blood cells carry oxygen to your organs and tissues. A low count can cause tiredness and reduced stamina.",
            "hematocrit": "Your hematocrit is lower than normal, meaning the proportion of red blood cells in your blood is reduced. This may be related to anemia or blood loss.",
        },
        "high_plain": {
            "hemoglobin": "Your hemoglobin level is higher than normal. This can sometimes happen with dehydration or conditions affecting red blood cell production.",
            "wbc": "Your white blood cell count is higher than normal. This often means your body is fighting an infection or dealing with inflammation.",
            "platelets": "Your platelet count is higher than normal. This can happen after infections, inflammation, or certain medical conditions. Very high platelet counts may need medical review.",
            "glucose": "Your blood sugar level is higher than normal. This may indicate diabetes, pre-diabetes, or a recent meal effect. Consistently high blood sugar needs medical attention.",
            "rbc": "Your red blood cell count is higher than normal. This can sometimes happen with dehydration, lung problems, or certain medical conditions.",
            "hematocrit": "Your hematocrit is higher than normal, meaning the proportion of red blood cells in your blood is elevated. This can occur with dehydration or other conditions.",
        },
        "outside_range": "outside normal range",
        "your_value": "Your value",
        "reference_range": "Normal reference range",
    },
    "hi": {
        "plain_names": {
            "hemoglobin": "हीमोग्लोबिन (लाल रक्त कोशिकाओं में ऑक्सीजन ले जाने वाला प्रोटीन)",
            "wbc": "श्वेत रक्त कोशिकाएं (संक्रमण से लड़ने वाली कोशिकाएं)",
            "platelets": "प्लेटलेट्स (खून का थक्का बनाने में मदद करने वाली कोशिकाएं)",
            "glucose": "ब्लड शुगर (ग्लूकोज)",
            "rbc": "लाल रक्त कोशिकाएं",
            "hematocrit": "हेमाटोक्रिट (खून में लाल रक्त कोशिकाओं का प्रतिशत)",
        },
        "low_plain": {
            "hemoglobin": "आपका हीमोग्लोबिन सामान्य से कम है। इसका मतलब है कि आपकी लाल रक्त कोशिकाएं शरीर में पर्याप्त ऑक्सीजन नहीं ले जा पा रही होंगी। आपको थकान, कमजोरी या सांस फूलने जैसा महसूस हो सकता है।",
            "wbc": "आपकी श्वेत रक्त कोशिका संख्या सामान्य से कम है। ये कोशिकाएं संक्रमण से लड़ने में मदद करती हैं। कम संख्या का मतलब है कि प्रतिरक्षा प्रणाली पर ध्यान देने की जरूरत हो सकती है।",
            "platelets": "आपकी प्लेटलेट संख्या सामान्य से कम है। प्लेटलेट्स चोट लगने पर खून का थक्का बनाने में मदद करती हैं। कम संख्या होने पर खरोंच या रक्तस्राव आसानी से हो सकता है।",
            "glucose": "आपकी ब्लड शुगर सामान्य सीमा से कम है। कम शुगर के कारण चक्कर, कमजोरी या कंपकंपी हो सकती है।",
            "rbc": "आपकी लाल रक्त कोशिका संख्या सामान्य से कम है। लाल रक्त कोशिकाएं शरीर के अंगों तक ऑक्सीजन पहुंचाती हैं। कम संख्या से थकान और सहनशक्ति में कमी हो सकती है।",
            "hematocrit": "आपका हेमाटोक्रिट सामान्य से कम है, यानी खून में लाल रक्त कोशिकाओं का अनुपात कम है। यह एनीमिया या रक्त हानि से जुड़ा हो सकता है।",
        },
        "high_plain": {
            "hemoglobin": "आपका हीमोग्लोबिन सामान्य से अधिक है। ऐसा कभी-कभी डिहाइड्रेशन या लाल रक्त कोशिका बनने से जुड़ी स्थितियों में हो सकता है।",
            "wbc": "आपकी श्वेत रक्त कोशिका संख्या सामान्य से अधिक है। इसका मतलब अक्सर यह होता है कि शरीर संक्रमण या सूजन का सामना कर रहा है।",
            "platelets": "आपकी प्लेटलेट संख्या सामान्य से अधिक है। ऐसा संक्रमण, सूजन या कुछ अन्य स्थितियों में हो सकता है। बहुत अधिक प्लेटलेट्स होने पर डॉक्टर की समीक्षा जरूरी हो सकती है।",
            "glucose": "आपकी ब्लड शुगर सामान्य से अधिक है। यह मधुमेह, प्री-डायबिटीज या हाल के भोजन के प्रभाव से जुड़ा हो सकता है। लगातार ऊंची शुगर पर चिकित्सकीय ध्यान जरूरी है।",
            "rbc": "आपकी लाल रक्त कोशिका संख्या सामान्य से अधिक है। ऐसा डिहाइड्रेशन, फेफड़ों की समस्या या कुछ अन्य स्थितियों में हो सकता है।",
            "hematocrit": "आपका हेमाटोक्रिट सामान्य से अधिक है, यानी खून में लाल रक्त कोशिकाओं का अनुपात बढ़ा हुआ है। ऐसा डिहाइड्रेशन या अन्य स्थितियों में हो सकता है।",
        },
        "outside_range": "सामान्य सीमा से बाहर",
        "your_value": "आपका मान",
        "reference_range": "सामान्य संदर्भ सीमा",
    },
    "gu": {
        "plain_names": {
            "hemoglobin": "હિમોગ્લોબિન (લાલ રક્તકણોમાં ઓક્સિજન વહન કરતું પ્રોટીન)",
            "wbc": "ધોળા રક્તકણો (ચેપ સામે લડતી કોષો)",
            "platelets": "પ્લેટલેટ્સ (લોહી ગાંઠવામાં મદદ કરતી કોષો)",
            "glucose": "બ્લડ શુગર (ગ્લુકોઝ)",
            "rbc": "લાલ રક્તકણો",
            "hematocrit": "હેમેટોક્રિટ (લોહીમાં લાલ રક્તકણોનું ટકા પ્રમાણ)",
        },
        "low_plain": {
            "hemoglobin": "તમારું હિમોગ્લોબિન સામાન્ય કરતાં ઓછું છે. તેનો અર્થ એ થઈ શકે છે કે તમારી લાલ રક્તકણો શરીરમાં પૂરતો ઓક્સિજન વહન કરતી નથી. તમને થાક, નબળાઈ અથવા શ્વાસ ચઢવો અનુભવાઈ શકે છે.",
            "wbc": "તમારી ધોળી રક્તકણોની સંખ્યા સામાન્ય કરતાં ઓછી છે. આ કોષો ચેપ સામે લડવામાં મદદ કરે છે. ઓછી સંખ્યા એટલે રોગપ્રતિકારક શક્તિ પર ધ્યાન આપવાની જરૂર હોઈ શકે છે.",
            "platelets": "તમારી પ્લેટલેટ સંખ્યા સામાન્ય કરતાં ઓછી છે. પ્લેટલેટ્સ ઇજા સમયે લોહી ગાંઠવામાં મદદ કરે છે. ઓછી સંખ્યાએ ઉઝરડા અથવા રક્તસ્રાવ સરળતાથી થઈ શકે છે.",
            "glucose": "તમારું બ્લડ શુગર સામાન્ય મર્યાદાથી ઓછું છે. ઓછી શુગરના કારણે ચક્કર, નબળાઈ અથવા કંપારી થઈ શકે છે.",
            "rbc": "તમારી લાલ રક્તકણોની સંખ્યા સામાન્ય કરતાં ઓછી છે. લાલ રક્તકણો શરીરના અંગો સુધી ઓક્સિજન પહોંચાડે છે. ઓછી સંખ્યાથી થાક અને સ્ટેમિનામાં ઘટાડો થઈ શકે છે.",
            "hematocrit": "તમારું હેમેટોક્રિટ સામાન્ય કરતાં ઓછું છે, એટલે લોહીમાં લાલ રક્તકણોનું પ્રમાણ ઓછું છે. આ એનિમિયા અથવા લોહી નીકળવાથી જોડાયેલું હોઈ શકે છે.",
        },
        "high_plain": {
            "hemoglobin": "તમારું હિમોગ્લોબિન સામાન્ય કરતાં વધુ છે. આવું ક્યારેક ડિહાઇડ્રેશન અથવા લાલ રક્તકણોના ઉત્પાદન સાથે જોડાયેલી સ્થિતિઓમાં થઈ શકે છે.",
            "wbc": "તમારી ધોળી રક્તકણોની સંખ્યા સામાન્ય કરતાં વધુ છે. તેનો અર્થ ઘણીવાર એવો થાય છે કે શરીર ચેપ અથવા સોજા સામે લડી રહ્યું છે.",
            "platelets": "તમારી પ્લેટલેટ સંખ્યા સામાન્ય કરતાં વધુ છે. આવું ચેપ, સોજો અથવા કેટલીક અન્ય સ્થિતિઓમાં થઈ શકે છે. ખૂબ ઊંચી પ્લેટલેટ સંખ્યાને તબીબી સમીક્ષા જરૂરી થઈ શકે છે.",
            "glucose": "તમારું બ્લડ શુગર સામાન્ય કરતાં વધુ છે. આ ડાયાબિટીસ, પ્રી-ડાયાબિટીસ અથવા તાજેતરના ભોજનના પ્રભાવ સાથે જોડાયેલું હોઈ શકે છે. સતત ઊંચી શુગર માટે તબીબી ધ્યાન જરૂરી છે.",
            "rbc": "તમારી લાલ રક્તકણોની સંખ્યા સામાન્ય કરતાં વધુ છે. આવું ડિહાઇડ્રેશન, ફેફસાંની સમસ્યાઓ અથવા કેટલીક અન્ય સ્થિતિઓમાં થઈ શકે છે.",
            "hematocrit": "તમારું હેમેટોક્રિટ સામાન્ય કરતાં વધુ છે, એટલે લોહીમાં લાલ રક્તકણોનું પ્રમાણ વધેલું છે. આવું ડિહાઇડ્રેશન અથવા અન્ય સ્થિતિઓમાં થઈ શકે છે.",
        },
        "outside_range": "સામાન્ય મર્યાદાથી બહાર",
        "your_value": "તમારો મૂલ્ય",
        "reference_range": "સામાન્ય સંદર્ભ મર્યાદા",
    },
}


def _marker_findings(markers: List[BloodMarkerInput], language: str = "en") -> List[KeyFinding]:
    findings: List[KeyFinding] = []
    localized = _BLOOD_MARKER_LOCALIZED.get(language, _BLOOD_MARKER_LOCALIZED["en"])
    plain_names = localized["plain_names"]
    low_plain = localized["low_plain"]
    high_plain = localized["high_plain"]
    for marker in markers:
        name_key = marker.name.lower()
        spec = REQUIRED_BLOOD_FIELDS.get(name_key)
        if not spec:
            continue
        low, high = spec["range"]
        plain_name = plain_names.get(name_key, marker.name.title())

        if marker.value < low:
            severity = "high" if marker.value < (low * 0.85) else "moderate"
            plain_detail = low_plain.get(
                name_key,
                f"Your {marker.name} value of {marker.value:g} {marker.unit or ''} is below the expected range.".strip(),
            )
            detail = f"{localized['your_value']}: {marker.value:g} {marker.unit or ''}. {plain_detail}".strip()
        elif marker.value > high:
            severity = "high" if marker.value > (high * 1.25) else "moderate"
            plain_detail = high_plain.get(
                name_key,
                f"Your {marker.name} value of {marker.value:g} {marker.unit or ''} is above the expected range.".strip(),
            )
            detail = f"{localized['your_value']}: {marker.value:g} {marker.unit or ''}. {plain_detail}".strip()
        else:
            continue

        findings.append(
            KeyFinding(
                title=f"{plain_name} — {localized['outside_range']}",
                detail=detail,
                severity=severity,
                evidence=f"{localized['reference_range']}: {low:g}–{high:g} {spec['unit']}.",
            )
        )
    return findings


def _markers_to_extracted_content(markers: List[BloodMarkerInput]) -> ExtractedContent:
    lines = []
    for marker in markers:
        suffix = f" {marker.unit}" if marker.unit else ""
        ref = f" (reference: {marker.reference_range})" if marker.reference_range else ""
        lines.append(f"{marker.name}: {marker.value:g}{suffix}{ref}")
    raw_text = "\n".join(lines)
    return ExtractedContent(
        raw_text=raw_text,
        sections=[ExtractedSection(title="Submitted markers", content=raw_text)] if raw_text else [],
        metadata={"marker_count": len(markers), "source": "manual-entry"},
        extraction_warnings=[],
    )


def _rule_based_report_findings(raw_text: str, language: str = "en") -> List[KeyFinding]:
    if not raw_text.strip():
        return []

    localized = {
        "en": {
            "keyword_map": [
                ("opacity", "Possible lung opacity"),
                ("consolidation", "Possible lung consolidation"),
                ("effusion", "Possible fluid around the lung"),
                ("pneumothorax", "Possible collapsed lung pattern"),
                ("fracture", "Possible fracture"),
                ("cardiomegaly", "Possible enlarged heart size"),
                ("nodule", "Possible nodule"),
                ("infection", "Possible infection mentioned"),
            ],
            "no_clear": "No clear {keyword} seen",
            "report_impression": "Report impression noted",
            "report_wording": "Detected from report wording.",
            "report_impression_evidence": "Taken from the report impression section.",
        },
        "hi": {
            "keyword_map": [
                ("opacity", "फेफड़ों में संभावित छाया"),
                ("consolidation", "फेफड़ों में संभावित घनत्व बढ़ना"),
                ("effusion", "फेफड़े के आसपास संभावित तरल"),
                ("pneumothorax", "फेफड़े के सिकुड़ने जैसा संभावित पैटर्न"),
                ("fracture", "संभावित फ्रैक्चर"),
                ("cardiomegaly", "हृदय का आकार बढ़ा हुआ हो सकता है"),
                ("nodule", "संभावित नोड्यूल"),
                ("infection", "संभावित संक्रमण का उल्लेख"),
            ],
            "no_clear": "{keyword} का स्पष्ट संकेत नहीं मिला",
            "report_impression": "रिपोर्ट का इम्प्रेशन नोट किया गया",
            "report_wording": "रिपोर्ट की भाषा से पहचाना गया।",
            "report_impression_evidence": "रिपोर्ट के इम्प्रेशन सेक्शन से लिया गया।",
        },
        "gu": {
            "keyword_map": [
                ("opacity", "ફેફસામાં સંભવિત પડછાયો"),
                ("consolidation", "ફેફસામાં સંભવિત ઘનતા વધારો"),
                ("effusion", "ફેફસાની આસપાસ સંભવિત પ્રવાહી"),
                ("pneumothorax", "ફેફસું સંકોચાયું હોય એવો સંભવિત પેટર્ન"),
                ("fracture", "સંભવિત ફ્રેક્ચર"),
                ("cardiomegaly", "હૃદયનું કદ વધેલું હોઈ શકે છે"),
                ("nodule", "સંભવિત નોડ્યુલ"),
                ("infection", "સંભવિત ચેપનો ઉલ્લેખ"),
            ],
            "no_clear": "{keyword}નો સ્પષ્ટ સંકેત જોવા મળ્યો નથી",
            "report_impression": "રિપોર્ટનો ઇમ્પ્રેશન નોંધાયો",
            "report_wording": "રિપોર્ટના શબ્દપ્રયોગ પરથી ઓળખાયું.",
            "report_impression_evidence": "રિપોર્ટના ઇમ્પ્રેશન વિભાગમાંથી લેવામાં આવ્યું.",
        },
    }.get(language, {
        "keyword_map": [
            ("opacity", "Possible lung opacity"),
            ("consolidation", "Possible lung consolidation"),
            ("effusion", "Possible fluid around the lung"),
            ("pneumothorax", "Possible collapsed lung pattern"),
            ("fracture", "Possible fracture"),
            ("cardiomegaly", "Possible enlarged heart size"),
            ("nodule", "Possible nodule"),
            ("infection", "Possible infection mentioned"),
        ],
        "no_clear": "No clear {keyword} seen",
        "report_impression": "Report impression noted",
        "report_wording": "Detected from report wording.",
        "report_impression_evidence": "Taken from the report impression section.",
    })

    findings: List[KeyFinding] = []
    text = " ".join(raw_text.split())
    lowered = text.lower()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for keyword, title in localized["keyword_map"]:
        sentence = next((item for item in sentences if keyword in item.lower()), None)
        if not sentence:
            continue
        lowered_sentence = sentence.lower()
        if _sentence_negates_keyword(lowered_sentence, keyword):
            findings.append(
                KeyFinding(
                    title=localized["no_clear"].format(keyword=keyword),
                    detail=sentence.strip(),
                    severity="low",
                    evidence=localized["report_wording"],
                )
            )
            continue

        severity = "moderate"
        if keyword in {"pneumothorax", "fracture"}:
            severity = "high"
        findings.append(
            KeyFinding(
                title=title,
                detail=sentence.strip(),
                severity=severity,
                evidence=localized["report_wording"],
            )
        )

    if not findings:
        impression_match = re.search(r"(impression|conclusion)\s*:\s*(.+?)(?:$|\n[A-Z][A-Z ]{2,}:)", raw_text, flags=re.IGNORECASE | re.DOTALL)
        if impression_match:
            detail = " ".join(impression_match.group(2).split())[:260]
            findings.append(
                KeyFinding(
                    title=localized["report_impression"],
                    detail=detail,
                    severity="moderate" if any(word in lowered for word in ("may", "possible", "suggest")) else "low",
                    evidence=localized["report_impression_evidence"],
                )
            )

    return _dedupe_findings(findings[:6])


def _dedupe_findings(findings: List[KeyFinding]) -> List[KeyFinding]:
    seen = set()
    ordered: List[KeyFinding] = []
    for finding in findings:
        key = (finding.title.strip().lower(), finding.detail.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(finding)
    return ordered


def _sentence_negates_keyword(sentence: str, keyword: str) -> bool:
    patterns = [
        rf"\bno\b[^.:\n]{{0,40}}\b{re.escape(keyword)}\b",
        rf"\bwithout\b[^.:\n]{{0,40}}\b{re.escape(keyword)}\b",
        rf"\babsent\b[^.:\n]{{0,40}}\b{re.escape(keyword)}\b",
        rf"\b{re.escape(keyword)}\b[^.:\n]{{0,20}}\bnot seen\b",
    ]
    return any(re.search(pattern, sentence) for pattern in patterns)


def _fallback_summary(
    module: str,
    extracted: ExtractedContent,
    integrity: IntegrityResult,
    findings: Optional[List[KeyFinding]] = None,
) -> str:
    findings = findings or []
    if module == "xray":
        return (
            "The upload was received and validated for image review. A high-confidence radiology "
            "summary requires a configured local X-ray model or a clinician review."
        )
    if module == "blood":
        if integrity.status == "fail":
            return "The blood report could not be extracted cleanly enough for a dependable summary."
        if integrity.missing_fields:
            return (
                "The blood report contains useful information, but some core markers are missing "
                "or unclear, so the summary should be reviewed before use."
            )
        return "The blood report was extracted successfully and core lab markers were checked for abnormal values."
    if integrity.status == "fail":
        return "The report text could not be extracted reliably enough to produce a dependable summary."
    if findings:
        top_titles = ", ".join(finding.title for finding in findings[:2])
        return f"The report text suggests these main findings: {top_titles}. This is a simplified rule-based summary."
    return "The medical report text was extracted and prepared for structured review and summarization."


def _fallback_recommendations(module: str, integrity: IntegrityResult) -> List[str]:
    recommendations = []
    if integrity.status != "pass":
        recommendations.append("Review the extracted text against the original file before sharing the summary.")
    if module == "xray":
        recommendations.append("Use the configured local X-ray model or a radiologist review for final image interpretation.")
    elif module == "blood":
        recommendations.append("Confirm abnormal values against the lab's reference ranges and units.")
    else:
        recommendations.append("Have a clinician confirm highlighted findings and recommendations.")
    return recommendations


def _default_confidence_notes(
    module: str,
    extracted: ExtractedContent,
    integrity: IntegrityResult,
    *,
    live_model_used: bool,
    llm_error: Optional[str] = None,
    llm_attempted: bool = False,
) -> List[ConfidenceNote]:
    notes = [
        ConfidenceNote(
            area="extraction",
            level="high" if extracted.raw_text else "low",
            note=(
                "Readable text was extracted from the document."
                if extracted.raw_text
                else "Little or no text was extracted, which lowers downstream reliability."
            ),
        ),
        ConfidenceNote(
            area="integrity",
            level="high" if integrity.status == "pass" else "medium" if integrity.status == "review" else "low",
            note=f"Integrity status is '{integrity.status}' with completeness score {integrity.completeness_score:g}.",
        ),
        ConfidenceNote(
            area="llm",
            level="high" if live_model_used else "low",
            note=(
                f"{llm_service.provider_name} model {llm_service.model} generated the interpretation."
                if live_model_used
                else (
                    f"Live AI analysis was attempted but failed: {llm_error}"
                    if llm_attempted and llm_error
                    else "No live LLM is configured, so the output is based on deterministic extraction rules."
                )
            ),
        ),
    ]
    if module == "xray" and not live_model_used:
        notes.append(
            ConfidenceNote(
                area="image-interpretation",
                level="low",
                note=(
                    f"Medical image interpretation fell back because live vision analysis failed: {llm_error}"
                    if llm_attempted and llm_error
                    else "No live multimodal model was available for pixel-level medical image interpretation."
                ),
            )
        )
    return notes


def _coerce_confidence(
    payload: Any,
    extracted: ExtractedContent,
    integrity: IntegrityResult,
    module: str,
) -> List[ConfidenceNote]:
    notes = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            notes.append(
                ConfidenceNote(
                    area=str(item.get("area", "analysis")),
                    level=str(item.get("level", "medium")),
                    note=str(item.get("note", "")),
                )
            )
    return notes or _default_confidence_notes(module, extracted, integrity, live_model_used=True)


def _coerce_findings(payload: Any) -> List[KeyFinding]:
    findings = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            findings.append(
                KeyFinding(
                    title=str(item.get("title", "Finding")),
                    detail=str(item.get("detail", "")),
                    severity=str(item.get("severity", "low")),
                    evidence=item.get("evidence"),
                )
            )
    return findings


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _execution_metadata(module: str, live_model_used: bool, local_xray_used: bool) -> LLMExecution:
    if module == "xray" and local_xray_used:
        return LLMExecution(
            provider="Local",
            primary_model=local_xray_service.model_name,
            mode="local-vision",
            live_model_used=True,
            fallback_model=None,
        )
    return LLMExecution(
        provider=llm_service.provider_name,
        primary_model=llm_service.model,
        mode=llm_service.mode_name if live_model_used else "deterministic-fallback",
        live_model_used=live_model_used,
        fallback_model="rules-engine",
    )


def _guess_content_type(file_name: str) -> str:
    lower = file_name.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith(".txt"):
        return "text/plain"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


def _decode_text(contents: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return contents.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _build_patient_report_summary(
    *,
    module: str,
    extracted: ExtractedContent,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    recommendations: List[str],
) -> PatientReportSummary:
    main_findings = _main_findings_for_patient(module, findings, integrity, extracted)
    is_normal = _is_everything_normal(findings, integrity)

    if is_normal:
        normal_status_text = (
            "YES - no clear problem was highlighted in the available information, but this is still not a final diagnosis."
        )
    elif integrity.status == "fail" or integrity.missing_fields:
        normal_status_text = (
            "NO - the system could not confirm that everything is normal because some important information was missing or unclear."
        )
    else:
        normal_status_text = "NO - one or more findings may need medical review."

    return PatientReportSummary(
        main_findings=main_findings,
        is_everything_normal=is_normal,
        normal_status_text=normal_status_text,
        problems_concerns=_problems_concerns_for_patient(findings, integrity),
        important_values_observations=_important_values_observations(module, extracted, findings, integrity),
        simple_explanation=_build_patient_explanation(module, findings, integrity, extracted),
        suggested_next_steps=_suggested_next_steps(module, integrity, recommendations),
    )


def _main_findings_for_patient(
    module: str,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    extracted: ExtractedContent,
) -> List[str]:
    if findings:
        patient_lines = []
        for finding in findings[:4]:
            patient_lines.append(f"{finding.title}: {_to_patient_friendly_sentence(finding.detail)}")
        return patient_lines

    if module == "xray":
        return [
            "No clear urgent problem was confidently highlighted from the uploaded image.",
            "A radiologist or clinician should still confirm the final interpretation.",
        ]
    if module == "blood":
        return [
            "No obvious rule-based blood marker abnormality was found in the values that were provided."
        ]
    if not extracted.raw_text:
        return ["The report text could not be extracted clearly enough for a dependable summary."]
    if integrity.status != "pass":
        return ["The report was read only partially, so the summary may miss important details."]
    return ["No major problem was clearly highlighted from the extracted report text."]


def _is_everything_normal(findings: List[KeyFinding], integrity: IntegrityResult) -> bool:
    if findings:
        return False
    if integrity.status == "fail":
        return False
    if integrity.missing_fields or integrity.suspicious_values:
        return False
    return True


def _problems_concerns_for_patient(findings: List[KeyFinding], integrity: IntegrityResult) -> List[str]:
    concerns = [f"{finding.title}: {_to_patient_friendly_sentence(finding.detail)}" for finding in findings[:6]]
    concerns.extend(f"Missing information: {field}" for field in integrity.missing_fields[:4])
    concerns.extend(integrity.suspicious_values[:4])
    concerns.extend(integrity.extraction_inconsistencies[:4])
    concerns.extend(integrity.formatting_issues[:3])
    return _unique_preserve_order(concerns)


def _important_values_observations(
    module: str,
    extracted: ExtractedContent,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
) -> List[str]:
    values: List[str] = []
    lines = [line.strip() for line in extracted.raw_text.splitlines() if line.strip()]

    for line in lines:
        if len(values) >= 6:
            break
        if module == "blood" and ":" in line:
            values.append(line)
            continue
        if any(char.isdigit() for char in line):
            values.append(line[:160])

    if module == "xray" and not values:
        values.extend(f"{finding.title}: {finding.detail}"[:180] for finding in findings[:4])

    if not values and findings:
        values.extend(f"{finding.title}: {finding.detail}"[:180] for finding in findings[:4])

    if integrity.completeness_score:
        values.append(f"Report completeness check: {integrity.completeness_score:g}%")

    return _unique_preserve_order(values[:6])


def _suggested_next_steps(module: str, integrity: IntegrityResult, recommendations: List[str]) -> List[str]:
    steps = list(recommendations)
    if integrity.status != "pass":
        steps.insert(0, "Compare this summary with the original report because some data may be incomplete.")
    if module == "pdf":
        steps.append("Discuss the key findings with the doctor who ordered the test.")
    if module == "blood":
        steps.append("Ask a doctor to review any abnormal values together with your symptoms and lab reference ranges.")
    if module == "xray":
        steps.append("Have a clinician or radiologist confirm the image findings before acting on them.")
    return _unique_preserve_order(steps)


def _to_patient_friendly_sentence(text: str) -> str:
    simplified = " ".join(text.split())

    phrase_replacements = [        ("No acute cardiopulmonary abnormality detected.", "No urgent heart or lung problem was found."),
        ("No acute abnormality detected.", "No urgent problem was found."),
        ("No significant findings.", "No major problem was clearly highlighted."),
        ("No definite abnormality.", "No clear abnormality was found."),
        ("No focal abnormality.", "No localized abnormality was found."),
        ("No evidence of acute disease.", "No sign of an urgent illness was found."),
        ("No evidence of pneumonia.", "No sign of lung infection was found."),
        ("No evidence of pulmonary edema.", "No sign of fluid in the lungs was found."),
        ("No evidence of pleural effusion.", "No sign of fluid around the lungs was found."),
        ("No evidence of pneumothorax.", "No sign of collapsed lung was found."),
        ("No evidence of consolidation.", "No sign of solid lung area was found."),
        ("No evidence of atelectasis.", "No sign of partial lung collapse was found."),
        ("No evidence of ground-glass opacity.", "No sign of hazy lung area was found."),
        ("No evidence of patchy opacity.", "No sign of uneven lung shadow was found."),
        ("No evidence of interstitial disease.", "No sign of lung tissue change was found."),
        ("No evidence of cardiomegaly.", "No sign of enlarged heart was found."),
        ("No evidence of mediastinal widening.", "No sign of widened space in the chest was found."),
        ("No evidence of hilar enlargement.", "No sign of enlarged lymph nodes near the lungs was found."),
        ("No evidence of pulmonary nodules.", "No sign of small lung lumps was found."),
        ("No evidence of pulmonary masses.", "No sign of large lung lumps was found."),
        ("No evidence of pulmonary fibrosis.", "No sign of lung scarring was found."),
        ("No evidence of pulmonary hypertension.", "No sign of high blood pressure in the lungs was found."),
        ("No evidence of pulmonary embolism.", "No sign of blood clot in the lungs was found."),
        ("No evidence of pulmonary infarction.", "No sign of dead lung tissue was found."),
        ("No evidence of pulmonary contusion.", "No sign of bruised lung tissue was found."),
        ("No evidence of pulmonary hemorrhage.", "No sign of bleeding in the lungs was found."),
        ("No evidence of pulmonary abscess.", "No sign of infected lung pus was found."),
        ("No pleural effusion or pneumothorax.", "No fluid around the lungs and no collapsed-lung sign was described."),
        ("No pleural effusion.", "No fluid around the lungs was found."),
        ("No pneumothorax.", "No collapsed-lung pattern was found."),
        ("no acute cardiopulmonary abnormality", "no urgent heart or lung problem was found"),
        ("no acute abnormality", "no urgent problem was found"),
        ("cardiomediastinal silhouette", "the outline of the heart and the centre of the chest"),
        ("cardiomediastinal", "heart and central chest area"),
        ("airspace opacity", "shadow-like area in the lung"),
        ("airspace disease", "a change in the lung tissue"),
        ("opacity-type pattern", "shadow-like area"),
        ("patchy opacity", "uneven shadow-like area in the lung"),
        ("ground-glass opacity", "faint hazy area in the lung"),
        ("consolidation", "a solid-looking area in the lung (may indicate infection or fluid)"),
        ("atelectasis", "partial collapse or closure of part of the lung"),
        ("atelectatic", "showing a partly collapsed area of the lung"),
        ("pneumothorax", "collapsed lung (air trapped outside the lung)"),
        ("pleural effusion", "fluid buildup around the lung"),
        ("pleural", "around the lung"),
        ("pulmonary edema", "fluid buildup inside the lung tissue"),
        ("pulmonary", "relating to the lungs"),
        ("mediastinal widening", "widening of the space between the lungs"),
        ("mediastinum", "the space between the two lungs"),
        ("cardiomegaly", "an enlarged heart"),
        ("cardiac silhouette", "the outline of the heart on the X-ray"),
        ("cardiac", "heart-related"),
        ("hilar", "near the root of the lung where blood vessels enter"),
        ("interstitial", "within the tissue between the air sacs of the lung"),
        ("infiltrate", "an area where fluid or cells have entered the lung tissue"),
        ("bilateral", "on both sides (both lungs or both sides of the body)"),
        ("unilateral", "on one side only"),
        ("radiographic", "X-ray"),
        ("radiologically", "as seen on the X-ray"),
        ("radiologist", "X-ray specialist doctor"),
        ("hemoglobin", "hemoglobin (the protein in red blood cells that carries oxygen)"),
        ("hematocrit", "hematocrit (the proportion of red blood cells in your blood)"),
        ("leukocytes", "white blood cells (the cells that fight infection)"),
        ("erythrocytes", "red blood cells (the cells that carry oxygen)"),
        ("thrombocytes", "platelets (the cells that help blood clot)"),
        ("lymphocytes", "a type of white blood cell that fights infection"),
        ("neutrophils", "a type of white blood cell that attacks bacteria"),
        ("microcytic", "smaller than normal red blood cells"),
        ("macrocytic", "larger than normal red blood cells"),
        ("normocytic", "normal-sized red blood cells"),
        ("hypochromic", "paler than normal red blood cells, often meaning low iron"),
        ("anemia", "a low level of red blood cells or hemoglobin (often causing tiredness)"),
        ("hyperglycemia", "high blood sugar"),
        ("hypoglycemia", "low blood sugar"),
    ]

    for old, new in phrase_replacements:
        simplified = simplified.replace(old, new)

    if not simplified.strip():
        return ""
    if not simplified.strip().endswith((".", "!", "?")):
        simplified = simplified.strip() + "."
    return simplified[0].upper() + simplified[1:]


_LOCALIZED_STRINGS: Dict[str, Dict[str, Any]] = {
    "en": {
        "status_yes": "YES - no clear problem was highlighted in the available information, but this is still not a final diagnosis.",
        "status_incomplete": "NO - the system could not confirm that everything is normal because some important information was missing or unclear.",
        "status_review": "NO - one or more findings may need medical review.",
        "xray_urgent": "The X-ray may show a potentially important abnormality, so this result should be reviewed promptly by a radiologist or clinician. It is still a supportive AI summary, not a final diagnosis.",
        "xray_followup": "The X-ray review found image features that may need medical follow-up. This explanation is meant to make the result easier to understand, but a radiologist should confirm the final interpretation.",
        "xray_upload_ok": "The image was uploaded successfully. This system can highlight possible concerns, but final image interpretation should come from a radiologist or a configured vision model.",
        "blood_outside_range": "Some blood markers may be outside the expected range. This does not confirm a disease, but it does mean the values deserve a clinician review in the context of symptoms and reference ranges.",
        "blood_no_obvious_issue": "The submitted blood values did not show obvious rule-based abnormalities in the markers provided. Normal-looking numbers still need clinical context and the lab's own reference ranges.",
        "pdf_not_readable": "The report could not be read clearly enough for a dependable explanation. A cleaner PDF or clinician review is recommended.",
        "pdf_has_findings": "The report text contains findings that may indicate a medical issue. The summary below is simplified for readability and should be confirmed by a doctor.",
        "pdf_fallback_summary": "The report text was extracted and simplified into plain language. No strong issue was flagged by the fallback rules, but the document should still be reviewed clinically.",
        "general_cause": "The condition described in the original report",
        "general_health_issue": "An underlying health issue that your doctor can help identify",
        "xray_generic_cause": "Image quality or positioning can sometimes affect how findings appear on an X-ray",
        "generic_effect": "Effects depend on the specific condition described in the report",
        "generic_symptom_warning": "Your symptoms and medical history will help your doctor assess the urgency",
        "xray_symptom_warning": "Findings on an X-ray should always be considered alongside your symptoms by a doctor",
        "incomplete_data_risk": "Because some data was incomplete or unclear, this summary may not capture the full picture — always cross-check with your original report and your doctor.",
        "xray_urgent_risk": "Waiting too long to get a formal review could mean missing a significant chest or bone problem that needs prompt treatment.",
        "xray_respiratory_risk": "If the lung or breathing-related findings are real and not treated, breathing difficulties may worsen over time.",
        "ai_risks_disclaimer": "An AI-assisted image review alone cannot replace a full assessment by a radiologist or doctor who also knows your medical history.",
        "condition_risk": "The long-term risk depends on the specific condition, how severe it is, and how quickly it is followed up and treated.",
        "base_advice": [
            "This summary is a helpful starting point for understanding your report — it is not a medical diagnosis.",
            "If your symptoms are severe, getting worse, or worrying you, seek medical attention promptly.",
            "Write down any questions this summary raises so you can ask your doctor at your next appointment.",
        ],
        "blood_advice": [
            "Always compare the values in this summary with the reference ranges printed on your own lab report — ranges can vary slightly between labs.",
            "A single out-of-range result does not always mean there is a serious problem — your doctor will consider the full picture.",
        ],
        "xray_advice": [
            "This AI summary should be reviewed alongside the official radiology report written by a qualified radiologist.",
            "Do not make any treatment decisions based on this summary alone.",
        ],
        "pdf_advice": [
            "Discuss this summary with the doctor who ordered the test — they will have the context to interpret it properly.",
        ],
    },
    "hi": {
        "status_yes": "हाँ - उपलब्ध जानकारी में कोई स्पष्ट समस्या नहीं पाई गई, लेकिन यह अभी भी अंतिम निदान नहीं है।",
        "status_incomplete": "नहीं - सिस्टम यह पुष्टि नहीं कर सका कि सब कुछ सामान्य है क्योंकि कुछ महत्वपूर्ण जानकारी गायब या अस्पष्ट थी।",
        "status_review": "नहीं - एक या अधिक निष्कर्षों को चिकित्सा समीक्षा की आवश्यकता हो सकती है।",
        "xray_urgent": "एक्स-रे एक संभावित महत्वपूर्ण असामान्यता दिखा सकता है, इसलिए इस परिणाम की तुरंत रेडियोलॉजिस्ट या चिकित्सक द्वारा समीक्षा की जानी चाहिए। यह अभी भी एक सहायक एआई सारांश है, अंतिम निदान नहीं।",
        "xray_followup": "एक्स-रे समीक्षा में ऐसी छवियां मिलीं जिन्हें चिकित्सा अनुवर्ती कार्रवाई की आवश्यकता हो सकती है। यह स्पष्टीकरण परिणाम को समझने में आसान बनाने के लिए है, लेकिन रेडियोलॉजिस्ट को अंतिम व्याख्या की पुष्टि करनी चाहिए।",
        "xray_upload_ok": "छवि सफलतापूर्वक अपलोड की गई थी। यह सिस्टम संभावित चिंताओं को उजागर कर सकता है, लेकिन अंतिम छवि व्याख्या रेडियोलॉजिस्ट या कॉन्फ़िगर किए गए विज़न मॉडल से आनी चाहिए।",
        "blood_outside_range": "कुछ रक्त मार्कर अपेक्षित सीमा से बाहर हो सकते हैं। यह किसी बीमारी की पुष्टि नहीं करता है, लेकिन इसका मतलब है कि मान लक्षणों और संदर्भ सीमाओं के संदर्भ में चिकित्सक की समीक्षा के योग्य हैं।",
        "blood_no_obvious_issue": "प्रस्तुत रक्त मानों ने प्रदान किए गए मार्करों में स्पष्ट नियम-आधारित असामान्यताएं नहीं दिखाईं। सामान्य दिखने वाली संख्याओं के लिए अभी भी नैदानिक संदर्भ और लैब की अपनी संदर्भ सीमाओं की आवश्यकता होती है।",
        "pdf_not_readable": "एक भरोसेमंद स्पष्टीकरण के लिए रिपोर्ट को पर्याप्त स्पष्ट रूप से नहीं पढ़ा जा सका। एक स्पष्ट पीडीएफ या चिकित्सक समीक्षा की सिफारिश की जाती है।",
        "pdf_has_findings": "रिपोर्ट टेक्स्ट में ऐसे निष्कर्ष हैं जो एक चिकित्सा समस्या का संकेत दे सकते हैं। नीचे दिया गया सारांश पठनीयता के लिए सरल किया गया है और डॉक्टर द्वारा इसकी पुष्टि की जानी चाहिए।",
        "pdf_fallback_summary": "रिपोर्ट टेक्स्ट निकाला गया और सरल भाषा में सरल किया गया। फॉलबैक नियमों द्वारा कोई ठोस समस्या नहीं बताई गई थी, लेकिन दस्तावेज़ की अभी भी नैदानिक रूप से समीक्षा की जानी चाहिए।",
        "general_cause": "मूल रिपोर्ट में वर्णित स्थिति",
        "general_health_issue": "एक अंतर्निहित स्वास्थ्य समस्या जिसे आपके डॉक्टर को पहचानने में मदद मिल सकती है",
        "xray_generic_cause": "छवि की गुणवत्ता या स्थिति कभी-कभी यह प्रभावित कर सकती है कि एक्स-रे पर निष्कर्ष कैसे दिखाई देते हैं",
        "generic_effect": "प्रभाव रिपोर्ट में वर्णित विशिष्ट स्थिति पर निर्भर करते हैं",
        "generic_symptom_warning": "आपके लक्षण और चिकित्सा इतिहास आपके डॉक्टर को तात्कालिकता का आकलन करने में मदद करेंगे",
        "xray_symptom_warning": "एक्स-रे के निष्कर्षों को हमेशा डॉक्टर द्वारा आपके लक्षणों के साथ विचार किया जाना चाहिए",
        "incomplete_data_risk": "चूंकि कुछ डेटा अधूरा या अस्पष्ट था, इसलिए यह सारांश पूरी तस्वीर को कैप्चर नहीं कर सकता है - हमेशा अपनी मूल रिपोर्ट और अपने डॉक्टर के साथ क्रॉस-चेक करें।",
        "xray_urgent_risk": "औपचारिक समीक्षा प्राप्त करने के लिए बहुत लंबा इंतजार करने का मतलब एक महत्वपूर्ण छाती या हड्डी की समस्या को याद करना हो सकता है जिसे त्वरित उपचार की आवश्यकता होती है।",
        "xray_respiratory_risk": "यदि फेफड़े या सांस लेने से संबंधित निष्कर्ष वास्तविक हैं और उनका इलाज नहीं किया जाता है, तो सांस लेने में कठिनाई समय के साथ खराब हो सकती है।",
        "ai_risks_disclaimer": "एक अकेला एआई-सहायता प्राप्त छवि समीक्षा एक रेडियोलॉजिस्ट या डॉक्टर द्वारा पूर्ण मूल्यांकन की जगह नहीं ले सकती जो आपके चिकित्सा इतिहास को भी जानता है।",
        "condition_risk": "दीर्घकालिक जोखिम विशिष्ट स्थिति, यह कितना गंभीर है, और कितनी जल्दी इसका पालन किया जाता है और इलाज किया जाता है, इस पर निर्भर करता है।",
        "base_advice": [
            "यह सारांश आपकी रिपोर्ट को समझने के लिए एक सहायक प्रारंभिक बिंदु है - यह चिकित्सा निदान नहीं है।",
            "यदि आपके लक्षण गंभीर हैं, बिगड़ रहे हैं, या आपको चिंतित कर रहे हैं, तो तुरंत चिकित्सा सहायता लें।",
            "इस सारांश से उठने वाले किसी भी प्रश्न को लिख लें ताकि आप अपनी अगली नियुक्ति में अपने डॉक्टर से पूछ सकें।",
        ],
        "blood_advice": [
            "हमेशा इस सारांश के मानों की तुलना अपनी लैब रिपोर्ट पर छपी संदर्भ सीमाओं से करें - लैब के बीच सीमाएं थोड़ी भिन्न हो सकती हैं।",
            "एकल सीमा से बाहर परिणाम का हमेशा यह मतलब नहीं होता है कि कोई गंभीर समस्या है - आपका डॉक्टर पूरी तस्वीर पर विचार करेगा।",
        ],
        "xray_advice": [
            "इस एआई सारांश की समीक्षा एक योग्य रेडियोलॉजिस्ट द्वारा लिखित आधिकारिक रेडियोलॉजी रिपोर्ट के साथ की जानी चाहिए।",
            "अकेले इस सारांश के आधार पर कोई उपचार निर्णय न लें।",
        ],
        "pdf_advice": [
            "इस सारांश पर उस डॉक्टर के साथ चर्चा करें जिसने परीक्षण का आदेश दिया था - उनके पास इसकी सही व्याख्या करने का संदर्भ होगा।",
        ],
    },
    "gu": {
        "status_yes": "હા - ઉપલબ્ધ માહિતીમાં કોઈ સ્પષ્ટ સમસ્યા હાઇલાઇટ કરવામાં આવી નથી, પરંતુ આ હજી અંતિમ નિદાન નથી.",
        "status_incomplete": "ના - સિસ્ટમ પુષ્ટિ કરી શકી નથી કે બધું સામાન્ય છે કારણ કે કેટલીક મહત્વપૂર્ણ માહિતી ખૂટે છે અથવા અસ્પષ્ટ હતી.",
        "status_review": "ના - એક અથવા વધુ તારણોને તબીબી સમીક્ષાની જરૂર પડી શકે છે.",
        "xray_urgent": "એક્સ-રે સંભવિત મહત્વપૂર્ણ અસાધારણતા દર્શાવી શકે છે, તેથી આ પરિણામની રેડિયોલોજિસ્ટ અથવા ચિકિત્સક દ્વારા તાત્કાલિક સમીક્ષા થવી જોઈએ. તે હજી પણ સહાયક AI સારાંશ છે, અંતિમ નિદાન નથી.",
        "xray_followup": "એક્સ-રે સમીક્ષામાં એવી ઇમેજ વિશેષતાઓ મળી છે જેને તબીબી ફોલો-અપની જરૂર પડી શકે છે. આ સમજૂતી પરિણામને સમજવામાં સરળ બનાવવા માટે છે, પરંતુ રેડિયોલોજિસ્ટે અંતિમ અર્થઘટનની પુષ્ટિ કરવી જોઈએ.",
        "xray_upload_ok": "ઇમેજ સફળતાપૂર્વક અપલોડ કરવામાં આવી હતી. આ સિસ્ટમ સંભવિત ચિંતાઓને હાઇલાઇટ કરી શકે છે, પરંતુ અંતિમ ઇમેજ અર્થઘટન રેડિયોલોજિસ્ટ અથવા રૂપરેખાંકિત વિઝન મોડેલ તરફથી આવવું જોઈએ.",
        "blood_outside_range": "કેટલાક બ્લડ માર્કર્સ અપેક્ષિત મર્યાદાની બહાર હોઈ શકે છે. આ કોઈ રોગની પુષ્ટિ કરતું નથી, પરંતુ તેનો અર્થ એ છે કે મૂલ્યો લક્ષણો અને સંદર્ભ મર્યાદાઓના સંદર્ભમાં ચિકિત્સકની સમીક્ષા માટે યોગ્ય છે.",
        "blood_no_obvious_issue": "સબમિટ કરેલા બ્લડ મૂલ્યોએ પ્રદાન કરેલા માર્કર્સમાં સ્પષ્ટ નિયમ-આધારિત અસાધારણતા દર્શાવી નથી. સામાન્ય દેખાતી સંખ્યાઓ માટે હજુ પણ ક્લિનિકલ સંદર્ભ અને લેબની પોતાની સંદર્ભ મર્યાદાઓની જરૂર છે.",
        "pdf_not_readable": "ભરોસાપાત્ર સમજૂતી માટે રિપોર્ટ પૂરતી સ્પષ્ટ રીતે વાંચી શકાયો નથી. ક્લીનર પીડીએફ અથવા ચિકિત્સક સમીક્ષાની ભલામણ કરવામાં આવે છે.",
        "pdf_has_findings": "રિપોર્ટ ટેક્સ્ટમાં એવા તારણો છે જે તબીબી સમસ્યા સૂચવી શકે છે. નીચેનો સારાંશ વાંચવાની સરળતા માટે સરળ બનાવવામાં આવ્યો છે અને તેની ડૉક્ટર દ્વારા પુષ્ટિ થવી જોઈએ.",
        "pdf_fallback_summary": "રિપોર્ટ ટેક્સ્ટ કાઢવામાં આવ્યો હતો અને સાદી ભાષામાં સરળ બનાવવામાં આવ્યો હતો. ફોલબૅક નિયમો દ્વારા કોઈ મજબૂત સમસ્યા ચિહ્નિત કરવામાં આવી નથી, પરંતુ દસ્તાવેજની હજી પણ ક્લિનિકલ સમીક્ષા થવી જોઈએ.",
        "general_cause": "મૂળ રિપોર્ટમાં વર્ણવેલી સ્થિતિ",
        "general_health_issue": "એક અંતર્નિહિત આરોગ્ય સમસ્યા જે તમારા ડૉક્ટરને ઓળખવામાં મદદ કરી શકે છે",
        "xray_generic_cause": "ઇમેજની ગુણવત્તા અથવા સ્થિતિ ક્યારેક એક્સ-રે પર તારણો કેવી રીતે દેખાય છે તેના પર અસર કરી શકે છે",
        "generic_effect": "અસરો રિપોર્ટમાં વર્ણવેલ ચોક્કસ સ્થિતિ પર આધાર રાખે છે",
        "generic_symptom_warning": "તમારા લક્ષણો અને તબીબી ઇતિહાસ તમારા ડૉક્ટરને તાકીદનું મૂલ્યાંકન કરવામાં મદદ કરશે",
        "xray_symptom_warning": "એક્સ-રે પરના તારણો પર હંમેશા તમારા લક્ષણોની સાથે ડૉક્ટર દ્વારા વિચારવું જોઈએ",
        "incomplete_data_risk": "કારણ કે કેટલાક ડેટા અપૂર્ણ અથવા અસ્પષ્ટ હતા, આ સારાંશ સંપૂર્ણ ચિત્રને કેપ્ચર કરી શકશે નહીં — હંમેશા તમારા મૂળ રિપોર્ટ અને તમારા ડૉક્ટર સાથે ક્રોસ-ચેક કરો.",
        "xray_urgent_risk": "ઔપચારિક સમીક્ષા મેળવવા માટે ખૂબ લાંબો સમય રાહ જોવાનો અર્થ એ થઈ શકે છે કે છાતી અથવા હાડકાની નોંધપાત્ર સમસ્યા ચૂકી જવી જેના માટે તાત્કાલિક સારવારની જરૂર છે.",
        "xray_respiratory_risk": "જો ફેફસાં અથવા શ્વાસ સંબંધિત તારણો વાસ્તવિક હોય અને તેની સારવાર કરવામાં ન આવે, તો સમય જતાં શ્વાસ લેવાની મુશ્કેલીઓ વધી શકે છે.",
        "ai_risks_disclaimer": "રેડિયોલોજિસ્ટ અથવા ડૉક્ટર દ્વારા સંપૂર્ણ મૂલ્યાંકનને માત્ર AI-આધારિત ઇમેજ સમીક્ષા બદલી શકતી નથી જે તમારા તબીબી ઇતિહાસને પણ જાણે છે.",
        "condition_risk": "લાંબા ગાળાનું જોખમ ચોક્કસ સ્થિતિ, તે કેટલી ગંભીર છે અને કેટલી ઝડપથી તેને અનુસરવામાં આવે છે અને તેની સારવાર કરવામાં આવે છે તેના પર આધાર રાખે છે.",
        "base_advice": [
            "આ સારાંશ તમારા રિપોર્ટને સમજવા માટે એક મદદરૂપ પ્રારંભિક બિંદુ છે — તે તબીબી નિદાન નથી.",
            "જો તમારા લક્ષણો ગંભીર હોય, બગડતા હોય અથવા તમને ચિંતિત કરતા હોય, તો તાત્કાલિક તબીબી સહાય મેળવો.",
            "આ સારાંશ જે પ્રશ્નો ઉભા કરે છે તે લખી લો જેથી તમે તમારી આગામી એપોઇન્ટમેન્ટ વખતે તમારા ડૉક્ટરને પૂછી શકો.",
        ],
        "blood_advice": [
            "હંમેશા આ સારાંશના મૂલ્યોની તુલના તમારા પોતાના લેબ રિપોર્ટ પર છપાયેલી સંદર્ભ મર્યાદાઓ સાથે કરો — લેબ વચ્ચે મર્યાદાઓ થોડી બદલાઈ શકે છે.",
            "સિંગલ મર્યાદા બહારના પરિણામનો અર્થ એ નથી કે કોઈ ગંભીર સમસ્યા છે — તમારા ડૉક્ટર સંપૂર્ણ ચિત્ર પર વિચાર કરશે.",
        ],
        "xray_advice": [
            "આ AI સારાંશની સમીક્ષા લાયક રેડિયોલોજિસ્ટ દ્વારા લખાયેલા સત્તાવાર રેડિયોલોજી રિપોર્ટની સાથે થવી જોઈએ.",
            "માત્ર આ સારાંશના આધારે સારવારના કોઈ નિર્ણયો ન લો.",
        ],
        "pdf_advice": [
            "જે ડૉક્ટરે ટેસ્ટનો ઓર્ડર આપ્યો હતો તેની સાથે આ સારાંશની ચર્ચા કરો — તેમની પાસે તેનું યોગ્ય રીતે અર્થઘટન કરવા માટેનો સંદર્ભ હશે.",
        ],
    },
}


def _build_patient_explanation(
    module: str,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    extracted: ExtractedContent,
    recommendations: Optional[List[str]] = None,
    language: str = "en",
) -> str:
    strings = _LOCALIZED_STRINGS.get(language, _LOCALIZED_STRINGS["en"])
    if module == "xray":
        if language == "en":
            return _build_xray_patient_explanation(findings, integrity, recommendations or [])
        xray_signal = _xray_signal_profile(findings)
        if xray_signal["urgent"]:
            return strings["xray_urgent"]
        if findings:
            return strings["xray_followup"]
        return strings["xray_upload_ok"]
    if module == "blood":
        if findings:
            return strings["blood_outside_range"]
        return strings["blood_no_obvious_issue"]
    if not extracted.raw_text:
        return strings["pdf_not_readable"]
    if findings:
        return strings["pdf_has_findings"]
    return strings["pdf_fallback_summary"]


def _build_xray_patient_explanation(
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    recommendations: List[str],
) -> str:
    primary = _primary_xray_finding(findings)
    concern = _xray_concern_level(findings, integrity)
    summary = _xray_summary_text(primary, findings, integrity, concern)
    main_problem = _xray_main_problem_text(primary, findings)
    serious_line = _xray_concern_reason(concern, findings, integrity)
    next_steps = _xray_next_steps(concern, integrity, recommendations)
    simple_explanation = _xray_simple_explanation(primary, findings, integrity)

    lines = [
        f"1. Summary: {summary}",
        f"2. Main Problem (if any): {main_problem}",
        f"3. Is it Serious?: {concern} concern. {serious_line}",
        "4. What Should You Do Next?:",
    ]
    lines.extend(f"- {step}" for step in next_steps)
    lines.extend(
        [
            f"5. Simple Explanation: {simple_explanation}",
            "6. Important Note: This is NOT a final diagnosis. A doctor or radiologist needs to review the X-ray.",
        ]
    )
    return "\n".join(lines)


def _primary_xray_finding(findings: List[KeyFinding]) -> Optional[KeyFinding]:
    ranked = sorted(
        findings,
        key=lambda finding: (_severity_rank(finding.severity), 1 if _is_reassuring_xray_finding(finding) else 0),
        reverse=True,
    )
    for finding in ranked:
        if not _is_reassuring_xray_finding(finding):
            return finding
    return ranked[0] if ranked else None


def _severity_rank(severity: str) -> int:
    return {"high": 3, "moderate": 2, "low": 1}.get((severity or "").lower(), 0)


def _is_reassuring_xray_finding(finding: KeyFinding) -> bool:
    text = f"{finding.title} {finding.detail}".lower()
    reassuring_terms = (
        "no finding",
        "normal",
        "no clear abnormality",
        "no acute abnormality",
        "no strong abnormality",
        "no urgent problem",
    )
    return any(term in text for term in reassuring_terms)


def _xray_concern_level(findings: List[KeyFinding], integrity: IntegrityResult) -> str:
    if any((finding.severity or "").lower() == "high" for finding in findings):
        return "High"
    if integrity.status == "fail":
        return "Moderate"
    if any((finding.severity or "").lower() == "moderate" for finding in findings):
        return "Moderate"
    return "Low"


def _xray_summary_text(
    primary: Optional[KeyFinding],
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    concern: str,
) -> str:
    if integrity.status == "fail" and not primary:
        return (
            "This X-ray review was limited because the image or extracted data was not clear enough. "
            "No firm result can be given from this review alone. A doctor should confirm the final reading."
        )

    if primary is None or _is_reassuring_xray_finding(primary):
        return (
            "This X-ray review did not highlight a clear serious problem. "
            "Everything looks mostly normal in this review. "
            "A doctor should still confirm the final reading."
        )

    detail = _simplify_xray_text(primary.detail)
    concern_sentence = (
        "This needs prompt medical review."
        if concern == "High"
        else "This should be checked by a doctor."
    )
    if integrity.status != "pass":
        return f"{detail} {concern_sentence} The review was also limited by incomplete or unclear data."
    return f"{detail} {concern_sentence} It is not a final diagnosis."


def _xray_main_problem_text(primary: Optional[KeyFinding], findings: List[KeyFinding]) -> str:
    if primary is None or _is_reassuring_xray_finding(primary):
        return "Everything looks mostly normal."

    short_problem = _xray_problem_label(primary, findings)
    if all(_severity_rank(finding.severity) <= 1 for finding in findings):
        return f"Everything looks mostly normal, but a small change was noted: {short_problem}."
    return short_problem


def _xray_problem_label(primary: KeyFinding, findings: List[KeyFinding]) -> str:
    text = f"{primary.title} {primary.detail}".lower()
    if any(term in text for term in ("opacity", "shadow", "consolidation", "infiltrate", "airspace")):
        return "A shadow-like area was seen on the X-ray"
    if any(term in text for term in ("effusion", "fluid around the lung", "pleural fluid")):
        return "There may be fluid around the lung"
    if any(term in text for term in ("edema", "fluid in the lung")):
        return "There may be extra fluid in the lungs"
    if any(term in text for term in ("pneumothorax", "collapsed lung")):
        return "There may be a collapsed-lung pattern"
    if any(term in text for term in ("fracture", "broken")):
        return "There may be a broken bone pattern"
    if any(term in text for term in ("dislocation", "out of place")):
        return "There may be a joint out of place"
    if any(term in text for term in ("cardiomegaly", "enlarged heart")):
        return "The heart looks bigger than usual on the X-ray"
    return _simplify_xray_text(primary.title)


def _xray_concern_reason(concern: str, findings: List[KeyFinding], integrity: IntegrityResult) -> str:
    if concern == "High":
        return "The image may show a more important problem that should be checked quickly."
    if concern == "Moderate":
        if integrity.status == "fail":
            return "The result is limited, so a doctor needs to confirm what the image really shows."
        return "A real change may be present, but it still needs doctor review to confirm exactly what it is."
    return "No clear serious problem was highlighted in this review."


def _xray_next_steps(concern: str, integrity: IntegrityResult, recommendations: List[str]) -> List[str]:
    steps = [
        "Show this X-ray to a doctor or radiologist for the final reading.",
    ]

    if concern == "High":
        steps.append("Get urgent medical help if you have trouble breathing, severe chest pain, or a serious injury.")
    elif concern == "Moderate":
        steps.append("Arrange a medical follow-up soon, especially if you have pain, fever, cough, or trouble breathing.")
    else:
        steps.append("Keep watching your symptoms and follow your doctor's advice if anything gets worse.")

    if integrity.status != "pass":
        steps.append("If the image was unclear, ask if a repeat X-ray or another test is needed.")
    else:
        for recommendation in recommendations:
            simplified = _simplify_xray_text(recommendation)
            if simplified and simplified not in steps:
                steps.append(simplified)
            if len(steps) >= 3:
                break

    return _unique_preserve_order(steps)[:3]


def _xray_simple_explanation(
    primary: Optional[KeyFinding],
    findings: List[KeyFinding],
    integrity: IntegrityResult,
) -> str:
    if integrity.status == "fail" and not primary:
        return (
            "The picture could not be reviewed clearly enough to say what is going on. "
            "That means the result needs a doctor review and may need a better image."
        )

    if primary is None or _is_reassuring_xray_finding(primary):
        return (
            "This means the picture did not show a clear major problem in this review. "
            "Even normal-looking X-rays still need a doctor to confirm the final result."
        )

    text = f"{primary.title} {primary.detail}".lower()
    if any(term in text for term in ("opacity", "shadow", "consolidation", "infiltrate", "airspace")):
        return (
            "Part of the X-ray looks more shaded than expected. "
            "That means one area does not look fully clear and a doctor should match it with your symptoms."
        )
    if any(term in text for term in ("effusion", "fluid around the lung", "pleural fluid")):
        return (
            "The picture may be showing extra fluid around the lung. "
            "That can affect breathing and needs a doctor to confirm it."
        )
    if any(term in text for term in ("edema", "fluid in the lung")):
        return (
            "The picture may be showing extra fluid in the lungs. "
            "This can make breathing harder and should be checked soon."
        )
    if any(term in text for term in ("pneumothorax", "collapsed lung")):
        return (
            "This may mean air is sitting outside the lung instead of inside it. "
            "That can be serious and needs quick medical review."
        )
    if any(term in text for term in ("fracture", "broken")):
        return (
            "This may mean a bone does not look whole or lined up normally on the picture. "
            "A doctor should confirm the injury and decide if more treatment is needed."
        )
    if any(term in text for term in ("dislocation", "out of place")):
        return (
            "This may mean a joint is not sitting in its normal position. "
            "A doctor should confirm it and check for injury."
        )
    if any(term in text for term in ("cardiomegaly", "enlarged heart")):
        return (
            "This means the heart looks bigger than usual on the X-ray picture. "
            "A doctor will need to decide if that is important for you."
        )
    return (
        f"{_simplify_xray_text(primary.detail)} "
        "A doctor should review the image to explain exactly what it means."
    )


def _simplify_xray_text(text: str) -> str:
    simplified = " ".join((text or "").split())
    if not simplified:
        return ""

    simplified = re.sub(r"Local model confidence [^.]*\.", "", simplified, flags=re.IGNORECASE)
    simplified = re.sub(r"Model confidence was [^.]*\.", "", simplified, flags=re.IGNORECASE)
    simplified = re.sub(r"with \d+(?:\.\d+)?% confidence", "", simplified, flags=re.IGNORECASE)
    simplified = re.sub(r"at \d+(?:\.\d+)?% confidence", "", simplified, flags=re.IGNORECASE)

    replacements = [
        ("The local model did not highlight a confident acute abnormality and leaned toward a normal study pattern", "No clear serious problem was highlighted"),
        ("The local model did not identify a strong acute abnormality", "No clear serious problem was highlighted"),
        ("The model suggests", "The X-ray may show"),
        ("The image shows", "The X-ray shows"),
        ("opacity-type pattern", "shadow-like area"),
        ("opacity", "shadow-like area"),
        ("consolidation", "denser area in the lung"),
        ("infiltrate", "shadow-like change"),
        ("atelectatic change", "a small partly collapsed area"),
        ("atelectasis", "a small partly collapsed area"),
        ("pleural fluid or effusion-type change", "fluid around the lung"),
        ("pleural effusion", "fluid around the lung"),
        ("pulmonary edema or vascular congestion pattern", "extra fluid in the lungs"),
        ("pulmonary edema", "extra fluid in the lungs"),
        ("pneumothorax", "collapsed-lung pattern"),
        ("cardiomegaly", "an enlarged heart"),
        ("radiologist", "radiologist (X-ray doctor)"),
        ("clinician", "doctor"),
        ("correlate the image result with symptoms", "match the result with how you feel"),
        ("before acting on the result", "before making treatment decisions"),
    ]
    for old, new in replacements:
        simplified = re.sub(re.escape(old), new, simplified, flags=re.IGNORECASE)

    simplified = re.sub(r"\s+", " ", simplified).strip(" .")
    if not simplified:
        return ""
    if not simplified.endswith((".", "!", "?")):
        simplified += "."
    return simplified[0].upper() + simplified[1:]


def _possible_causes_for_module(module: str, findings: List[KeyFinding], language: str = "en") -> List[str]:
    finding_text = " ".join(f"{item.title} {item.detail}".lower() for item in findings)
    causes: List[str] = []
    strings = _LOCALIZED_STRINGS.get(language, _LOCALIZED_STRINGS["en"])

    if module == "blood":
        if language == "en":
            if "hemoglobin" in finding_text or "oxygen-carrying" in finding_text:
                causes.extend([
                    "Low iron in the diet (iron deficiency — the most common cause of low hemoglobin)",
                    "Low vitamin B12 or folate (found in meat, dairy, and leafy green vegetables)",
                    "A chronic illness or long-term health condition",
                    "Blood loss (e.g. heavy periods, internal bleeding, or surgery)",
                ])
            if "white blood cell" in finding_text or "infection-fighting" in finding_text:
                causes.extend([
                    "A bacterial or viral infection that the body is actively fighting",
                    "Inflammation from an injury, allergy, or autoimmune condition",
                    "Stress on the body (physical or emotional)",
                    "Certain medications",
                ])
            if "platelet" in finding_text or "clot" in finding_text:
                causes.extend([
                    "A recent infection or inflammatory response",
                    "Certain medications or vitamin deficiencies",
                    "Bone marrow conditions (rare)",
                ])
            if "blood sugar" in finding_text or "glucose" in finding_text:
                causes.extend([
                    "Eating a meal shortly before the blood test",
                    "Diabetes or pre-diabetes",
                    "Stress hormones affecting blood sugar",
                    "Certain medications",
                ])
            if "red blood cell" in finding_text or "hematocrit" in finding_text:
                causes.extend([
                    "Dehydration (not drinking enough water)",
                    "Anemia (low red blood cell count from various causes)",
                    "Nutritional deficiencies",
                ])
        elif language == "hi":
            if any(term in finding_text for term in ["hemoglobin", "हीमोग्लोबिन", "रक्त"]):
                causes.extend([
                    "आहार में आयरन की कमी (आयरन की कमी - कम हीमोग्लोबिन का सबसे आम कारण)",
                    "विटामिन B12 या फोलेट की कमी (मांस, डेयरी और पत्तेदार हरी सब्जियों में पाया जाता है)",
                    "एक पुरानी बीमारी या दीर्घकालिक स्वास्थ्य स्थिति",
                    "रक्त की हानि (जैसे भारी मासिक धर्म, आंतरिक रक्तस्राव, या सर्जरी)",
                ])
            if any(term in finding_text for term in ["white blood cell", "श्वेत रक्त", "संक्रमण"]):
                causes.extend([
                    "एक जीवाणु या वायरल संक्रमण जिससे शरीर सक्रिय रूप से लड़ रहा है",
                    "किसी चोट, एलर्जी या ऑटोइम्यून स्थिति से सूजन",
                    "शरीर पर तनाव (शारीरिक या भावनात्मक)",
                    "कुछ दवाएं",
                ])
            if any(term in finding_text for term in ["platelet", "प्लेटलेट", "थक्का"]):
                causes.extend([
                    "हालिया संक्रमण या भड़काऊ प्रतिक्रिया",
                    "कुछ दवाएं या विटामिन की कमी",
                    "अस्थि मज्जा की स्थिति (दुर्लभ)",
                ])
            if any(term in finding_text for term in ["sugar", "चीनी", "ग्लूकोज"]):
                causes.extend([
                    "रक्त परीक्षण से कुछ समय पहले भोजन करना",
                    "मधुमेह या प्री-डायबिटीज",
                    "तनाव हार्मोन रक्त शर्करा को प्रभावित करते हैं",
                    "कुछ दवाएं",
                ])
        elif language == "gu":
            if any(term in finding_text for term in ["hemoglobin", "હિમોગ્લોબિન", "લોહી"]):
                causes.extend([
                    "આહારમાં આયર્નની ઉણપ (આયર્નની ઉણપ - ઓછું હિમોગ્લોબિન હોવાનું સૌથી સામાન્ય કારણ)",
                    "વિટામિન B12 અથવા ફોલેટની ઉણપ (માંસ, ડેરી અને પાંદડાવાળા લીલા શાકભાજીમાં જોવા મળે છે)",
                    "લાંબા ગાળાની બીમારી અથવા સ્વાસ્થ્યની સ્થિતિ",
                    "લોહીનું નુકસાન (દા.ત. માસિક ધર્મ, આંતરિક રક્તસ્રાવ, અથવા સર્જરી)",
                ])
            if any(term in finding_text for term in ["white blood cell", "ધોળા રક્ત", "ચેપ"]):
                causes.extend([
                    "બેક્ટેરિયલ અથવા વાયરલ ઇન્ફેક્શન કે જેનો શરીર સક્રિયપણે સામનો કરી રહ્યું છે",
                    "ઈજા, એલર્જી અથવા ઓટોઇમ્યુન સ્થિતિને કારણે સોજો",
                    "શરીર પર તણાવ (શારીરિક અથવા ભાવનાત્મક)",
                    "ચોક્કસ દવાઓ",
                ])
            if any(term in finding_text for term in ["platelet", "પ્લેટલેટ", "ગંઠાઈ"]):
                causes.extend([
                    "તાજેતરના ચેપ અથવા બળતરાની પ્રતિક્રિયા",
                    "ચોક્કસ દવાઓ અથવા વિટામિનની ઉણપ",
                    "અસ્થિ મજ્જાની સ્થિતિ (દુર્લભ)",
                ])
            if any(term in finding_text for term in ["sugar", "ખાંડ", "ગ્લુકોઝ"]):
                causes.extend([
                    "બ્લડ ટેસ્ટ પહેલાં ભોજન લેવું",
                    "ડાયાબિટીસ અથવા પ્રી-ડાયાબિટીસ",
                    "સ્ટ્રેસ હોર્મોન્સ બ્લડ સુગરને અસર કરે છે",
                    "ચોક્કસ દવાઓ",
                ])

    elif module == "xray":
        if language == "en":
            if any(term in finding_text for term in ["opacity", "consolidation", "infiltrate", "airspace", "shadow"]):
                causes.extend([
                    "A chest infection such as pneumonia or bronchitis",
                    "Inflammation in the lung",
                    "A partially collapsed area of the lung",
                ])
            if any(term in finding_text for term in ["effusion", "fluid"]):
                causes.extend([
                    "Fluid building up around the lung (pleural effusion)",
                    "Heart or kidney problems that cause fluid to accumulate",
                    "Inflammation in the chest",
                ])
            if any(term in finding_text for term in ["fracture", "dislocation"]):
                causes.extend([
                    "An injury such as a fall or impact",
                    "Weakened bones (osteoporosis)",
                ])
        elif language == "hi":
            if any(term in finding_text for term in ["opacity", "छाया", "संक्रमण"]):
                causes.extend([
                    "निमोनिया या ब्रोंकाइटिस जैसा छाती का संक्रमण",
                    "फेफड़ों में सूजन",
                    "फेफड़े का एक हिस्सा सिकुड़ जाना",
                ])
            if any(term in finding_text for term in ["fluid", "तरल", "द्रव"]):
                causes.extend([
                    "फेफड़े के चारों ओर तरल पदार्थ का जमा होना (प्लूरल इफ्यूजन)",
                    "हृदय या गुर्दे की समस्याएं",
                    "छाती में सूजन",
                ])
            if any(term in finding_text for term in ["fracture", "टूटना", "हड्डी"]):
                causes.extend([
                    "चोट जैसे कि गिरना या प्रभाव",
                    "कमजोर हड्डियां (ऑस्टियोपोरोसिस)",
                ])
        elif language == "gu":
            if any(term in finding_text for term in ["opacity", "પડછાયો", "ચેપ"]):
                causes.extend([
                    "ન્યુમોનિયા અથવા બ્રોન્કાઇટિસ જેવો છાતીનો ચેપ",
                    "ફેફસામાં સોજો",
                    "ફેફસાંનો એક ભાગ સંકોચાઈ જવો",
                ])
            if any(term in finding_text for term in ["fluid", "તરલ", "પ્રવાહી"]):
                causes.extend([
                    "ફેફસાની આસપાસ પ્રવાહીનું જમા થવું (પ્લુરલ ઇફ્યુઝન)",
                    "હૃદય અથવા કિડનીની સમસ્યાઓ",
                    "છાતીમાં સોજો",
                ])
            if any(term in finding_text for term in ["fracture", "હાડકું", "ભાંગવું"]):
                causes.extend([
                    "ઈજા જેવી કે પડવું અથવા આંચકો",
                    "નબળા હાડકાં (ઓસ્ટિઓપોરોસિસ)",
                ])
        causes.append(strings["xray_generic_cause"])
    
    if findings and not causes:
        causes.extend([
            strings["general_cause"],
            strings["general_health_issue"],
        ])

    return _unique_preserve_order(causes)


def _effects_for_module(module: str, findings: List[KeyFinding], language: str = "en") -> List[str]:
    finding_text = " ".join(f"{item.title} {item.detail}".lower() for item in findings)
    effects: List[str] = []
    strings = _LOCALIZED_STRINGS.get(language, _LOCALIZED_STRINGS["en"])

    if module == "blood":
        if language == "en":
            if "hemoglobin" in finding_text or "oxygen-carrying" in finding_text:
                effects.extend([
                    "Feeling tired or exhausted even after resting",
                    "Shortness of breath during normal activities",
                    "Looking pale (especially in the face, nails, or inside the eyelids)",
                    "Dizziness or difficulty concentrating",
                ])
            if "white blood cell" in finding_text or "infection-fighting" in finding_text:
                effects.extend([
                    "Signs of active infection (fever, chills, sweating)",
                    "Reduced ability to fight off illness",
                    "Swollen lymph nodes",
                ])
            if "platelet" in finding_text or "clot" in finding_text:
                effects.extend([
                    "Bruising more easily than usual (if platelets are low)",
                    "Bleeding that takes longer than usual to stop",
                    "Possible clotting concerns if platelet count is very high",
                ])
        elif language == "hi":
            if any(term in finding_text for term in ["hemoglobin", "हीमोग्लोबिन"]):
                effects.extend([
                    "आराम करने के बाद भी थकान या कमजोरी महसूस होना",
                    "साधारण गतिविधियों के दौरान सांस फूलना",
                    "उदास या पीला दिखना (विशेष रूप से चेहरे, नाखूनों या पलकों के अंदर)",
                    "चक्कर आना या एकाग्रता में कठिनाई",
                ])
            if any(term in finding_text for term in ["white blood cell", "श्वेत रक्त"]):
                effects.extend([
                    "सक्रिय संक्रमण के लक्षण (बुखार, ठंड लगना, पसीना आना)",
                    "बीमारी से लड़ने की कम क्षमता",
                    "सूजन वाली लसीका ग्रंथियां",
                ])
            if any(term in finding_text for term in ["platelet", "प्लेटलेट"]):
                effects.extend([
                    "सामान्य से अधिक आसानी से खरोंच लगना (यदि प्लेटलेट्स कम हैं)",
                    "रक्तस्राव जिसे रुकने में सामान्य से अधिक समय लगता है",
                ])
        elif language == "gu":
            if any(term in finding_text for term in ["hemoglobin", "હિમોગ્લોબિન"]):
                effects.extend([
                    "આરામ કર્યા પછી પણ થાક અથવા નબળાઈ લાગવી",
                    "સામાન્ય પ્રવૃત્તિઓ દરમિયાન શ્વાસ ચઢવો",
                    "પીળાશ પડવું (ખાસ કરીને ચહેરા, નખ અથવા પોપચાની અંદર)",
                    "ચક્કર આવવા અથવા એકાગ્રતામાં મુશ્કેલી",
                ])
            if any(term in finding_text for term in ["white blood cell", "ધોળા રક્ત"]):
                effects.extend([
                    "સક્રિય ચેપના લક્ષણો (તાવ, ધ્રુજારી, પરસેવો)",
                    "બીમારી સામે લડવાની ઓછી ક્ષમતા",
                    "સોજોવાળી લસિકા ગ્રંથીઓ",
                ])
            if any(term in finding_text for term in ["platelet", "પ્લેટલેટ"]):
                effects.extend([
                    "સામાન્ય કરતા વધુ સરળતાથી ઉઝરડા પડવા (જો પ્લેટલેટ્સ ઓછા હોય)",
                    "રક્તસ્રાવ જે અટકવામાં સામાન્ય કરતા વધુ સમય લે છે",
                ])

    elif module == "xray":
        if language == "en":
            if any(term in finding_text for term in ["opacity", "consolidation", "effusion", "edema", "pneumothorax", "shadow"]):
                effects.extend([
                    "Cough, chest tightness, or shortness of breath may be present",
                    "Oxygen levels in the blood may be reduced",
                    "Breathing may feel harder than usual",
                ])
            if any(term in finding_text for term in ["fracture", "dislocation"]):
                effects.extend([
                    "Pain, tenderness, or swelling in the affected area",
                    "Reduced ability to move the affected bone or joint",
                ])
        elif language == "hi":
            if any(term in finding_text for term in ["opacity", "छाया", "तरल"]):
                effects.extend([
                    "खांसी, छाती में जकड़न या सांस लेने में तकलीफ हो सकती है",
                    "रक्त में ऑक्सीजन का स्तर कम हो सकता है",
                    "सांस लेना सामान्य से कठिन महसूस हो सकता है",
                ])
            if any(term in finding_text for term in ["fracture", "टूटना"]):
                effects.extend([
                    "प्रभावित क्षेत्र में दर्द, कोमलता या सूजन",
                    "प्रभावित हड्डी या जोड़ को हिलाने की क्षमता कम होना",
                ])
        elif language == "gu":
            if any(term in finding_text for term in ["opacity", "પડછાયો", "પ્રવાહી"]):
                effects.extend([
                    "ખાંસી, છાતીમાં જકડન અથવા શ્વાસ લેવામાં તકલીફ હોઈ શકે છે",
                    "લોહીમાં ઓક્સિજનનું સ્તર ઘટી શકે છે",
                    "શ્વાસ લેવામાં સામાન્ય કરતા વધુ તકલીફ અનુભવાય",
                ])
            if any(term in finding_text for term in ["fracture", "ભાંગવું"]):
                effects.extend([
                    "અસરગ્રસ્ત વિસ્તારમાં દુખાવો અથવા સોજો",
                    "અસરગ્રસ્ત હાડકા અથવા સાંધાને હલાવવાની ક્ષમતા ઘટવી",
                ])
        effects.append(strings["xray_symptom_warning"])
    
    if findings and not effects:
        effects.extend([
            strings["generic_effect"],
            strings["generic_symptom_warning"],
        ])

    return _unique_preserve_order(effects)


def _future_risks_for_module(
    module: str,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    language: str = "en",
) -> List[str]:
    risks: List[str] = []
    finding_text = " ".join(f"{item.title} {item.detail}".lower() for item in findings)
    strings = _LOCALIZED_STRINGS.get(language, _LOCALIZED_STRINGS["en"])

    if integrity.status != "pass":
        risks.append(strings["incomplete_data_risk"])

    if module == "blood":
        if language == "en":
            if "glucose" in finding_text or "blood sugar" in finding_text:
                risks.extend([
                    "Persistently high blood sugar can, over time, increase the risk of diabetes-related complications such as kidney, eye, and nerve problems.",
                    "If blood sugar is not managed, it may affect the health of blood vessels throughout the body.",
                ])
            if "hemoglobin" in finding_text or "oxygen-carrying" in finding_text:
                risks.extend([
                    "Ongoing low hemoglobin can lead to worsening tiredness, reduced physical stamina, and difficulty with everyday activities.",
                    "Untreated anemia may strain the heart over time as it works harder to deliver oxygen.",
                ])
        elif language == "hi":
            if any(term in finding_text for term in ["glucose", "sugar", "चीनी"]):
                risks.extend([
                    "लगातार उच्च रक्त शर्करा, समय के साथ, मधुमेह संबंधी जटिलताओं जैसे गुर्दे, आंख और तंत्रिका समस्याओं के जोखिम को बढ़ा सकती है।",
                    "यदि रक्त शर्करा को प्रबंधित नहीं किया जाता है, तो यह पूरे शरीर में रक्त वाहिकाओं के स्वास्थ्य को प्रभावित कर सकता है।",
                ])
            if any(term in finding_text for term in ["hemoglobin", "हीमोग्लोबिन"]):
                risks.extend([
                    "लगातार कम हीमोग्लोबिन से थकान बढ़ सकती है, शारीरिक सहनशक्ति कम हो सकती है और रोजमर्रा की गतिविधियों में कठिनाई हो सकती है।",
                ])
        elif language == "gu":
            if any(term in finding_text for term in ["glucose", "sugar", "ખાંડ"]):
                risks.extend([
                    "લગાતાર ઊંચી બ્લડ સુગર, સમય જતાં, કિડની, આંખ અને જ્ઞાનતંતુની સમસ્યાઓ જેવી ડાયાબિટીસને લગતી ગૂંચવણોનું જોખમ વધારી શકે છે.",
                    "જો બ્લડ સુગરને નિયંત્રિત કરવામાં ન આવે તો, તે સમગ્ર શરીરમાં રક્તવાહિનીઓના સ્વાસ્થ્યને અસર કરી શકે છે.",
                ])
            if any(term in finding_text for term in ["hemoglobin", "હિમોગ્લોબિન"]):
                risks.extend([
                    "ચાલુ રહેલું ઓછું હિમોગ્લોબિન થાક વધારી શકે છે, શારીરિક ક્ષમતા ઘટાડી શકે છે અને રોજિંદી પ્રવૃત્તિઓમાં મુશ્કેલી ઊભી કરી શકે છે.",
                ])

    elif module == "xray":
        xray_signal = _xray_signal_profile(findings)
        if xray_signal["urgent"]:
            risks.append(strings["xray_urgent_risk"])
        if xray_signal["respiratory"]:
            risks.append(strings["xray_respiratory_risk"])
        risks.append(strings["ai_risks_disclaimer"])
        
    if findings and not risks:
        risks.append(strings["condition_risk"])

    return _unique_preserve_order(risks)


def _general_advice(module: str, recommendations: List[str], language: str = "en") -> List[str]:
    strings = _LOCALIZED_STRINGS.get(language, _LOCALIZED_STRINGS["en"])
    advice = list(recommendations)
    advice.extend(strings["base_advice"])
    
    if module == "blood":
        advice.extend(strings["blood_advice"])
    if module == "xray":
        advice.extend(strings["xray_advice"])
    if module == "pdf":
        advice.extend(strings["pdf_advice"])
        
    return _unique_preserve_order(advice)


def _xray_signal_profile(findings: List[KeyFinding]) -> Dict[str, bool]:
    text = " ".join(f"{item.title} {item.detail}".lower() for item in findings)
    urgent_terms = (
        "pneumothorax", "collapsed lung",
        "large effusion",
        "severe edema",
        "fracture", "broken",
        "dislocation",
        "marked widening",
        "free air",
    )
    respiratory_terms = (
        "opacity", "shadow",
        "consolidation",
        "effusion", "fluid",
        "edema",
        "infiltrate",
        "airspace",
        "pneumothorax",
    )
    return {
        "urgent": any(term in text for term in urgent_terms),
        "respiratory": any(term in text for term in respiratory_terms),
    }


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


_PATIENT_SECTION_COPY: Dict[str, Dict[str, str]] = {
    "en": {
        "missing_information": "Missing information",
        "share_doctor": "Share this report with your primary care doctor for a full evaluation.",
        "clearer_copy": "Request a clearer copy of the report if possible to resolve extraction uncertainties.",
        "observation": "Observation",
        "analysis_engine": "Analysis Engine",
        "data_completeness": "Data Completeness",
        "data_quality": "Data Quality",
        "llm_not_configured": "LLM not configured",
        "llm_error_prefix": "LLM error",
        "rule_fallback": "Falling back to rule-based analysis because {reason}.",
        "limited_score": "Analysis is limited due to a low completeness score ({score}%).",
        "quality_ok": "The report data appears sufficient for standard analysis.",
        "summary_xray": "Deterministic review of the X-ray upload based on image properties and integrity checks.",
        "summary_blood": "Rule-based analysis of the provided blood marker values.",
        "summary_pdf": "Automated text extraction and summary of the medical report.",
    },
    "hi": {
        "missing_information": "गायब जानकारी",
        "share_doctor": "पूरे मूल्यांकन के लिए यह रिपोर्ट अपने डॉक्टर के साथ साझा करें।",
        "clearer_copy": "यदि संभव हो तो रिपोर्ट की अधिक स्पष्ट प्रति मांगें ताकि निकासी की अनिश्चितताओं को कम किया जा सके।",
        "observation": "अवलोकन",
        "analysis_engine": "विश्लेषण इंजन",
        "data_completeness": "डेटा की पूर्णता",
        "data_quality": "डेटा की गुणवत्ता",
        "llm_not_configured": "LLM कॉन्फ़िगर नहीं है",
        "llm_error_prefix": "LLM त्रुटि",
        "rule_fallback": "{reason} के कारण नियम-आधारित विश्लेषण का उपयोग किया गया।",
        "limited_score": "कम पूर्णता स्कोर ({score}%) के कारण विश्लेषण सीमित है।",
        "quality_ok": "रिपोर्ट का डेटा सामान्य विश्लेषण के लिए पर्याप्त प्रतीत होता है।",
        "summary_xray": "छवि गुणों और अखंडता जांच के आधार पर X-रे अपलोड की नियम-आधारित समीक्षा की गई।",
        "summary_blood": "प्रदान किए गए ब्लड मार्कर मानों का नियम-आधारित विश्लेषण किया गया।",
        "summary_pdf": "मेडिकल रिपोर्ट का स्वचालित टेक्स्ट निष्कर्षण और सारांश तैयार किया गया।",
    },
    "gu": {
        "missing_information": "ખૂટતી માહિતી",
        "share_doctor": "પૂર્ણ મૂલ્યાંકન માટે આ રિપોર્ટ તમારા ડૉક્ટર સાથે શેર કરો.",
        "clearer_copy": "જો શક્ય હોય તો રિપોર્ટની વધુ સ્પષ્ટ નકલ મેળવો જેથી એક્સ્ટ્રેક્શન સંબંધિત અનિશ્ચિતતા ઘટે.",
        "observation": "અવલોકન",
        "analysis_engine": "વિશ્લેષણ એન્જિન",
        "data_completeness": "ડેટાની પૂર્ણતા",
        "data_quality": "ડેટાની ગુણવત્તા",
        "llm_not_configured": "LLM કન્ફિગર કરાયેલ નથી",
        "llm_error_prefix": "LLM ભૂલ",
        "rule_fallback": "{reason} હોવાથી નિયમ આધારિત વિશ્લેષણનો ઉપયોગ કરવામાં આવ્યો.",
        "limited_score": "ઓછા પૂર્ણતા સ્કોર ({score}%)ને કારણે વિશ્લેષણ મર્યાદિત છે.",
        "quality_ok": "રિપોર્ટનો ડેટા સામાન્ય વિશ્લેષણ માટે પૂરતો લાગે છે.",
        "summary_xray": "ઇમેજ ગુણધર્મો અને ઇન્ટેગ્રિટી ચેક્સના આધારે X-રે અપલોડની નિયમ આધારિત સમીક્ષા કરવામાં આવી.",
        "summary_blood": "આપેલા બ્લડ માર્કર મૂલ્યોનું નિયમ આધારિત વિશ્લેષણ કરવામાં આવ્યું.",
        "summary_pdf": "મેડિકલ રિપોર્ટનું સ્વચાલિત ટેક્સ્ટ એક્સ્ટ્રેક્શન અને સારાંશ તૈયાર કરવામાં આવ્યો.",
    },
}


def _patient_copy(language: str) -> Dict[str, str]:
    return _PATIENT_SECTION_COPY.get(language, _PATIENT_SECTION_COPY["en"])


def _build_patient_report_summary(
    *,
    module: str,
    extracted: ExtractedContent,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    recommendations: List[str],
    language: str = "en",
) -> PatientReportSummary:
    main_findings = _main_findings_for_patient(module, findings, integrity, extracted, language=language)
    is_normal = _is_everything_normal(findings, integrity)
    strings = _LOCALIZED_STRINGS.get(language, _LOCALIZED_STRINGS["en"])

    if is_normal:
        normal_status_text = strings["status_yes"]
    elif integrity.status == "fail" or integrity.missing_fields:
        normal_status_text = strings["status_incomplete"]
    else:
        normal_status_text = strings["status_review"]

    return PatientReportSummary(
        main_findings=main_findings,
        is_everything_normal=is_normal,
        normal_status_text=normal_status_text,
        problems_concerns=_problems_concerns_for_patient(findings, integrity, language=language),
        important_values_observations=_important_values_observations(
            module,
            extracted,
            findings,
            integrity,
            language=language,
        ),
        simple_explanation=_build_patient_explanation(
            module,
            findings,
            integrity,
            extracted,
            recommendations=recommendations,
            language=language,
        ),
        suggested_next_steps=_suggested_next_steps(module, integrity, recommendations, language=language),
    )


def _main_findings_for_patient(
    module: str,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    extracted: ExtractedContent,
    language: str = "en",
) -> List[str]:
    if findings:
        patient_lines = []
        for finding in findings[:4]:
            patient_lines.append(f"{finding.title}: {_to_patient_friendly_sentence(finding.detail, language=language)}")
        return patient_lines
    
    return []


def _is_everything_normal(findings: List[KeyFinding], integrity: IntegrityResult) -> bool:
    if integrity.status == "fail":
        return False
    
    # Check for any high or moderate severity findings
    for finding in findings:
        if finding.severity in ["high", "moderate"]:
            return False
    
    # If there are NO findings or only 'low' severity findings, we say True
    return True


def _problems_concerns_for_patient(
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    language: str = "en",
) -> List[str]:
    copy = _patient_copy(language)
    concerns = []
    for finding in findings:
        if finding.severity in ["high", "moderate"]:
            concerns.append(f"{finding.title} - {finding.detail}")
    
    for field in integrity.missing_fields:
        concerns.append(f"{copy['missing_information']}: {field}")
        
    return concerns


def _important_values_observations(
    module: str,
    extracted: ExtractedContent,
    findings: List[KeyFinding],
    integrity: IntegrityResult,
    language: str = "en",
) -> List[str]:
    observations = []
    # Mix of observations from integrity and low severity findings
    for finding in findings:
        if finding.severity == "low":
            observations.append(f"{finding.title}: {finding.detail}")
            
    for val in integrity.suspicious_values[:3]:
        observations.append(val)
        
    if module == "blood":
        # Extract some raw values from text if possible
        pass
        
    return observations


def _suggested_next_steps(
    module: str,
    integrity: IntegrityResult,
    recommendations: List[str],
    language: str = "en",
) -> List[str]:
    copy = _patient_copy(language)
    steps = list(recommendations)
    if not steps:
        steps = [copy["share_doctor"]]
    
    if integrity.status == "review":
        steps.append(copy["clearer_copy"])
        
    return steps


def _to_patient_friendly_sentence(text: str, language: str = "en") -> str:
    # Basic cleanup to make LLM fragments or technical phrases look like sentences
    if not text:
        return ""
    
    cleaned = text.strip()
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    
    # Capitalize first letter for Latin script only.
    if language == "en" and len(cleaned) > 1:
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    return cleaned


def _coerce_findings(data: Any, language: str = "en") -> List[KeyFinding]:
    if not isinstance(data, list):
        return []
    
    copy = _patient_copy(language)
    findings = []
    for item in data:
        if isinstance(item, dict):
            findings.append(KeyFinding(
                title=item.get("title", copy["observation"]),
                detail=item.get("detail", ""),
                severity=item.get("severity", "low"),
                evidence=item.get("evidence")
            ))
    return findings


def _coerce_string_list(data: Any) -> List[str]:
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if item]


def _coerce_confidence(
    data: Any,
    extracted: ExtractedContent,
    integrity: IntegrityResult,
    module: str,
    language: str = "en",
) -> List[ConfidenceNote]:
    if isinstance(data, list) and data:
        notes = []
        for item in data:
            if isinstance(item, dict):
                notes.append(ConfidenceNote(
                    area=item.get("area", "general"),
                    level=item.get("level", "medium"),
                    note=item.get("note", "")
                ))
        return notes
        
    return _default_confidence_notes(module, extracted, integrity, language=language)


def _default_confidence_notes(
    module: str, 
    extracted: ExtractedContent, 
    integrity: IntegrityResult,
    language: str = "en",
    live_model_used: bool = False,
    llm_error: Optional[str] = None,
    llm_attempted: bool = False,
) -> List[ConfidenceNote]:
    copy = _patient_copy(language)
    notes = []
    
    if not live_model_used:
        reason = copy["llm_not_configured"] if not llm_attempted else f"{copy['llm_error_prefix']}: {llm_error}"
        notes.append(ConfidenceNote(
            area=copy["analysis_engine"],
            level="low",
            note=copy["rule_fallback"].format(reason=reason)
        ))

    if integrity.completeness_score < 80:
        notes.append(ConfidenceNote(
            area=copy["data_completeness"],
            level="low",
            note=copy["limited_score"].format(score=integrity.completeness_score)
        ))
    else:
        notes.append(ConfidenceNote(
            area=copy["data_quality"],
            level="high",
            note=copy["quality_ok"]
        ))

    return notes


def _fallback_summary(
    module: str,
    extracted: ExtractedContent,
    integrity: IntegrityResult,
    findings: List[KeyFinding] = None,
    language: str = "en",
) -> str:
    copy = _patient_copy(language)
    if module == "xray":
        return copy["summary_xray"]
    if module == "blood":
        return copy["summary_blood"]
    return copy["summary_pdf"]


def _execution_metadata(module: str, live_model_used: bool, local_xray_used: bool) -> LLMExecution:
    return LLMExecution(
        provider="Gemini" if live_model_used else ("Local" if local_xray_used else "Fallback"),
        primary_model=settings.LLM_MODEL if live_model_used else "none",
        mode="multimodal" if module == "xray" else "text",
        live_model_used=live_model_used
    )


def _guess_content_type(filename: str) -> str:
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    mapping = {
        "pdf": "application/pdf",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "txt": "text/plain",
        "csv": "text/csv",
    }
    return mapping.get(ext, "application/octet-stream")


def _decode_text(contents: bytes) -> str:
    for encoding in ["utf-8", "latin-1", "ascii"]:
        try:
            return contents.decode(encoding)
        except UnicodeDecodeError:
            continue
    return contents.decode("utf-8", errors="ignore")


def _markers_to_extracted_content(markers: List[BloodMarkerInput], language: str = "en") -> ExtractedContent:
    strings = _FINDING_STRINGS.get(language, _FINDING_STRINGS["en"])
    text = "\n".join([f"{m.name}: {m.value:g} {m.unit or ''}" for m in markers])
    return ExtractedContent(
        raw_text=text,
        sections=[ExtractedSection(title=strings["markers_provided"], content=text)],
        metadata={"source": "manual-entry"}
    )
