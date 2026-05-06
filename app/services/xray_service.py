import io
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.config import settings
from app.schemas.models import ConfidenceNote, IntegrityResult, KeyFinding


@dataclass
class XRayAnalysisPayload:
    summary: str
    key_findings: List[KeyFinding]
    recommendations: List[str]
    confidence_notes: List[ConfidenceNote]
    metadata: Dict[str, Any]


class LocalXRayModelService:
    def __init__(self) -> None:
        self.model_path = settings.LOCAL_XRAY_MODEL_PATH.strip()
        self.top_k = settings.LOCAL_XRAY_TOP_K
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._load_error: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.model_path)

    @property
    def model_name(self) -> str:
        return self.model_path or "local-xray-model"

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def analyze_image(self, image_bytes: bytes, content_type: str, integrity: IntegrityResult) -> XRayAnalysisPayload:
        image = self._load_image(image_bytes)
        width, height = image.size
        image_quality_notes = _image_quality_notes(image)
        processor, model = self._get_or_load_model()

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probabilities = logits.softmax(dim=0)
        top_count = min(self.top_k, int(probabilities.shape[0]))
        top_indices = probabilities.topk(top_count).indices.tolist()

        predictions: List[Tuple[str, float]] = []
        for index in top_indices:
            label = model.config.id2label.get(int(index), f"class_{index}")
            score = float(probabilities[int(index)].item())
            predictions.append((label, score))

        findings = _predictions_to_findings(predictions)
        summary = _build_summary(predictions, image_quality_notes)
        recommendations = _recommendations_from_predictions(predictions, image_quality_notes)
        confidence_notes = _confidence_notes(
            predictions=predictions,
            integrity=integrity,
            image_quality_notes=image_quality_notes,
            width=width,
            height=height,
            model_name=self.model_name,
        )

        return XRayAnalysisPayload(
            summary=summary,
            key_findings=findings,
            recommendations=recommendations,
            confidence_notes=confidence_notes,
            metadata={
                "model_name": self.model_name,
                "image_width": width,
                "image_height": height,
                "predictions": [{"label": label, "score": round(score, 4)} for label, score in predictions],
            },
        )

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("The uploaded file could not be opened as an image.") from exc

    def _get_or_load_model(self) -> Tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "Local X-ray inference requires PyTorch. Install backend dependencies from requirements.txt first."
            )

        try:
            self._processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModelForImageClassification.from_pretrained(self.model_path, local_files_only=True)
        except Exception as exc:
            self._load_error = str(exc)
            raise RuntimeError(
                "Local X-ray model could not be loaded. Put a Hugging Face image-classification model "
                f"inside LOCAL_XRAY_MODEL_PATH: {self.model_path}"
            ) from exc

        return self._processor, self._model


def _normalize_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").split()).strip().lower()


def _predictions_to_findings(predictions: List[Tuple[str, float]]) -> List[KeyFinding]:
    findings: List[KeyFinding] = []
    for raw_label, score in predictions:
        normalized = _normalize_label(raw_label)
        if score < 0.2:
            continue
        severity = _severity_for_label(normalized, score)
        detail = _detail_for_label(normalized, score)
        evidence = f"Local model confidence {score * 100:.1f}% for label '{raw_label}'."
        findings.append(
            KeyFinding(
                title=_title_for_label(raw_label),
                detail=detail,
                severity=severity,
                evidence=evidence,
            )
        )

    if findings:
        return findings

    if predictions:
        label, score = predictions[0]
        return [
            KeyFinding(
                title="Low-confidence X-ray signal",
                detail=(
                    f"The local model's top label was '{label}' but confidence was only {score * 100:.1f}%, "
                    "so no strong abnormality was highlighted."
                ),
                severity="low",
                evidence="Low-confidence prediction from the local image model.",
            )
        ]

    return [
        KeyFinding(
            title="No prediction returned",
            detail="The local X-ray model did not return any classes for this image.",
            severity="moderate",
            evidence="Model output was empty.",
        )
    ]


def _severity_for_label(label: str, score: float) -> str:
    severe_terms = ("pneumothorax", "fracture", "dislocation", "effusion", "edema")
    moderate_terms = ("pneumonia", "opacity", "consolidation", "atelectasis", "cardiomegaly", "nodule")
    if any(term in label for term in severe_terms) and score >= 0.45:
        return "high"
    if any(term in label for term in moderate_terms) or score >= 0.5:
        return "moderate"
    return "low"


def _title_for_label(label: str) -> str:
    return " ".join(part.capitalize() for part in label.replace("_", " ").replace("-", " ").split())


def _detail_for_label(label: str, score: float) -> str:
    if "no finding" in label or "normal" in label:
        return (
            f"The local model did not highlight a confident acute abnormality and leaned toward a normal study "
            f"pattern with {score * 100:.1f}% confidence."
        )
    if "pneumonia" in label or "consolidation" in label or "opacity" in label:
        return (
            f"The image shows a lung opacity-type pattern that can be seen with infection, inflammation, or "
            f"atelectatic change. Model confidence was {score * 100:.1f}%."
        )
    if "effusion" in label:
        return f"The model suggests pleural fluid or effusion-type change with {score * 100:.1f}% confidence."
    if "edema" in label:
        return f"The model suggests a pulmonary edema or vascular congestion pattern with {score * 100:.1f}% confidence."
    if "cardiomegaly" in label or "enlarged cardiac silhouette" in label:
        return f"The model suggests an enlarged heart-size pattern with {score * 100:.1f}% confidence."
    if "fracture" in label or "dislocation" in label:
        return f"The model suggests a possible bony injury pattern with {score * 100:.1f}% confidence."
    return f"The local model highlighted '{label}' with {score * 100:.1f}% confidence."


def _build_summary(predictions: List[Tuple[str, float]], image_quality_notes: List[str]) -> str:
    if not predictions:
        return "The local X-ray model did not produce a usable prediction for this image."

    top_label, top_score = predictions[0]
    normalized = _normalize_label(top_label)
    quality_suffix = ""
    if image_quality_notes:
        quality_suffix = " Image quality limits may reduce reliability."

    if "no finding" in normalized or "normal" in normalized:
        return (
            f"The local model did not identify a strong acute abnormality and the top class was '{top_label}' "
            f"at {top_score * 100:.1f}% confidence.{quality_suffix}"
        )

    return (
        f"The local X-ray model's top signal was '{top_label}' at {top_score * 100:.1f}% confidence. "
        f"This is a supportive image-model prediction and still needs clinical or radiology confirmation.{quality_suffix}"
    )


def _recommendations_from_predictions(predictions: List[Tuple[str, float]], image_quality_notes: List[str]) -> List[str]:
    recommendations = ["Have a clinician or radiologist review the X-ray before acting on the result."]
    top_text = " ".join(label.lower() for label, _ in predictions[:3])

    if any(term in top_text for term in ("pneumothorax", "effusion", "fracture", "dislocation")):
        recommendations.append("Seek urgent review if the patient has severe pain, breathing symptoms, or trauma history.")
    if any(term in top_text for term in ("pneumonia", "opacity", "consolidation", "edema")):
        recommendations.append("Correlate the image result with symptoms such as fever, cough, chest pain, or shortness of breath.")
    if image_quality_notes:
        recommendations.append("Repeat or improve the X-ray view if the original image was dim, low-contrast, or poorly centered.")

    return _unique_preserve_order(recommendations)


def _confidence_notes(
    *,
    predictions: List[Tuple[str, float]],
    integrity: IntegrityResult,
    image_quality_notes: List[str],
    width: int,
    height: int,
    model_name: str,
) -> List[ConfidenceNote]:
    top_score = predictions[0][1] if predictions else 0.0
    prediction_level = "high" if top_score >= 0.75 else "medium" if top_score >= 0.45 else "low"
    notes = [
        ConfidenceNote(
            area="model",
            level="high",
            note=f"Local X-ray model '{model_name}' produced the analysis.",
        ),
        ConfidenceNote(
            area="prediction",
            level=prediction_level,
            note=(
                f"Top class confidence was {top_score * 100:.1f}%."
                if predictions
                else "The model returned no confident prediction."
            ),
        ),
        ConfidenceNote(
            area="image-size",
            level="high" if min(width, height) >= 512 else "medium",
            note=f"Input image size was {width}x{height} pixels.",
        ),
        ConfidenceNote(
            area="integrity",
            level="high" if integrity.status == "pass" else "medium",
            note=f"Upload integrity status is '{integrity.status}' with completeness score {integrity.completeness_score:g}.",
        ),
    ]
    if image_quality_notes:
        notes.append(
            ConfidenceNote(
                area="image-quality",
                level="medium",
                note=" ".join(image_quality_notes),
            )
        )
    return notes


def _image_quality_notes(image: Image.Image) -> List[str]:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    notes: List[str] = []
    mean_intensity = float(array.mean())
    contrast = float(array.std())

    if mean_intensity < 35:
        notes.append("The image appears quite dark, which can hide detail.")
    elif mean_intensity > 220:
        notes.append("The image appears very bright, which can wash out detail.")

    if contrast < 25:
        notes.append("The image has low contrast, so subtle findings may be harder for the model to detect.")

    width, height = image.size
    if width < 512 or height < 512:
        notes.append("The image resolution is relatively low for medical review.")

    return notes


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
