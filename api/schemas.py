"""Skema request/response (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ModelKey(str, Enum):
    yolo26 = "yolo26"
    frcnn = "frcnn"


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    created_at: datetime
    filename: str
    model_key: str
    model_name: str
    conf_threshold: float
    image_width: int
    image_height: int
    n_detections: int
    top_class: str | None
    top_confidence: float | None
    inference_ms: float
    annotated_url: str | None = None
    detections: list[DetectionOut] = []


class PredictResponse(BaseModel):
    """Response endpoint prediksi (dipakai langsung oleh Streamlit)."""

    model_config = ConfigDict(protected_namespaces=())

    prediction_id: int | None = Field(None, description="None bila save=false")
    model_key: str
    model_name: str
    filename: str
    conf_threshold: float
    image_width: int
    image_height: int
    inference_ms: float
    n_detections: int
    top_class: str | None
    top_confidence: float | None
    detections: list[dict]
    annotated_url: str | None = None
    annotated_image_b64: str | None = Field(None, description="PNG base64 bila return_image=true")


class ComparePair(BaseModel):
    yolo26: PredictResponse
    frcnn: PredictResponse


class PredictionList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionOut]


class ClassCount(BaseModel):
    class_name: str
    count: int
    avg_confidence: float


class StatsOut(BaseModel):
    total_predictions: int
    total_detections: int
    by_model: dict[str, int]
    avg_inference_ms: dict[str, float]
    top_classes: list[ClassCount]
    last_prediction_at: datetime | None


class ModelInfo(BaseModel):
    key: str
    name: str
    weights_path: str
    available: bool
    loaded: bool
    classes: list[str] | None = None
    meta: dict = {}


class BestEpoch(BaseModel):
    epoch: int
    mAP50: float | None = None
    mAP50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    val_loss: float | None = None


class TrainingHistory(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    model_name: str
    n_epochs: int
    best: BestEpoch
    columns: list[str]
    records: list[dict]


class HealthOut(BaseModel):
    status: str
    version: str
    device: str
    database: str
    models: list[ModelInfo]
