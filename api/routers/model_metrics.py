"""Endpoint metrik performa model (dibaca dari *_history.csv)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api import metrics
from api.inference import registry_status
from api.schemas import ModelInfo, ModelKey, TrainingHistory

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/models", response_model=list[ModelInfo], summary="Status kedua model")
def models_info() -> list[ModelInfo]:
    return [ModelInfo(**m) for m in registry_status()]


@router.get(
    "/training/{model_key}",
    response_model=TrainingHistory,
    summary="Riwayat training per epoch (untuk grafik performa)",
)
def training_history(model_key: ModelKey) -> TrainingHistory:
    try:
        return TrainingHistory(**metrics.history_payload(model_key.value))
    except metrics.HistoryNotAvailable as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/comparison", summary="Ringkasan perbandingan YOLO26 vs Faster R-CNN")
def comparison() -> dict:
    rows = metrics.comparison_table()
    if not rows:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Tidak ada file riwayat training di folder models/.",
        )
    return {"rows": rows}
