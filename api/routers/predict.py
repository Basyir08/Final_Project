"""Endpoint inferensi: upload gambar X-ray -> deteksi fraktur."""

from __future__ import annotations

import base64
import uuid
from datetime import UTC, datetime
from typing import Any

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from api import crud
from api.config import settings
from api.database import get_db
from api.inference import (
    DISPLAY_NAME,
    MODEL_FRCNN,
    MODEL_YOLO,
    ModelNotAvailable,
    get_detector,
)
from api.preprocessing import (
    MODEL_COLORS,
    decode_image,
    draw_detections,
    encode_png,
)
from api.schemas import ComparePair, ModelKey, PredictResponse

router = APIRouter(prefix="/predict", tags=["prediction"])


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File kosong.")
    mb = len(raw) / 1024 / 1024
    if mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Ukuran file {mb:.1f} MB melebihi batas {settings.MAX_UPLOAD_MB} MB.",
        )
    return raw


def _save_annotated(img_bgr, model_key: str, filename: str) -> tuple[str, str]:
    """Simpan gambar beranotasi ke disk; kembalikan (path, url publik)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{model_key}_{uuid.uuid4().hex[:8]}.png"
    path = settings.STORAGE_DIR / name
    cv2.imwrite(str(path), img_bgr)
    return str(path), f"/static/{name}"


def _run(
    model_key: str,
    raw: bytes,
    filename: str,
    conf: float,
    save: bool,
    return_image: bool,
    db: Session,
) -> dict[str, Any]:
    img = decode_image(raw)
    if img is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"'{filename}' bukan file gambar yang valid.",
        )

    try:
        result = get_detector(model_key).predict(img, conf=conf)
    except ModelNotAvailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    detections = result["detections"]
    h, w = img.shape[:2]

    # anotasi digambar di atas gambar ASLI user (koordinat sudah dipetakan balik)
    color = MODEL_COLORS.get(model_key, (0, 0, 255))
    annotated = draw_detections(img, detections, color=color, box_key="box_original")

    annotated_path = annotated_url = None
    if save:
        annotated_path, annotated_url = _save_annotated(annotated, model_key, filename)

    top = detections[0] if detections else None
    payload: dict[str, Any] = {
        "prediction_id": None,
        "model_key": model_key,
        "model_name": DISPLAY_NAME[model_key],
        "filename": filename,
        "conf_threshold": result["conf_threshold"],
        "image_width": w,
        "image_height": h,
        "inference_ms": result["inference_ms"],
        "n_detections": len(detections),
        "top_class": top["class_name"] if top else None,
        "top_confidence": top["confidence"] if top else None,
        "detections": detections,
        "annotated_url": annotated_url,
        "annotated_image_b64": (
            base64.b64encode(encode_png(annotated)).decode() if return_image else None
        ),
    }

    if save:
        row = crud.create_prediction(
            db,
            filename=filename,
            model_key=model_key,
            model_name=DISPLAY_NAME[model_key],
            conf_threshold=result["conf_threshold"],
            image_width=w,
            image_height=h,
            inference_ms=result["inference_ms"],
            detections=detections,
            annotated_path=annotated_path,
        )
        payload["prediction_id"] = row.id

    return payload


@router.post(
    "/{model_key}",
    response_model=PredictResponse,
    summary="Prediksi dengan satu model (yolo26 atau frcnn)",
)
async def predict(
    model_key: ModelKey,
    file: UploadFile = File(..., description="Gambar X-ray (jpg/png/bmp/tif)"),
    conf: float = Form(settings.DEFAULT_CONF, ge=0.0, le=1.0),
    save: bool = Form(True, description="Simpan hasil ke database"),
    return_image: bool = Form(True, description="Sertakan gambar beranotasi (PNG base64)"),
    db: Session = Depends(get_db),
) -> PredictResponse:
    raw = await _read_upload(file)
    data = _run(model_key.value, raw, file.filename or "upload", conf, save, return_image, db)
    return PredictResponse(**data)


@router.post(
    "/compare/both",
    response_model=ComparePair,
    summary="Jalankan kedua model pada satu gambar yang sama",
)
async def predict_compare(
    file: UploadFile = File(...),
    conf: float = Form(settings.DEFAULT_CONF, ge=0.0, le=1.0),
    save: bool = Form(True),
    return_image: bool = Form(True),
    db: Session = Depends(get_db),
) -> ComparePair:
    raw = await _read_upload(file)
    fname = file.filename or "upload"
    return ComparePair(
        yolo26=PredictResponse(**_run(MODEL_YOLO, raw, fname, conf, save, return_image, db)),
        frcnn=PredictResponse(**_run(MODEL_FRCNN, raw, fname, conf, save, return_image, db)),
    )
