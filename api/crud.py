"""Operasi database untuk hasil prediksi."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from api.db_models import Detection, Prediction


def create_prediction(
    db: Session,
    *,
    filename: str,
    model_key: str,
    model_name: str,
    conf_threshold: float,
    image_width: int,
    image_height: int,
    inference_ms: float,
    detections: list[dict[str, Any]],
    annotated_path: str | None = None,
    notes: str | None = None,
) -> Prediction:
    top = detections[0] if detections else None
    row = Prediction(
        filename=filename,
        model_key=model_key,
        model_name=model_name,
        conf_threshold=conf_threshold,
        image_width=image_width,
        image_height=image_height,
        n_detections=len(detections),
        top_class=top["class_name"] if top else None,
        top_confidence=top["confidence"] if top else None,
        inference_ms=inference_ms,
        annotated_path=annotated_path,
        notes=notes,
    )
    # bbox disimpan dalam koordinat gambar ASLI agar bisa dipakai ulang
    for det in detections:
        box = det.get("box_original") or det["box"]
        row.detections.append(
            Detection(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                x1=box[0],
                y1=box[1],
                x2=box[2],
                y2=box[3],
            )
        )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_prediction(db: Session, pred_id: int) -> Prediction | None:
    return db.get(Prediction, pred_id)


def list_predictions(
    db: Session,
    *,
    model_key: str | None = None,
    class_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, list[Prediction]]:
    stmt = select(Prediction)
    count_stmt = select(func.count()).select_from(Prediction)

    if model_key:
        stmt = stmt.where(Prediction.model_key == model_key)
        count_stmt = count_stmt.where(Prediction.model_key == model_key)
    if class_name:
        stmt = stmt.where(Prediction.top_class == class_name)
        count_stmt = count_stmt.where(Prediction.top_class == class_name)

    total = db.execute(count_stmt).scalar_one()
    rows = (
        db.execute(
            stmt.order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return total, list(rows)


def delete_prediction(db: Session, pred_id: int) -> bool:
    row = db.get(Prediction, pred_id)
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def clear_predictions(db: Session) -> int:
    n = db.execute(select(func.count()).select_from(Prediction)).scalar_one()
    db.execute(delete(Prediction))
    db.commit()
    return n


def stats(db: Session, top_n: int = 10) -> dict[str, Any]:
    total_pred = db.execute(select(func.count()).select_from(Prediction)).scalar_one()
    total_det = db.execute(select(func.count()).select_from(Detection)).scalar_one()

    by_model = dict(
        db.execute(select(Prediction.model_key, func.count()).group_by(Prediction.model_key)).all()
    )
    avg_ms = {
        k: round(float(v), 2)
        for k, v in db.execute(
            select(Prediction.model_key, func.avg(Prediction.inference_ms)).group_by(
                Prediction.model_key
            )
        ).all()
    }
    top_classes = [
        {"class_name": name, "count": int(cnt), "avg_confidence": round(float(avg), 4)}
        for name, cnt, avg in db.execute(
            select(Detection.class_name, func.count(), func.avg(Detection.confidence))
            .group_by(Detection.class_name)
            .order_by(func.count().desc())
            .limit(top_n)
        ).all()
    ]
    last: datetime | None = db.execute(select(func.max(Prediction.created_at))).scalar_one()

    return {
        "total_predictions": total_pred,
        "total_detections": total_det,
        "by_model": by_model,
        "avg_inference_ms": avg_ms,
        "top_classes": top_classes,
        "last_prediction_at": last,
    }
