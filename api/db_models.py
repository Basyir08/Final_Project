"""Skema tabel: setiap prediksi dan setiap bbox-nya disimpan permanen."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    filename: Mapped[str] = mapped_column(String(255))
    model_key: Mapped[str] = mapped_column(String(32), index=True)  # yolo26 | frcnn
    model_name: Mapped[str] = mapped_column(String(64))
    conf_threshold: Mapped[float] = mapped_column(Float)

    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    n_detections: Mapped[int] = mapped_column(Integer, default=0)

    top_class: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    top_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    inference_ms: Mapped[float] = mapped_column(Float)
    annotated_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(512), nullable=True)

    detections: Mapped[list[Detection]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", lazy="selectin"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), index=True
    )

    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)

    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)

    prediction: Mapped[Prediction] = relationship(back_populates="detections")


Index("ix_predictions_model_created", Prediction.model_key, Prediction.created_at)
