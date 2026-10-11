"""Endpoint riwayat prediksi yang tersimpan di database."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from api import crud
from api.database import get_db
from api.db_models import Prediction
from api.schemas import PredictionList, PredictionOut, StatsOut

router = APIRouter(prefix="/predictions", tags=["history"])


def _to_out(row: Prediction) -> PredictionOut:
    out = PredictionOut.model_validate(row)
    if row.annotated_path:
        out.annotated_url = "/static/" + row.annotated_path.rsplit("/", 1)[-1]
    return out


@router.get("", response_model=PredictionList, summary="Daftar riwayat prediksi")
def list_predictions(
    model_key: str | None = Query(None, pattern="^(yolo26|frcnn)$"),
    class_name: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> PredictionList:
    total, rows = crud.list_predictions(
        db, model_key=model_key, class_name=class_name, limit=limit, offset=offset
    )
    return PredictionList(total=total, limit=limit, offset=offset, items=[_to_out(r) for r in rows])


@router.get("/stats", response_model=StatsOut, summary="Statistik agregat")
def prediction_stats(db: Session = Depends(get_db)) -> StatsOut:
    return StatsOut(**crud.stats(db))


@router.get("/{pred_id}", response_model=PredictionOut, summary="Detail satu prediksi")
def get_prediction(pred_id: int, db: Session = Depends(get_db)) -> PredictionOut:
    row = crud.get_prediction(db, pred_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Prediksi {pred_id} tidak ditemukan")
    return _to_out(row)


@router.delete(
    "/{pred_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Hapus satu prediksi beserta bbox-nya",
)
def delete_prediction(pred_id: int, db: Session = Depends(get_db)) -> Response:
    if not crud.delete_prediction(db, pred_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Prediksi {pred_id} tidak ditemukan")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
