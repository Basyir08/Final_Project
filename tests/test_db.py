"""Uji lapisan database tanpa menyentuh model."""

from __future__ import annotations

from api import crud

SAMPLE = [
    {
        "class_id": 3,
        "class_name": "Linear",
        "confidence": 0.91,
        "box": [10, 20, 110, 220],
        "box_original": [5, 10, 55, 110],
    },
    {
        "class_id": 9,
        "class_name": "Transverse",
        "confidence": 0.44,
        "box": [30, 40, 90, 140],
        "box_original": [15, 20, 45, 70],
    },
]


def _create(db, model_key="yolo26", detections=None):
    return crud.create_prediction(
        db,
        filename="xray.png",
        model_key=model_key,
        model_name="YOLO26" if model_key == "yolo26" else "Faster R-CNN",
        conf_threshold=0.25,
        image_width=400,
        image_height=600,
        inference_ms=42.0,
        detections=SAMPLE if detections is None else detections,
    )


def test_create_stores_detections_and_top_class(db):
    row = _create(db)
    assert row.id is not None
    assert row.n_detections == 2
    assert row.top_class == "Linear"  # deteksi pertama = confidence tertinggi
    assert row.top_confidence == 0.91
    # bbox disimpan dalam koordinat gambar asli
    assert row.detections[0].x1 == 5


def test_prediction_without_detection(db):
    row = _create(db, detections=[])
    assert row.n_detections == 0
    assert row.top_class is None


def test_list_filter_by_model(db):
    _create(db, model_key="yolo26")
    _create(db, model_key="frcnn")
    total_all, _ = crud.list_predictions(db)
    total_frcnn, rows = crud.list_predictions(db, model_key="frcnn")
    assert total_all >= total_frcnn >= 1
    assert all(r.model_key == "frcnn" for r in rows)


def test_stats_aggregation(db):
    _create(db)
    s = crud.stats(db)
    assert s["total_predictions"] >= 1
    assert s["total_detections"] >= 2
    assert "yolo26" in s["by_model"]
    assert any(c["class_name"] == "Linear" for c in s["top_classes"])


def test_delete_cascades(db):
    row = _create(db)
    pred_id = row.id
    assert crud.delete_prediction(db, pred_id) is True
    assert crud.get_prediction(db, pred_id) is None
    assert crud.delete_prediction(db, pred_id) is False
