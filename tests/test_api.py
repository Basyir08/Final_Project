"""Uji endpoint. Sengaja TIDAK memerlukan file bobot, supaya CI tetap cepat."""

from __future__ import annotations

from api.config import settings


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert {m["key"] for m in body["models"]} == {"yolo26", "frcnn"}


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/v1/predict/{model_key}" in paths
    assert "/api/v1/predictions" in paths


def test_models_info(client):
    r = client.get("/api/v1/metrics/models")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_training_history_yolo(client):
    r = client.get("/api/v1/metrics/training/yolo26")
    assert r.status_code == 200
    body = r.json()
    assert body["n_epochs"] > 0
    assert body["best"]["epoch"] >= 1
    assert 0 <= body["best"]["mAP50"] <= 1
    assert "mAP50_95" in body["columns"]


def test_training_history_frcnn(client):
    r = client.get("/api/v1/metrics/training/frcnn")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "Faster R-CNN"
    assert body["best"]["mAP50_95"] is not None


def test_training_history_unknown_model(client):
    assert client.get("/api/v1/metrics/training/resnet").status_code == 422


def test_comparison(client):
    r = client.get("/api/v1/metrics/comparison")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert {row["model_key"] for row in rows} == {"yolo26", "frcnn"}


def test_predict_rejects_non_image(client):
    r = client.post(
        "/api/v1/predict/yolo26",
        files={"file": ("catatan.txt", b"halo", "text/plain")},
        data={"conf": "0.25", "save": "false", "return_image": "false"},
    )
    # 422 = bukan gambar, 503 = bobot model belum tersedia (kondisi CI)
    assert r.status_code in (422, 503)


def test_predict_empty_file(client):
    r = client.post(
        "/api/v1/predict/yolo26",
        files={"file": ("kosong.png", b"", "image/png")},
        data={"save": "false"},
    )
    assert r.status_code == 400


def test_predict_contract(client, xray_bytes):
    """Bila bobot ada -> 200 dengan skema lengkap. Bila tidak -> 503 yang informatif."""
    r = client.post(
        "/api/v1/predict/yolo26",
        files={"file": ("xray.png", xray_bytes, "image/png")},
        data={"conf": "0.25", "save": "true", "return_image": "false"},
    )
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        assert "tidak ditemukan" in r.json()["detail"].lower()
        return

    body = r.json()
    assert body["model_key"] == "yolo26"
    assert body["image_width"] == 400 and body["image_height"] == 600
    assert body["prediction_id"] is not None
    for det in body["detections"]:
        assert 0 <= det["confidence"] <= 1
        assert len(det["box_original"]) == 4


def test_conf_out_of_range_rejected(client, xray_bytes):
    r = client.post(
        "/api/v1/predict/yolo26",
        files={"file": ("xray.png", xray_bytes, "image/png")},
        data={"conf": "1.5"},
    )
    assert r.status_code == 422


def test_upload_too_large(client):
    big = b"x" * ((settings.MAX_UPLOAD_MB + 1) * 1024 * 1024)
    r = client.post(
        "/api/v1/predict/yolo26",
        files={"file": ("besar.png", big, "image/png")},
        data={"save": "false"},
    )
    assert r.status_code == 413


def test_predictions_list_and_stats(client):
    r = client.get("/api/v1/predictions", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert {"total", "limit", "offset", "items"} <= body.keys()

    s = client.get("/api/v1/predictions/stats")
    assert s.status_code == 200
    assert "total_predictions" in s.json()


def test_prediction_not_found(client):
    assert client.get("/api/v1/predictions/999999").status_code == 404
    assert client.delete("/api/v1/predictions/999999").status_code == 404
