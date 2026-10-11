"""Uji alur endpoint prediksi memakai detektor tiruan.

Tujuannya menguji jalur kode `_run` (preprocessing -> pemetaan bbox ->
anotasi -> simpan DB -> response) tanpa perlu torch, ultralytics, maupun
file bobot — sehingga tetap bisa dijalankan di CI.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from api import inference
from api.preprocessing import full_preprocess


class StubDetector:
    """Selalu mengembalikan satu bbox di tengah kanvas letterbox."""

    key = "yolo26"
    class_names = ["Linear"]

    def predict(self, img_bgr, conf=None):
        img_prep, meta = full_preprocess(img_bgr)
        size = img_prep.shape[0]
        c = size / 2
        box = [c - 60, c - 60, c + 60, c + 60]
        from api.preprocessing import boxes_to_original

        obox = boxes_to_original(np.array([box]), meta)[0]
        return {
            "model": self.key,
            "model_name": "YOLO26",
            "conf_threshold": 0.25 if conf is None else conf,
            "inference_ms": 12.3,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "Linear",
                    "confidence": 0.87,
                    "box": [round(v, 2) for v in box],
                    "box_original": [round(float(v), 2) for v in obox],
                }
            ],
            "image_prep": img_prep,
            "letterbox": meta,
        }


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(
        "api.routers.predict.get_detector", lambda key: StubDetector(), raising=True
    )


def test_predict_full_flow(client, xray_bytes, stub):
    r = client.post(
        "/api/v1/predict/yolo26",
        files={"file": ("xray.png", xray_bytes, "image/png")},
        data={"conf": "0.3", "save": "true", "return_image": "true"},
    )
    assert r.status_code == 200
    body = r.json()

    assert body["n_detections"] == 1
    assert body["top_class"] == "Linear"
    assert body["conf_threshold"] == 0.3
    assert body["image_width"] == 400 and body["image_height"] == 600

    # bbox hasil pemetaan balik harus berada di dalam gambar asli
    x1, y1, x2, y2 = body["detections"][0]["box_original"]
    assert 0 <= x1 < x2 <= 400
    assert 0 <= y1 < y2 <= 600

    # gambar beranotasi dikembalikan sebagai PNG base64 yang valid
    png = base64.b64decode(body["annotated_image_b64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    # tersimpan ke database dan bisa dibaca kembali
    pred_id = body["prediction_id"]
    assert pred_id is not None
    detail = client.get(f"/api/v1/predictions/{pred_id}").json()
    assert detail["model_name"] == "YOLO26"
    assert detail["detections"][0]["class_name"] == "Linear"
    assert detail["annotated_url"].startswith("/static/")


def test_predict_without_saving(client, xray_bytes, stub):
    r = client.post(
        "/api/v1/predict/yolo26",
        files={"file": ("xray.png", xray_bytes, "image/png")},
        data={"save": "false", "return_image": "false"},
    )
    body = r.json()
    assert body["prediction_id"] is None
    assert body["annotated_image_b64"] is None
    assert body["annotated_url"] is None


def test_compare_runs_both_models(client, xray_bytes, monkeypatch):
    monkeypatch.setattr(
        "api.routers.predict.get_detector", lambda key: StubDetector(), raising=True
    )
    r = client.post(
        "/api/v1/predict/compare/both",
        files={"file": ("xray.png", xray_bytes, "image/png")},
        data={"conf": "0.25", "save": "false", "return_image": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["yolo26"]["n_detections"] == 1
    assert body["frcnn"]["n_detections"] == 1


def test_unavailable_model_returns_503(client, xray_bytes, monkeypatch):
    def boom(key):
        raise inference.ModelNotAvailable("Bobot Faster R-CNN tidak ditemukan di models/")

    monkeypatch.setattr("api.routers.predict.get_detector", boom, raising=True)
    r = client.post(
        "/api/v1/predict/frcnn",
        files={"file": ("xray.png", xray_bytes, "image/png")},
        data={"save": "false"},
    )
    assert r.status_code == 503
    assert "tidak ditemukan" in r.json()["detail"]
