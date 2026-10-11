"""Pembungkus tipis untuk memanggil REST API dari Streamlit."""

from __future__ import annotations

import base64
import os
from typing import Any

import requests

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PREFIX = "/api/v1"
TIMEOUT = 180  # inferensi Faster R-CNN di CPU bisa lama


class ApiError(RuntimeError):
    """Error yang sudah berbentuk pesan ramah untuk ditampilkan ke pengguna."""


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def _handle(resp: requests.Response) -> Any:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise ApiError(f"[{resp.status_code}] {detail}")
    if resp.status_code == 204:
        return None
    return resp.json()


def _get(base: str, path: str, **params) -> Any:
    try:
        return _handle(requests.get(_url(base, path), params=params, timeout=TIMEOUT))
    except requests.RequestException as exc:
        raise ApiError(
            f"Tidak bisa terhubung ke API di {base} ({exc.__class__.__name__})."
        ) from exc


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------
def health(base: str) -> dict:
    return _get(base, "/health")


def models_info(base: str) -> list[dict]:
    return _get(base, f"{PREFIX}/metrics/models")


def predict(
    base: str,
    *,
    model_key: str,
    filename: str,
    content: bytes,
    conf: float = 0.25,
    save: bool = True,
    return_image: bool = True,
) -> dict:
    files = {"file": (filename, content, "application/octet-stream")}
    data = {"conf": str(conf), "save": str(save).lower(), "return_image": str(return_image).lower()}
    try:
        resp = requests.post(
            _url(base, f"{PREFIX}/predict/{model_key}"), files=files, data=data, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Tidak bisa terhubung ke API di {base} ({exc.__class__.__name__})."
        ) from exc
    return _handle(resp)


def predict_both(
    base: str, *, filename: str, content: bytes, conf: float = 0.25, save: bool = True
) -> dict:
    files = {"file": (filename, content, "application/octet-stream")}
    data = {"conf": str(conf), "save": str(save).lower(), "return_image": "true"}
    try:
        resp = requests.post(
            _url(base, f"{PREFIX}/predict/compare/both"), files=files, data=data, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Tidak bisa terhubung ke API di {base} ({exc.__class__.__name__})."
        ) from exc
    return _handle(resp)


def training_history(base: str, model_key: str) -> dict:
    return _get(base, f"{PREFIX}/metrics/training/{model_key}")


def comparison(base: str) -> dict:
    return _get(base, f"{PREFIX}/metrics/comparison")


def list_predictions(base: str, **params) -> dict:
    return _get(base, f"{PREFIX}/predictions", **params)


def prediction_stats(base: str) -> dict:
    return _get(base, f"{PREFIX}/predictions/stats")


def delete_prediction(base: str, pred_id: int) -> None:
    try:
        _handle(requests.delete(_url(base, f"{PREFIX}/predictions/{pred_id}"), timeout=TIMEOUT))
    except requests.RequestException as exc:
        raise ApiError(f"Gagal menghapus data ({exc.__class__.__name__}).") from exc


def decode_b64_image(b64: str | None) -> bytes | None:
    return base64.b64decode(b64) if b64 else None
