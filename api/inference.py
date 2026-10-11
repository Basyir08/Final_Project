"""Pemuatan bobot & inferensi untuk kedua model.

Desain penting:
* Model dimuat **lazy** (saat request pertama) dan di-cache sebagai singleton,
  supaya proses CI/test tidak memerlukan file bobot sama sekali.
* Kedua detektor mengembalikan struktur yang sama persis, sehingga endpoint
  API dan UI tidak perlu tahu model mana yang dipakai.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from api.config import settings
from api.preprocessing import LetterboxMeta, boxes_to_original, full_preprocess

MODEL_YOLO = "yolo26"
MODEL_FRCNN = "frcnn"
MODEL_KEYS = (MODEL_YOLO, MODEL_FRCNN)

DISPLAY_NAME = {MODEL_YOLO: "YOLO26", MODEL_FRCNN: "Faster R-CNN"}


class ModelNotAvailable(RuntimeError):
    """Bobot model tidak ditemukan atau gagal dimuat."""


def resolve_device() -> str:
    import torch

    if settings.DEVICE != "auto":
        return settings.DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Detektor
# --------------------------------------------------------------------------
class BaseDetector:
    key: str = ""
    weights: Path

    def __init__(self, weights: Path) -> None:
        self.weights = Path(weights)
        self._model: Any = None
        self._lock = threading.Lock()
        self.class_names: list[str] = list(settings.FALLBACK_CLASSES)
        self.meta: dict[str, Any] = {}

    # --- siklus hidup -----------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self.weights.exists()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # pengecekan ganda (thread-safe)
                return
            if not self.weights.exists():
                raise ModelNotAvailable(
                    f"Bobot {DISPLAY_NAME[self.key]} tidak ditemukan di {self.weights}. "
                    "Salin file bobot ke folder models/ (lihat models/README.md)."
                )
            t0 = time.perf_counter()
            self._load()
            self.meta["load_seconds"] = round(time.perf_counter() - t0, 2)

    def _load(self) -> None:  # pragma: no cover - diimplementasikan subclass
        raise NotImplementedError

    def class_name(self, idx: int) -> str:
        idx = int(idx)
        return self.class_names[idx] if 0 <= idx < len(self.class_names) else f"class_{idx}"

    # --- inferensi --------------------------------------------------------
    def _raw_predict(self, img_prep: np.ndarray, conf: float):  # pragma: no cover
        raise NotImplementedError

    def predict(self, img_bgr: np.ndarray, conf: float | None = None) -> dict[str, Any]:
        """Jalankan preprocessing + inferensi pada satu gambar BGR."""
        conf = settings.DEFAULT_CONF if conf is None else float(conf)
        self.ensure_loaded()

        img_prep, meta = full_preprocess(img_bgr)
        t0 = time.perf_counter()
        boxes, labels, scores = self._raw_predict(img_prep, conf)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        orig_boxes = boxes_to_original(boxes, meta)
        detections = [
            {
                "class_id": int(lbl),
                "class_name": self.class_name(lbl),
                "confidence": round(float(sc), 4),
                # koordinat pada kanvas 832x832 (ruang kerja model)
                "box": [round(float(v), 2) for v in box],
                # koordinat pada gambar asli yang diupload user
                "box_original": [round(float(v), 2) for v in obox],
            }
            for box, obox, lbl, sc in zip(boxes, orig_boxes, labels, scores)
        ]
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return {
            "model": self.key,
            "model_name": DISPLAY_NAME[self.key],
            "conf_threshold": conf,
            "inference_ms": round(elapsed_ms, 2),
            "detections": detections,
            "image_prep": img_prep,
            "letterbox": meta,
        }


class YoloDetector(BaseDetector):
    key = MODEL_YOLO

    def _load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))
        names = self._model.names
        if isinstance(names, dict):
            self.class_names = [names[i] for i in sorted(names)]
        elif names:
            self.class_names = list(names)
        self.meta.update(
            {"n_classes": len(self.class_names), "device": resolve_device(), "task": "detect"}
        )

    def _raw_predict(self, img_prep: np.ndarray, conf: float):
        res = self._model.predict(
            source=img_prep,  # ultralytics menerima array BGR langsung
            imgsz=settings.IMG_SIZE,
            conf=conf,
            device=resolve_device(),
            verbose=False,
        )[0]
        return (
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
            res.boxes.conf.cpu().numpy(),
        )


class FrcnnDetector(BaseDetector):
    key = MODEL_FRCNN

    def _load(self) -> None:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        try:
            ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 1.13
            ckpt = torch.load(self.weights, map_location="cpu")

        if isinstance(ckpt, dict):
            if isinstance(ckpt.get("classes"), list | tuple) and ckpt["classes"]:
                self.class_names = list(ckpt["classes"])
            for k in ("epoch", "map50_95", "img_size"):
                if k in ckpt:
                    self.meta[k] = ckpt[k]

        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        if hasattr(state, "state_dict"):  # checkpoint menyimpan objek model utuh
            model = state
        else:
            n_out = state["roi_heads.box_predictor.cls_score.weight"].shape[0]
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,  # bobot pretrained langsung ditimpa
                weights_backbone=None,
                min_size=settings.IMG_SIZE,
                max_size=settings.IMG_SIZE,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                box_detections_per_img=100,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_out)
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.meta["missing_keys"] = len(missing)
            self.meta["unexpected_keys"] = len(unexpected)

        device = resolve_device()
        self._model = model.to(device).eval()
        self._torch = torch
        self.meta.update({"n_classes": len(self.class_names), "device": device})

    def _raw_predict(self, img_prep: np.ndarray, conf: float):
        import cv2

        torch = self._torch
        rgb = cv2.cvtColor(img_prep, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(resolve_device())
        with torch.no_grad():
            out = self._model([tensor])[0]

        scores = out["scores"].cpu().numpy()
        keep = scores >= conf
        return (
            out["boxes"].cpu().numpy()[keep],
            out["labels"].cpu().numpy()[keep] - 1,  # -1: balik ke indeks 0-based
            scores[keep],
        )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
_registry: dict[str, BaseDetector] = {
    MODEL_YOLO: YoloDetector(settings.YOLO_WEIGHTS),
    MODEL_FRCNN: FrcnnDetector(settings.FRCNN_WEIGHTS),
}


def get_detector(key: str) -> BaseDetector:
    try:
        return _registry[key]
    except KeyError:
        raise ModelNotAvailable(
            f"Model '{key}' tidak dikenal. Pilihan: {', '.join(MODEL_KEYS)}"
        ) from None


def registry_status() -> list[dict[str, Any]]:
    return [
        {
            "key": d.key,
            "name": DISPLAY_NAME[d.key],
            "weights_path": str(d.weights),
            "available": d.is_available,
            "loaded": d.is_loaded,
            "classes": d.class_names if d.is_loaded else None,
            "meta": d.meta,
        }
        for d in _registry.values()
    ]


def preload_all() -> None:
    """Dipanggil saat startup bila PRELOAD_MODELS=true."""
    for d in _registry.values():
        if d.is_available:
            try:
                d.ensure_loaded()
            except Exception as exc:  # noqa: BLE001 - startup tidak boleh gagal total
                print(f"[WARN] gagal memuat {DISPLAY_NAME[d.key]}: {exc}")


__all__ = [
    "MODEL_KEYS",
    "MODEL_YOLO",
    "MODEL_FRCNN",
    "DISPLAY_NAME",
    "LetterboxMeta",
    "ModelNotAvailable",
    "get_detector",
    "registry_status",
    "preload_all",
]
