"""Preprocessing X-ray — HARUS identik dengan pipeline saat training.

grayscale -> (opsional denoise ringan) -> CLAHE -> 3 channel -> letterbox IMG_SIZE.

Kalau langkah di sini berbeda sedikit saja dengan notebook training,
akurasi model akan turun drastis tanpa error apa pun. Jadi modul ini
sengaja dipisah dan diuji di tests/test_preprocessing.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from api.config import settings


@dataclass(frozen=True)
class LetterboxMeta:
    """Informasi untuk memetakan koordinat kembali ke gambar asli."""

    ratio: float
    pad_left: int
    pad_top: int
    orig_w: int
    orig_h: int


def decode_image(raw: bytes) -> np.ndarray | None:
    """Ubah bytes hasil upload menjadi array BGR. None bila bukan gambar."""
    buf = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance(img: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
    """Grayscale + CLAHE, dikembalikan sebagai 3 channel BGR."""
    gray = to_gray(img)
    if settings.APPLY_LIGHT_DENOISE:
        gray = cv2.medianBlur(gray, 3)
    if apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=settings.CLAHE_CLIP,
            tileGridSize=(settings.CLAHE_TILE, settings.CLAHE_TILE),
        )
        gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def letterbox(
    img: np.ndarray, new_size: int | None = None, color: tuple[int, int, int] = (0, 0, 0)
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize menjaga rasio aspek + padding simetris ke kanvas persegi."""
    new_size = new_size or settings.IMG_SIZE
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top, left = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized

    return canvas, LetterboxMeta(ratio=r, pad_left=left, pad_top=top, orig_w=w, orig_h=h)


def full_preprocess(img: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
    """Pipeline lengkap yang dipakai model saat inferensi."""
    return letterbox(enhance(img, apply_clahe=True))


def boxes_to_original(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Petakan bbox dari ruang letterbox (IMG_SIZE) kembali ke gambar asli."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_left) / meta.ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_top) / meta.ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, meta.orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, meta.orig_h)
    return out


# --------------------------------------------------------------------------
# Visualisasi
# --------------------------------------------------------------------------
# Warna per model supaya mudah dibedakan di UI (format BGR untuk OpenCV).
MODEL_COLORS: dict[str, tuple[int, int, int]] = {
    "yolo26": (0, 0, 255),  # merah saat ditampilkan sebagai RGB
    "frcnn": (255, 0, 0),  # biru saat ditampilkan sebagai RGB
}


def draw_detections(
    img_bgr: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (0, 0, 255),
    box_key: str = "box",
) -> np.ndarray:
    """Gambar bbox + label. Mengembalikan array BGR (siap di-encode)."""
    out = img_bgr.copy()
    h = out.shape[0]
    thick = max(2, int(round(h / 400)))
    scale = max(0.5, h / 1400)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det[box_key])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ytext = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ytext - th - 4), (x1 + tw + 4, ytext + 4), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, ytext),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )
    return out


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Gagal meng-encode gambar ke PNG")
    return buf.tobytes()
