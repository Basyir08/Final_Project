"""Preprocessing wajib identik dengan pipeline training; diuji ketat."""

from __future__ import annotations

import numpy as np
import pytest

from api.config import settings
from api.preprocessing import (
    boxes_to_original,
    decode_image,
    draw_detections,
    enhance,
    full_preprocess,
    letterbox,
    to_gray,
)


def test_decode_invalid_bytes_returns_none():
    assert decode_image(b"bukan gambar sama sekali") is None


def test_decode_valid(xray_bytes):
    img = decode_image(xray_bytes)
    assert img is not None and img.shape == (600, 400, 3)


def test_to_gray_handles_bgra():
    bgra = np.zeros((10, 10, 4), np.uint8)
    assert to_gray(bgra).shape == (10, 10)


def test_enhance_returns_three_channels():
    img = np.random.randint(0, 255, (120, 90, 3), dtype=np.uint8)
    out = enhance(img)
    assert out.shape == (120, 90, 3)
    # CLAHE bekerja pada grayscale lalu disalin ke 3 channel -> semua channel sama
    assert np.array_equal(out[:, :, 0], out[:, :, 2])


@pytest.mark.parametrize("shape", [(600, 400), (400, 600), (832, 832), (97, 1301)])
def test_letterbox_is_square_and_preserves_aspect(shape):
    img = np.full((*shape, 3), 200, np.uint8)
    canvas, meta = letterbox(img)
    size = settings.IMG_SIZE
    assert canvas.shape == (size, size, 3)
    assert meta.orig_h, meta.orig_w == shape
    # sisi terpanjang harus pas memenuhi kanvas (padding hanya di satu sumbu)
    assert min(meta.pad_left, meta.pad_top) == 0


def test_boxes_map_back_to_original_coordinates():
    img = np.full((600, 400, 3), 128, np.uint8)
    _, meta = full_preprocess(img)

    # kotak yang menutupi seluruh area gambar pada ruang letterbox
    size = settings.IMG_SIZE
    box = np.array([[meta.pad_left, meta.pad_top, size - meta.pad_left, size - meta.pad_top]])
    back = boxes_to_original(box, meta)[0]

    assert back[0] == pytest.approx(0, abs=1.5)
    assert back[1] == pytest.approx(0, abs=1.5)
    assert back[2] == pytest.approx(400, abs=1.5)
    assert back[3] == pytest.approx(600, abs=1.5)


def test_boxes_to_original_empty():
    from api.preprocessing import LetterboxMeta

    meta = LetterboxMeta(ratio=1.0, pad_left=0, pad_top=0, orig_w=10, orig_h=10)
    assert boxes_to_original(np.zeros((0, 4)), meta).shape == (0, 4)


def test_draw_detections_does_not_mutate_input():
    img = np.zeros((300, 300, 3), np.uint8)
    dets = [{"class_name": "Linear", "confidence": 0.91, "box": [10, 10, 100, 100]}]
    out = draw_detections(img, dets)
    assert out.shape == img.shape
    assert img.sum() == 0  # gambar asli tidak berubah
    assert out.sum() > 0  # anotasi tergambar di salinan
