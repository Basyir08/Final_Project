"""Fixture pytest.

Variabel environment di-set SEBELUM paket api diimpor supaya database dan
folder penyimpanan mengarah ke direktori sementara, bukan data asli.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

TMP = Path(tempfile.mkdtemp(prefix="fracture-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{TMP / 'test.db'}"
os.environ["STORAGE_DIR"] = str(TMP / "uploads")
os.environ["PRELOAD_MODELS"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from api.database import Base, engine  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db():
    from api.database import SessionLocal

    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def xray_bytes() -> bytes:
    """Gambar X-ray palsu: gradien + garis terang menyerupai tulang."""
    import cv2

    img = np.zeros((600, 400, 3), np.uint8)
    cv2.rectangle(img, (150, 50), (250, 550), (120, 120, 120), -1)
    cv2.line(img, (150, 300), (250, 320), (240, 240, 240), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()
