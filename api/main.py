"""Entry point layanan REST API.

Jalankan lokal:
    uvicorn api.main:app --reload --port 8000

Dokumentasi interaktif otomatis tersedia di:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.database import engine, init_db
from api.inference import preload_all, registry_status
from api.routers import model_metrics, predict, predictions
from api.schemas import HealthOut, ModelInfo


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.PRELOAD_MODELS:
        preload_all()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "REST API deteksi fraktur tulang pada citra X-ray menggunakan dua model "
        "deteksi objek: **YOLO26** (single-stage) dan **Faster R-CNN** (two-stage). "
        "Setiap hasil prediksi disimpan ke database."
    ),
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# gambar beranotasi bisa diakses langsung lewat /static/<nama file>
app.mount("/static", StaticFiles(directory=settings.STORAGE_DIR), name="static")

app.include_router(predict.router, prefix=settings.API_PREFIX)
app.include_router(predictions.router, prefix=settings.API_PREFIX)
app.include_router(model_metrics.router, prefix=settings.API_PREFIX)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthOut, tags=["system"])
def health() -> HealthOut:
    """Dipakai oleh Docker healthcheck, CI, dan halaman status Streamlit."""
    try:
        from api.inference import resolve_device

        device = resolve_device()
    except Exception:  # noqa: BLE001 - torch bisa saja belum terpasang
        device = "unknown"

    try:
        with engine.connect():
            db_status = "connected"
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc.__class__.__name__}"

    return HealthOut(
        status="ok",
        version=settings.APP_VERSION,
        device=device,
        database=db_status,
        models=[ModelInfo(**m) for m in registry_status()],
    )
