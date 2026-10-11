"""Konfigurasi terpusat untuk layanan API.

Semua nilai bisa ditimpa lewat environment variable atau file .env
(lihat .env.example).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- aplikasi ----------
    APP_NAME: str = "Bone Fracture Detection API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    CORS_ORIGINS: str = "*"  # dipisah koma, mis. "http://localhost:8501"

    # ---------- bobot model ----------
    YOLO_WEIGHTS: Path = BASE_DIR / "models" / "YOLO_best.pt"
    FRCNN_WEIGHTS: Path = BASE_DIR / "models" / "best_frcnn.pt"
    YOLO_HISTORY: Path = BASE_DIR / "models" / "YOLO_history.csv"
    FRCNN_HISTORY: Path = BASE_DIR / "models" / "FRCNN_history.csv"

    # Muat model saat startup. False = lazy load pada request pertama
    # (dipakai di CI/test supaya tidak butuh file bobot).
    PRELOAD_MODELS: bool = False
    DEVICE: str = "auto"  # auto | cpu | cuda

    # ---------- preprocessing (HARUS sama dengan saat training) ----------
    IMG_SIZE: int = 832
    CLAHE_CLIP: float = 2.0
    CLAHE_TILE: int = 8
    APPLY_LIGHT_DENOISE: bool = False

    # ---------- inferensi ----------
    DEFAULT_CONF: float = 0.25
    MAX_UPLOAD_MB: int = 20

    # ---------- penyimpanan & database ----------
    STORAGE_DIR: Path = BASE_DIR / "data" / "uploads"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'predictions.db'}"

    # daftar kelas cadangan bila checkpoint tidak menyimpan metadata "classes"
    FALLBACK_CLASSES: list[str] = [
        "Comminuted",
        "Greenstick",
        "Healthy",
        "Linear",
        "Oblique Displaced",
        "Oblique",
        "Segmental",
        "Spiral",
        "Transverse Displaced",
        "Transverse",
    ]

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
