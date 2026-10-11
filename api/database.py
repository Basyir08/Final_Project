"""Koneksi database (SQLAlchemy 2.x).

Default SQLite agar langsung jalan tanpa setup. Untuk produksi cukup ganti
DATABASE_URL ke PostgreSQL, mis:
    DATABASE_URL=postgresql+psycopg://user:pass@host:5432/fracture
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Buat tabel bila belum ada (dipanggil saat startup aplikasi)."""
    from api import db_models  # noqa: F401  (registrasi model ke metadata)

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency FastAPI: satu sesi database per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
