# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libglib2 & libgomp dibutuhkan opencv-headless dan torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .

# Torch versi CPU: image jauh lebih kecil (~800MB vs ~6GB untuk versi CUDA).
# Hapus baris --index-url bila server punya GPU NVIDIA.
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    grep -v -E '^(torch|torchvision)==' requirements-api.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt

COPY api ./api
COPY models/*.csv ./models/

RUN mkdir -p data/uploads models && \
    useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
