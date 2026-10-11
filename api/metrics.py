"""Pembaca riwayat training (*_history.csv) untuk grafik performa model.

Kedua CSV punya nama kolom yang berbeda:

YOLO26 (results.csv ultralytics)
    epoch, train/box_loss, train/cls_loss, train/l1_loss,
    metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
    val/box_loss, val/cls_loss, val/l1_loss, lr/pg0 ...

Faster R-CNN (ditulis manual oleh training loop)
    epoch, lr, train_loss, loss_classifier, loss_box_reg, loss_objectness,
    loss_rpn_box_reg, val_loss, val_mAP50, val_mAP50_95, time_s

Modul ini menyatukan keduanya ke nama kolom standar supaya UI bisa
memplot grafik yang sama untuk kedua model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from api.config import settings
from api.inference import DISPLAY_NAME, MODEL_FRCNN, MODEL_YOLO

HISTORY_PATH: dict[str, Path] = {
    MODEL_YOLO: settings.YOLO_HISTORY,
    MODEL_FRCNN: settings.FRCNN_HISTORY,
}

# nama kolom standar -> kandidat nama kolom di masing-masing CSV
STANDARD_MAP: dict[str, tuple[str, ...]] = {
    "train_loss": ("train_loss",),
    "val_loss": ("val_loss",),
    "mAP50": ("metrics/mAP50(B)", "val_mAP50"),
    "mAP50_95": ("metrics/mAP50-95(B)", "val_mAP50_95"),
    "precision": ("metrics/precision(B)",),
    "recall": ("metrics/recall(B)",),
    "lr": ("lr", "lr/pg0"),
}

# komponen loss yang menarik untuk ditampilkan terpisah
LOSS_COLUMNS: dict[str, tuple[str, ...]] = {
    MODEL_YOLO: (
        "train/box_loss",
        "train/cls_loss",
        "train/l1_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/l1_loss",
    ),
    MODEL_FRCNN: ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
}


class HistoryNotAvailable(FileNotFoundError):
    pass


def _standardize(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for std, candidates in STANDARD_MAP.items():
        for c in candidates:
            if c in df.columns:
                df[std] = pd.to_numeric(df[c], errors="coerce")
                break

    # YOLO tidak punya kolom total loss -> dijumlahkan dari komponennya
    if "train_loss" not in df.columns:
        parts = [c for c in df.columns if c.startswith("train/") and c.endswith("loss")]
        if parts:
            df["train_loss"] = df[parts].sum(axis=1)
    if "val_loss" not in df.columns:
        parts = [c for c in df.columns if c.startswith("val/") and c.endswith("loss")]
        if parts:
            df["val_loss"] = df[parts].sum(axis=1)

    return df


@lru_cache(maxsize=4)
def load_history(model_key: str) -> pd.DataFrame:
    path = HISTORY_PATH.get(model_key)
    if path is None:
        raise HistoryNotAvailable(f"Model '{model_key}' tidak dikenal")
    if not Path(path).exists():
        raise HistoryNotAvailable(
            f"File riwayat training tidak ditemukan: {path}. "
            "Salin *_history.csv ke folder models/."
        )
    return _standardize(pd.read_csv(path), model_key)


def best_epoch(model_key: str) -> dict[str, Any]:
    """Epoch dengan mAP50-95 tertinggi (kriteria yang dipakai saat training)."""
    df = load_history(model_key)
    metric = "mAP50_95" if "mAP50_95" in df.columns else "mAP50"
    idx = df[metric].idxmax()
    row = df.loc[idx]

    def val(col: str) -> float | None:
        return round(float(row[col]), 6) if col in df.columns and pd.notna(row[col]) else None

    return {
        "epoch": int(row["epoch"]),
        "mAP50": val("mAP50"),
        "mAP50_95": val("mAP50_95"),
        "precision": val("precision"),
        "recall": val("recall"),
        "val_loss": val("val_loss"),
    }


def history_payload(model_key: str) -> dict[str, Any]:
    df = load_history(model_key)
    clean = df.where(pd.notna(df), None)
    return {
        "model_key": model_key,
        "model_name": DISPLAY_NAME[model_key],
        "n_epochs": int(len(df)),
        "best": best_epoch(model_key),
        "columns": list(df.columns),
        "records": clean.to_dict(orient="records"),
    }


def comparison_table() -> list[dict[str, Any]]:
    """Ringkasan untuk bar chart perbandingan kedua model."""
    rows = []
    for key in (MODEL_YOLO, MODEL_FRCNN):
        try:
            df = load_history(key)
            best = best_epoch(key)
        except HistoryNotAvailable:
            continue
        epoch_time = None
        if "time_s" in df.columns:
            epoch_time = round(float(df["time_s"].mean()), 2)
        elif "time" in df.columns:  # YOLO menulis waktu kumulatif
            epoch_time = round(float(df["time"].diff().dropna().mean()), 2)
        rows.append(
            {
                "model_key": key,
                "model_name": DISPLAY_NAME[key],
                "n_epochs": int(len(df)),
                "best_epoch": best["epoch"],
                "mAP50": best["mAP50"],
                "mAP50_95": best["mAP50_95"],
                "precision": best["precision"],
                "recall": best["recall"],
                "val_loss": best["val_loss"],
                "avg_epoch_seconds": epoch_time,
            }
        )
    return rows


def available_models() -> list[str]:
    return [k for k, p in HISTORY_PATH.items() if Path(p).exists()]
