"""Halaman 2 — grafik performa model dari riwayat training."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import api_client as api

MODELS = {"yolo26": "YOLO26", "frcnn": "Faster R-CNN"}
COLORS = {"YOLO26": "#e45756", "Faster R-CNN": "#4c78a8"}


@st.cache_data(ttl=300, show_spinner=False)
def _history(base_url: str, key: str) -> dict:
    return api.training_history(base_url, key)


@st.cache_data(ttl=300, show_spinner=False)
def _comparison(base_url: str) -> dict:
    return api.comparison(base_url)


def _line(df: pd.DataFrame, cols: list[str], title: str, ylab: str) -> None:
    cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not cols:
        st.info(f"Kolom untuk '{title}' tidak tersedia di CSV.")
        return
    fig = px.line(df, x="epoch", y=cols, title=title)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title="",
        yaxis_title=ylab,
        xaxis_title="Epoch",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _single_model(base_url: str, key: str) -> None:
    try:
        payload = _history(base_url, key)
    except api.ApiError as exc:
        st.error(str(exc))
        return

    df = pd.DataFrame(payload["records"])
    best = payload["best"]

    st.subheader(f"{payload['model_name']} — {payload['n_epochs']} epoch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best epoch", best["epoch"])
    c2.metric("mAP@0.5", f"{best['mAP50']:.4f}" if best["mAP50"] else "—")
    c3.metric("mAP@0.5:0.95", f"{best['mAP50_95']:.4f}" if best["mAP50_95"] else "—")
    if best.get("precision") is not None:
        c4.metric("Precision", f"{best['precision']:.4f}")
    elif best.get("val_loss") is not None:
        c4.metric("Val loss (best)", f"{best['val_loss']:.4f}")

    left, right = st.columns(2)
    with left:
        _line(df, ["train_loss", "val_loss"], "Kurva loss", "Loss")
    with right:
        _line(df, ["mAP50", "mAP50_95"], "Kurva mAP (validation)", "mAP")

    left, right = st.columns(2)
    with left:
        if key == "yolo26":
            _line(
                df,
                ["precision", "recall"],
                "Precision & Recall per epoch",
                "Skor",
            )
        else:
            _line(
                df,
                ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
                "Komponen loss",
                "Loss",
            )
    with right:
        _line(df, ["lr"], "Learning rate schedule", "LR")

    with st.expander("📄 Data mentah riwayat training"):
        st.dataframe(df, use_container_width=True, height=320)
        st.download_button(
            "Unduh CSV",
            df.to_csv(index=False).encode(),
            file_name=f"{key}_history.csv",
            mime="text/csv",
        )


def _overlay(base_url: str) -> None:
    frames = []
    for key, label in MODELS.items():
        try:
            payload = _history(base_url, key)
        except api.ApiError as exc:
            st.warning(f"{label}: {exc}")
            continue
        df = pd.DataFrame(payload["records"])
        df["model"] = label
        frames.append(df)

    if not frames:
        return
    allo = pd.concat(frames, ignore_index=True)

    for metric, title in [
        ("mAP50_95", "mAP@0.5:0.95 per epoch"),
        ("mAP50", "mAP@0.5 per epoch"),
        ("val_loss", "Validation loss per epoch"),
    ]:
        if metric not in allo.columns:
            continue
        fig = px.line(
            allo.dropna(subset=[metric]),
            x="epoch",
            y=metric,
            color="model",
            title=title,
            color_discrete_map=COLORS,
        )
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
            legend_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Catatan: jumlah epoch kedua model berbeda karena early stopping berhenti "
        "pada titik yang berbeda, jadi sumbu-X tidak sepanjang satu sama lain."
    )


def _summary(base_url: str) -> None:
    try:
        rows = _comparison(base_url)["rows"]
    except api.ApiError as exc:
        st.error(str(exc))
        return

    df = pd.DataFrame(rows)
    show = df[
        [
            "model_name",
            "n_epochs",
            "best_epoch",
            "mAP50",
            "mAP50_95",
            "avg_epoch_seconds",
        ]
    ].rename(
        columns={
            "model_name": "Model",
            "n_epochs": "Epoch dijalankan",
            "best_epoch": "Best epoch",
            "avg_epoch_seconds": "Rata-rata detik/epoch",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

    metrics = [m for m in ("mAP50", "mAP50_95") if df[m].notna().any()]
    melted = df.melt(
        id_vars="model_name", value_vars=metrics, var_name="metrik", value_name="nilai"
    ).dropna()
    fig = px.bar(
        melted,
        x="metrik",
        y="nilai",
        color="model_name",
        barmode="group",
        title="Perbandingan metrik pada epoch terbaik",
        text_auto=".3f",
        color_discrete_map=COLORS,
    )
    fig.update_layout(height=420, yaxis_range=[0, 1], legend_title="", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    if df["avg_epoch_seconds"].notna().any():
        fig2 = go.Figure(
            go.Bar(
                x=df["model_name"],
                y=df["avg_epoch_seconds"],
                marker_color=[COLORS.get(m, "#888") for m in df["model_name"]],
                text=[f"{v:.0f}s" for v in df["avg_epoch_seconds"]],
            )
        )
        fig2.update_layout(
            title="Rata-rata waktu training per epoch (detik)",
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis_title="detik",
        )
        st.plotly_chart(fig2, use_container_width=True)


def render(ctx: dict) -> None:
    st.title("📈 Performa Model")
    st.caption(
        "Grafik dibaca dari riwayat training (YOLO_history.csv & FRCNN_history.csv) "
        "lewat endpoint /api/v1/metrics."
    )

    tabs = st.tabs(["🏁 Ringkasan & perbandingan", "🔴 YOLO26", "🔵 Faster R-CNN", "🔀 Overlay"])
    with tabs[0]:
        _summary(ctx["base_url"])
    with tabs[1]:
        _single_model(ctx["base_url"], "yolo26")
    with tabs[2]:
        _single_model(ctx["base_url"], "frcnn")
    with tabs[3]:
        _overlay(ctx["base_url"])
