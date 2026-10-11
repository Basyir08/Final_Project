"""Halaman 3 — riwayat prediksi yang tersimpan di database."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import api_client as api

MODEL_FILTER = {"Semua model": None, "YOLO26": "yolo26", "Faster R-CNN": "frcnn"}


def _stats_block(base_url: str) -> None:
    try:
        stats = api.prediction_stats(base_url)
    except api.ApiError as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total prediksi", stats["total_predictions"])
    c2.metric("Total bbox tersimpan", stats["total_detections"])
    by_model = stats["by_model"]
    c3.metric("YOLO26", by_model.get("yolo26", 0))
    c4.metric("Faster R-CNN", by_model.get("frcnn", 0))

    avg = stats["avg_inference_ms"]
    if avg:
        dfa = pd.DataFrame([{"model": k, "ms": v} for k, v in avg.items()]).replace(
            {"yolo26": "YOLO26", "frcnn": "Faster R-CNN"}
        )
        fig = px.bar(
            dfa,
            x="model",
            y="ms",
            text_auto=".0f",
            title="Rata-rata waktu inferensi nyata (ms/gambar)",
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    if stats["top_classes"]:
        dfc = pd.DataFrame(stats["top_classes"])
        fig = px.bar(
            dfc,
            x="class_name",
            y="count",
            color="avg_confidence",
            title="Distribusi kelas pada seluruh prediksi",
            labels={"class_name": "", "count": "jumlah", "avg_confidence": "rata² conf"},
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)


def render(ctx: dict) -> None:
    st.title("🗂️ Riwayat Prediksi")
    st.caption("Seluruh hasil prediksi yang disimpan ke database lewat REST API.")

    _stats_block(ctx["base_url"])
    st.divider()

    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Filter model", list(MODEL_FILTER))
    limit = c2.number_input("Jumlah baris", 10, 500, 50, step=10)

    params = {"limit": int(limit)}
    if MODEL_FILTER[choice]:
        params["model_key"] = MODEL_FILTER[choice]

    try:
        data = api.list_predictions(ctx["base_url"], **params)
    except api.ApiError as exc:
        st.error(str(exc))
        return

    items = data["items"]
    if not items:
        st.info("Belum ada prediksi tersimpan. Jalankan deteksi di halaman pertama.")
        return

    st.caption(f"Menampilkan {len(items)} dari {data['total']} data.")
    df = pd.DataFrame(
        [
            {
                "ID": it["id"],
                "waktu": it["created_at"],
                "file": it["filename"],
                "model": it["model_name"],
                "deteksi": it["n_detections"],
                "kelas dominan": it["top_class"],
                "confidence": it["top_confidence"],
                "ms": round(it["inference_ms"], 1),
            }
            for it in items
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True, height=360)
    st.download_button(
        "⬇️ Unduh riwayat (CSV)",
        df.to_csv(index=False).encode(),
        file_name="riwayat_prediksi.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Detail prediksi")
    ids = [it["id"] for it in items]
    pred_id = st.selectbox("Pilih ID", ids)
    row = next(it for it in items if it["id"] == pred_id)

    left, right = st.columns([2, 1])
    with left:
        if row.get("annotated_url"):
            st.image(
                ctx["base_url"].rstrip("/") + row["annotated_url"],
                caption=f"#{row['id']} — {row['model_name']}",
                use_container_width=True,
            )
        else:
            st.info("Gambar beranotasi tidak tersimpan untuk prediksi ini.")
    with right:
        st.json(
            {
                "model": row["model_name"],
                "threshold": row["conf_threshold"],
                "ukuran": f"{row['image_width']}x{row['image_height']}",
                "deteksi": row["n_detections"],
                "inferensi_ms": row["inference_ms"],
            }
        )
        if st.button("🗑️ Hapus prediksi ini", use_container_width=True):
            try:
                api.delete_prediction(ctx["base_url"], pred_id)
                st.success(f"Prediksi #{pred_id} dihapus.")
                st.rerun()
            except api.ApiError as exc:
                st.error(str(exc))

    if row["detections"]:
        st.dataframe(pd.DataFrame(row["detections"]), use_container_width=True, hide_index=True)
