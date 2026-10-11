"""Antarmuka Streamlit untuk deteksi fraktur tulang.

Jalankan:
    streamlit run ui/app.py

Pastikan REST API sudah hidup (default http://localhost:8000).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui import api_client as api  # noqa: E402
from ui.views import about, detect, history, performance  # noqa: E402

st.set_page_config(
    page_title="Deteksi Fraktur Tulang",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_LABELS = {
    "yolo26": "YOLO26 (single-stage)",
    "frcnn": "Faster R-CNN (two-stage)",
    "both": "Bandingkan keduanya",
}

PAGES = {
    "🔍 Deteksi X-ray": detect.render,
    "📈 Performa Model": performance.render,
    "🗂️ Riwayat Prediksi": history.render,
    "ℹ️ Tentang": about.render,
}


def sidebar() -> dict:
    st.sidebar.title("🦴 Fracture Detection")
    st.sidebar.caption("YOLO26 vs Faster R-CNN")

    page = st.sidebar.radio("Navigasi", list(PAGES), label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    model_key = st.sidebar.radio(
        "Pilih model",
        list(MODEL_LABELS),
        format_func=lambda k: MODEL_LABELS[k],
        key="model_key",
    )
    conf = st.sidebar.slider(
        "Confidence threshold",
        0.05,
        0.95,
        0.25,
        0.05,
        help="Deteksi dengan confidence di bawah nilai ini dibuang.",
    )
    save = st.sidebar.toggle("Simpan hasil ke database", value=True)

    st.sidebar.divider()
    with st.sidebar.expander("⚙️ Koneksi API", expanded=False):
        base_url = st.text_input("Base URL", value=api.DEFAULT_BASE_URL)
        if st.button("Tes koneksi", use_container_width=True):
            try:
                st.session_state["_health"] = api.health(base_url)
            except api.ApiError as exc:
                st.session_state["_health"] = {"error": str(exc)}

    _status_badge(base_url)
    return {"page": page, "model_key": model_key, "conf": conf, "save": save, "base_url": base_url}


def _status_badge(base_url: str) -> None:
    health = st.session_state.get("_health")
    if health is None:
        try:
            health = api.health(base_url)
            st.session_state["_health"] = health
        except api.ApiError as exc:
            health = {"error": str(exc)}

    if "error" in health:
        st.sidebar.error("API tidak terhubung")
        st.sidebar.caption(health["error"])
        return

    st.sidebar.success(f"API aktif · v{health.get('version', '?')} · {health.get('device', '-')}")
    for m in health.get("models", []):
        icon = "🟢" if m["available"] else "🔴"
        loaded = " (dimuat)" if m.get("loaded") else ""
        st.sidebar.caption(f"{icon} {m['name']}{loaded}")


def main() -> None:
    ctx = sidebar()
    PAGES[ctx["page"]](ctx)


if __name__ == "__main__":
    main()
