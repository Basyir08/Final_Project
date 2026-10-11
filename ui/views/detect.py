"""Halaman 1 — unggah / input gambar X-ray dan lihat prediksi real-time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import api_client as api

CLASS_HINT = (
    "Kelas fraktur: Comminuted, Greenstick, Healthy, Linear, Oblique, "
    "Oblique Displaced, Segmental, Spiral, Transverse, Transverse Displaced."
)


def _collect_inputs() -> list[tuple[str, bytes]]:
    """Tiga cara input: upload file, kamera, atau path lokal (input manual)."""
    tab_upload, tab_cam, tab_path = st.tabs(
        ["📁 Unggah file", "📷 Kamera", "✍️ Input manual (path)"]
    )
    items: list[tuple[str, bytes]] = []

    with tab_upload:
        files = st.file_uploader(
            "Pilih satu atau beberapa gambar X-ray",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            accept_multiple_files=True,
        )
        items += [(f.name, f.getvalue()) for f in (files or [])]

    with tab_cam:
        shot = st.camera_input("Ambil foto lembar X-ray")
        if shot is not None:
            items.append(("camera.png", shot.getvalue()))

    with tab_path:
        st.caption("Ketik path file di server/komputer ini, satu baris satu file.")
        raw = st.text_area("Path gambar", placeholder="/data/xray/001.jpg", height=100)
        for line in [ln.strip() for ln in raw.splitlines() if ln.strip()]:
            try:
                with open(line, "rb") as fh:
                    items.append((line.rsplit("/", 1)[-1], fh.read()))
            except OSError as exc:
                st.warning(f"Gagal membaca {line}: {exc.__class__.__name__}")

    return items


def _detections_df(detections: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "kelas": d["class_name"],
            "confidence": d["confidence"],
            "x1": d["box_original"][0],
            "y1": d["box_original"][1],
            "x2": d["box_original"][2],
            "y2": d["box_original"][3],
        }
        for d in detections
    ]
    return pd.DataFrame(rows)


def _show_result(result: dict, original: bytes) -> None:
    left, right = st.columns(2)
    with left:
        st.image(original, caption="Input asli", use_container_width=True)
    with right:
        img = api.decode_b64_image(result.get("annotated_image_b64"))
        caption = f"{result['model_name']} — {result['n_detections']} deteksi"
        if img:
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.info("Gambar beranotasi tidak dikembalikan API.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediksi dominan", result["top_class"] or "—")
    c2.metric(
        "Confidence",
        f"{result['top_confidence']:.1%}" if result["top_confidence"] else "—",
    )
    c3.metric("Jumlah deteksi", result["n_detections"])
    c4.metric("Waktu inferensi", f"{result['inference_ms']:.0f} ms")

    if result["detections"]:
        st.dataframe(
            _detections_df(result["detections"]), use_container_width=True, hide_index=True
        )
    else:
        st.warning(
            "Tidak ada fraktur terdeteksi di atas threshold. "
            "Turunkan confidence threshold di sidebar bila ingin melihat deteksi lemah."
        )

    if result.get("prediction_id"):
        st.caption(f"Tersimpan di database dengan ID #{result['prediction_id']}")


def _show_compare(pair: dict, original: bytes) -> None:
    cols = st.columns(3)
    cols[0].image(original, caption="Input asli", use_container_width=True)
    for col, key in zip(cols[1:], ("yolo26", "frcnn")):
        res = pair[key]
        img = api.decode_b64_image(res.get("annotated_image_b64"))
        if img:
            col.image(
                img,
                caption=f"{res['model_name']} — {res['n_detections']} deteksi "
                f"({res['inference_ms']:.0f} ms)",
                use_container_width=True,
            )

    rows = []
    for key in ("yolo26", "frcnn"):
        res = pair[key]
        rows.append(
            {
                "model": res["model_name"],
                "deteksi": res["n_detections"],
                "prediksi dominan": res["top_class"] or "—",
                "confidence": res["top_confidence"],
                "inferensi (ms)": round(res["inference_ms"], 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    detail = []
    for key in ("yolo26", "frcnn"):
        for d in pair[key]["detections"]:
            detail.append(
                {
                    "model": pair[key]["model_name"],
                    "kelas": d["class_name"],
                    "confidence": d["confidence"],
                }
            )
    if detail:
        with st.expander("Detail seluruh deteksi"):
            st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)


def render(ctx: dict) -> None:
    st.title("🔍 Deteksi Fraktur pada Citra X-ray")
    st.caption(CLASS_HINT)

    items = _collect_inputs()
    if not items:
        st.info("Unggah gambar, ambil foto lewat kamera, atau isi path file untuk memulai.")
        return

    st.divider()
    run = st.button("🚀 Jalankan prediksi", type="primary", use_container_width=True)
    auto = st.toggle(
        "Prediksi otomatis saat gambar/threshold berubah",
        value=True,
        help="Mode real-time: setiap perubahan langsung memicu inferensi ulang.",
    )
    if not (run or auto):
        return

    for name, content in items:
        st.subheader(name)
        try:
            with st.spinner("Memproses..."):
                if ctx["model_key"] == "both":
                    pair = api.predict_both(
                        ctx["base_url"],
                        filename=name,
                        content=content,
                        conf=ctx["conf"],
                        save=ctx["save"],
                    )
                    _show_compare(pair, content)
                else:
                    result = api.predict(
                        ctx["base_url"],
                        model_key=ctx["model_key"],
                        filename=name,
                        content=content,
                        conf=ctx["conf"],
                        save=ctx["save"],
                    )
                    _show_result(result, content)
        except api.ApiError as exc:
            st.error(str(exc))
        st.divider()
