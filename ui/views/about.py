"""Halaman 4 — keterangan singkat sistem."""

from __future__ import annotations

import streamlit as st

from ui import api_client as api

TEXT = """
### Tentang sistem

Aplikasi ini mendeteksi dan mengklasifikasikan **fraktur tulang pada citra X-ray**
menggunakan dua pendekatan deteksi objek yang dibandingkan secara berdampingan:

| | YOLO26 | Faster R-CNN |
|---|---|---|
| Arsitektur | single-stage | two-stage (ResNet50-FPN v2) |
| Kelebihan | inferensi sangat cepat | umumnya lebih teliti pada objek kecil |
| Kekurangan | bisa melewatkan garis fraktur halus | lambat & berat di memori |

**Alur data**

1. Gambar diunggah lewat Streamlit → dikirim ke REST API (FastAPI).
2. API menjalankan preprocessing yang identik dengan saat training:
   grayscale → CLAHE (clip 2.0, tile 8×8) → letterbox 832×832.
3. Model terpilih melakukan inferensi; koordinat bbox dipetakan kembali
   ke ukuran gambar asli.
4. Hasil disimpan ke database (tabel `predictions` dan `detections`)
   beserta gambar beranotasinya.

**Peringatan klinis.** Sistem ini dibuat untuk keperluan pembelajaran dan
riset. Hasil prediksi bukan diagnosis medis dan tidak boleh dijadikan dasar
keputusan klinis tanpa pemeriksaan tenaga medis berkualifikasi.
"""


def render(ctx: dict) -> None:
    st.title("ℹ️ Tentang")
    st.markdown(TEXT)

    st.divider()
    st.subheader("Status layanan")
    try:
        st.json(api.health(ctx["base_url"]))
    except api.ApiError as exc:
        st.error(str(exc))

    st.caption(f"Dokumentasi API: {ctx['base_url'].rstrip('/')}/docs")
