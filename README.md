# 🦴 Bone Fracture Detection — REST API + Streamlit

Deployment dua model deteksi fraktur tulang pada citra X-ray:

| | YOLO26 | Faster R-CNN |
|---|---|---|
| Arsitektur | single-stage | ResNet50-FPN v2 (two-stage) |
| Bobot | `models/YOLO_best.pt` (±5 MB) | `models/best_frcnn.pt` (±174 MB) |
| Best mAP@0.5 | 0.889 (epoch 85) | 0.888 (epoch 61) |
| Best mAP@0.5:0.95 | 0.476 | 0.452 |

Sistem terdiri dari dua layanan terpisah:

```
┌──────────────────┐   HTTP/multipart   ┌──────────────────┐
│  Streamlit UI    │ ─────────────────► │   FastAPI REST   │
│  (port 8501)     │ ◄───────────────── │   (port 8000)    │
└──────────────────┘   JSON + base64    └────────┬─────────┘
                                                 │ SQLAlchemy
                                        ┌────────▼─────────┐
                                        │ SQLite/PostgreSQL│
                                        │ predictions +    │
                                        │ detections       │
                                        └──────────────────┘
```

---

## 1. Struktur project

```
bone-fracture-detection/
├── api/                        layanan REST (FastAPI)
│   ├── main.py                 entry point + /health
│   ├── config.py               konfigurasi dari .env
│   ├── preprocessing.py        CLAHE + letterbox (identik dengan training)
│   ├── inference.py            loader & wrapper kedua model
│   ├── metrics.py              pembaca *_history.csv
│   ├── database.py             engine & session SQLAlchemy
│   ├── db_models.py            tabel predictions & detections
│   ├── crud.py                 operasi database
│   ├── schemas.py              skema request/response
│   └── routers/
│       ├── predict.py          POST /predict/{model}
│       ├── predictions.py      riwayat prediksi
│       └── model_metrics.py    metrik performa model
├── ui/                         antarmuka Streamlit
│   ├── app.py                  sidebar + navigasi
│   ├── api_client.py           pemanggil REST API
│   └── views/
│       ├── detect.py           upload & prediksi real-time
│       ├── performance.py      grafik performa model
│       ├── history.py          riwayat + statistik
│       └── about.py
├── models/                     bobot (.pt tidak di-commit) + *_history.csv
├── tests/                      35 test, jalan tanpa file bobot
├── docker/                     Dockerfile api & ui
├── scripts/                    fetch_weights.py, smoke_test.py
├── .github/workflows/ci-cd.yml pipeline CI/CD
└── docker-compose.yml
```

---

## 2. Menjalankan secara lokal

### 2.1 Siapkan bobot

Salin empat file ini ke folder `models/` (lihat `models/README.md`):

```
models/YOLO_best.pt
models/best_frcnn.pt
models/YOLO_history.csv      ← sudah ikut di repo
models/FRCNN_history.csv     ← sudah ikut di repo
```

### 2.2 Pasang dependency

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-api.txt -r requirements-ui.txt
cp .env.example .env
```

> Tanpa GPU, pasang torch versi CPU agar jauh lebih ringan:
> `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu`

### 2.3 Jalankan

Terminal 1 — REST API:

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Terminal 2 — Streamlit:

```bash
streamlit run ui/app.py
# UI: http://localhost:8501
```

Uji cepat dari CLI:

```bash
python scripts/smoke_test.py path/ke/xray.jpg
```

### 2.4 Lewat Docker (sekali jalan)

```bash
docker compose up -d --build
# API  → http://localhost:8000/docs
# UI   → http://localhost:8501
```

Bobot di-mount dari host (`./models:/app/models:ro`), jadi image tetap kecil.
Untuk memakai PostgreSQL: `docker compose --profile postgres up -d` lalu ubah
`DATABASE_URL` pada service `api`.

---

## 3. Endpoint REST API

Base URL: `http://localhost:8000`, prefix `"/api/v1"`.

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/health` | status layanan, device, database, ketersediaan model |
| POST | `/api/v1/predict/yolo26` | prediksi dengan YOLO26 |
| POST | `/api/v1/predict/frcnn` | prediksi dengan Faster R-CNN |
| POST | `/api/v1/predict/compare/both` | jalankan kedua model pada satu gambar |
| GET | `/api/v1/predictions` | daftar riwayat (filter `model_key`, `class_name`, paginasi) |
| GET | `/api/v1/predictions/stats` | statistik agregat + distribusi kelas |
| GET | `/api/v1/predictions/{id}` | detail satu prediksi |
| DELETE | `/api/v1/predictions/{id}` | hapus prediksi (bbox ikut terhapus) |
| GET | `/api/v1/metrics/models` | status & metadata kedua model |
| GET | `/api/v1/metrics/training/{model}` | riwayat training per epoch (untuk grafik) |
| GET | `/api/v1/metrics/comparison` | ringkasan perbandingan kedua model |
| GET | `/static/{file}.png` | gambar beranotasi yang tersimpan |

### Contoh pemakaian

```bash
curl -X POST http://localhost:8000/api/v1/predict/yolo26 \
  -F "file=@xray.jpg" \
  -F "conf=0.25" \
  -F "save=true" \
  -F "return_image=false"
```

```json
{
  "prediction_id": 12,
  "model_key": "yolo26",
  "model_name": "YOLO26",
  "filename": "xray.jpg",
  "conf_threshold": 0.25,
  "image_width": 640,
  "image_height": 1024,
  "inference_ms": 83.4,
  "n_detections": 1,
  "top_class": "Transverse",
  "top_confidence": 0.8731,
  "detections": [
    {
      "class_id": 9,
      "class_name": "Transverse",
      "confidence": 0.8731,
      "box": [312.5, 402.1, 455.8, 491.0],
      "box_original": [196.1, 252.3, 286.0, 308.1]
    }
  ],
  "annotated_url": "/static/20260612-101500_yolo26_a1b2c3d4.png"
}
```

`box` berada pada kanvas kerja model (832×832), sedangkan `box_original`
sudah dipetakan kembali ke ukuran gambar yang diunggah pengguna — ini yang
dipakai untuk menggambar anotasi dan yang disimpan ke database.

---

## 4. Integrasi database

Dua tabel, relasi satu-ke-banyak dengan cascade delete:

**`predictions`** — `id`, `created_at`, `filename`, `model_key`, `model_name`,
`conf_threshold`, `image_width`, `image_height`, `n_detections`, `top_class`,
`top_confidence`, `inference_ms`, `annotated_path`, `notes`

**`detections`** — `id`, `prediction_id` (FK), `class_id`, `class_name`,
`confidence`, `x1`, `y1`, `x2`, `y2`

Default SQLite (`data/predictions.db`), dibuat otomatis saat startup. Untuk
PostgreSQL cukup ganti satu baris di `.env`:

```env
DATABASE_URL=postgresql+psycopg://fracture:fracture@db:5432/fracture
```

Tidak ada kode yang perlu diubah — SQLAlchemy menangani perbedaannya.

---

## 5. Private repository & CI/CD

### 5.1 Buat repo privat

```bash
git init && git add . && git commit -m "feat: API + Streamlit deteksi fraktur"
gh repo create bone-fracture-detection --private --source=. --push
```

Bobot `.pt` **tidak ikut** (sudah di `.gitignore`). Unggah sebagai GitHub Release:

```bash
gh release create v1.0.0 models/YOLO_best.pt models/best_frcnn.pt \
  --title "Model weights v1.0.0" --notes "YOLO26 mAP50 0.889 | FRCNN mAP50 0.888"
```

Saat deploy, tarik kembali dengan `python scripts/fetch_weights.py --repo <owner>/<repo> --release v1.0.0`.

### 5.2 Pipeline (`.github/workflows/ci-cd.yml`)

```
push/PR ─► lint (ruff) ─┐
                        ├─► build & push image ke GHCR ─► deploy (SSH)
push/PR ─► test (3.11, 3.12, coverage) ─┘        hanya pada branch main
```

* **lint** — `ruff check` + `ruff format --check`.
* **test** — `pytest` dengan `requirements-dev.txt` yang **tidak memuat torch
  maupun ultralytics**. Seluruh test dirancang tidak memerlukan file bobot,
  sehingga CI selesai dalam hitungan detik, bukan belasan menit.
* **build** — dua image (`-api` dan `-ui`) di-push ke `ghcr.io` dengan tag
  `latest` dan `sha-<commit>`, memakai cache GitHub Actions.
* **deploy** — SSH ke server, `docker compose pull && up -d`, lalu cek `/health`.

### 5.3 Secrets & variables yang perlu diisi

| Nama | Tipe | Keterangan |
|---|---|---|
| `DEPLOY_ENABLED` | variable | isi `true` untuk mengaktifkan job deploy |
| `DEPLOY_HOST` | secret | IP/hostname server |
| `DEPLOY_USER` | secret | user SSH |
| `DEPLOY_SSH_KEY` | secret | private key SSH |
| `DEPLOY_PATH` | secret | folder berisi `docker-compose.yml` di server |

`GITHUB_TOKEN` otomatis tersedia dan sudah cukup untuk push ke GHCR.

---

## 6. Streamlit UI

Empat halaman, dipilih lewat sidebar:

1. **🔍 Deteksi X-ray** — tiga cara input (unggah banyak file, kamera, atau
   ketik path file), slider confidence, dan mode prediksi otomatis. Hasil
   ditampilkan berdampingan: gambar asli vs gambar beranotasi, kartu metrik
   (kelas dominan, confidence, jumlah deteksi, waktu inferensi), dan tabel bbox.
2. **📈 Performa Model** — grafik dari riwayat training: kurva loss, kurva mAP,
   precision/recall, learning-rate schedule, overlay kedua model pada satu
   sumbu, serta bar chart perbandingan metrik pada epoch terbaik.
3. **🗂️ Riwayat Prediksi** — data dari database: statistik agregat, distribusi
   kelas, tabel riwayat yang bisa diunduh sebagai CSV, pratinjau gambar
   beranotasi, dan tombol hapus.
4. **ℹ️ Tentang** — penjelasan sistem dan status layanan.

Pemilihan model (YOLO26 / Faster R-CNN / bandingkan keduanya), confidence
threshold, dan opsi simpan-ke-database ada di sidebar sehingga berlaku di
semua halaman.

---

## 7. Testing & kualitas kode

```bash
pytest --cov=api --cov-report=term-missing     # 35 test
ruff check api ui tests
ruff format api ui tests
```

Cakupan test: preprocessing (letterbox, pemetaan bbox balik ke koordinat asli,
CLAHE), seluruh endpoint, lapisan database (cascade delete, agregasi), dan
alur prediksi lengkap memakai detektor tiruan.

---

## 8. Catatan penting

**Preprocessing harus identik dengan training.** `api/preprocessing.py`
mereplikasi persis pipeline notebook: grayscale → CLAHE (clip 2.0, tile 8×8) →
letterbox 832×832 dengan padding hitam simetris. Mengubah `IMG_SIZE` atau
`CLAHE_CLIP` di `.env` akan menurunkan akurasi tanpa memunculkan error apa pun.

**Model dimuat lazy.** Bobot baru dibaca saat request prediksi pertama, lalu
di-cache sebagai singleton. Request pertama karena itu lebih lambat (±10–30
detik untuk Faster R-CNN di CPU). Set `PRELOAD_MODELS=true` agar dimuat saat
startup.

**Faster R-CNN di CPU lambat.** Pada resolusi 832×832 satu gambar bisa memakan
2–8 detik. Untuk produksi, gunakan GPU atau turunkan `IMG_SIZE` (dengan
konsekuensi akurasi menurun).

**Bukan alat diagnosis.** Sistem ini untuk pembelajaran dan riset. Hasil
prediksi tidak boleh dijadikan dasar keputusan klinis.

---

## 9. Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `503 ... Bobot tidak ditemukan` | file `.pt` belum ada di `models/`; cek path di `.env` |
| UI: "API tidak terhubung" | API belum jalan, atau `API_BASE_URL` salah |
| `ImportError: libGL.so.1` | pakai `opencv-python-headless`, bukan `opencv-python` |
| Prediksi kosong terus | turunkan confidence threshold; pastikan `IMG_SIZE=832` |
| Grafik performa 404 | `*_history.csv` belum ada di `models/` |
| `database is locked` (SQLite) | beralih ke PostgreSQL untuk akses bersamaan |
