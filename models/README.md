# Folder bobot model

File `.pt` **tidak di-commit** ke Git (lihat `.gitignore`) karena ukurannya besar:

| File | Isi | Ukuran |
|---|---|---|
| `YOLO_best.pt` | bobot YOLO26 hasil fine-tuning | ± 5 MB |
| `best_frcnn.pt` | checkpoint Faster R-CNN (`model`, `classes`, `img_size`, `epoch`) | ± 174 MB |
| `YOLO_history.csv` | log training per epoch YOLO26 (115 epoch) | ikut di-commit |
| `FRCNN_history.csv` | log training per epoch Faster R-CNN (81 epoch) | ikut di-commit |

## Cara menaruh bobot

**Lokal** — salin manual dari Google Drive ke folder ini:

```
models/
├── YOLO_best.pt
├── best_frcnn.pt
├── YOLO_history.csv
└── FRCNN_history.csv
```

**Server / CI** — pilih salah satu:

1. **GitHub Release (disarankan untuk repo privat)**
   ```bash
   gh release create v1.0.0 models/YOLO_best.pt models/best_frcnn.pt \
       --repo <owner>/<repo> --title "Model weights v1.0.0"
   # saat deploy
   gh release download v1.0.0 --dir models --repo <owner>/<repo>
   ```

2. **Git LFS**
   ```bash
   git lfs install
   git lfs track "models/*.pt"
   git add .gitattributes models/*.pt && git commit -m "add weights via LFS"
   ```

3. **Object storage** (S3/GCS/MinIO) — unduh saat container start.

Skrip bantu: `python scripts/fetch_weights.py --release v1.0.0`.
