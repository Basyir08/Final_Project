#!/usr/bin/env python3
"""Uji cepat end-to-end terhadap API yang sedang berjalan.

    python scripts/smoke_test.py path/ke/xray.jpg --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if not args.image.exists():
        print(f"File tidak ditemukan: {args.image}", file=sys.stderr)
        return 1

    health = requests.get(f"{args.base_url}/health", timeout=30).json()
    print(f"API v{health['version']} | device={health['device']} | db={health['database']}")
    for m in health["models"]:
        print(f"  {m['name']:<14} tersedia={m['available']}")

    content = args.image.read_bytes()
    for key in ("yolo26", "frcnn"):
        resp = requests.post(
            f"{args.base_url}/api/v1/predict/{key}",
            files={"file": (args.image.name, content)},
            data={"conf": str(args.conf), "save": "true", "return_image": "false"},
            timeout=300,
        )
        if resp.status_code != 200:
            print(f"[{key}] GAGAL {resp.status_code}: {resp.text[:200]}")
            continue
        body = resp.json()
        print(
            f"[{body['model_name']}] {body['n_detections']} deteksi | "
            f"dominan={body['top_class']} | {body['inference_ms']:.0f} ms | "
            f"id={body['prediction_id']}"
        )
        for d in body["detections"][:5]:
            print(f"    {d['class_name']:<22} {d['confidence']:.3f}  {d['box_original']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
