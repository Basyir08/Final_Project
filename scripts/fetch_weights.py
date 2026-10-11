#!/usr/bin/env python3
"""Unduh bobot model dari GitHub Release (repo privat didukung).

Contoh:
    export GITHUB_TOKEN=ghp_xxx
    python scripts/fetch_weights.py --repo owner/bone-fracture-detection --release v1.0.0

Dipakai saat deploy agar file .pt tidak perlu ikut di-commit ke Git.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

WEIGHTS = ("YOLO_best.pt", "best_frcnn.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/nama-repo")
    ap.add_argument("--release", default="latest", help="tag release, mis. v1.0.0")
    ap.add_argument("--dir", default="models", help="folder tujuan")
    args = ap.parse_args()

    dest = Path(args.dir)
    dest.mkdir(parents=True, exist_ok=True)

    if all((dest / w).exists() for w in WEIGHTS):
        print("Semua bobot sudah ada, tidak perlu mengunduh.")
        return 0

    if not os.getenv("GITHUB_TOKEN"):
        print("[WARN] GITHUB_TOKEN belum di-set; repo privat akan gagal diakses.")

    cmd = ["gh", "release", "download", "--repo", args.repo, "--dir", str(dest), "--clobber"]
    if args.release != "latest":
        cmd.insert(3, args.release)
    for w in WEIGHTS:
        cmd += ["--pattern", w]

    print("$", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("GitHub CLI (gh) tidak terpasang: https://cli.github.com/", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Gagal mengunduh (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode

    for w in WEIGHTS:
        p = dest / w
        status = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "TIDAK ADA"
        print(f"  {w:<16} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
