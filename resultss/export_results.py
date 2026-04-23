#!/usr/bin/env python3
"""
Pack all PySR / Stacking outputs into a single ZIP for easy Kaggle download.

Usage (in a Kaggle cell):
    !python resultss/export_results.py

Output: /kaggle/working/ALL_RESULTS.zip  (visible in Kaggle Output panel)
        OR resultss/ALL_RESULTS.zip if not on Kaggle
"""
import zipfile
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
RESULTSS = ROOT / "resultss"

INCLUDE_DIRS = ["equations", "figures", "logs", "models"]
INCLUDE_EXTS = {".txt", ".latex", ".json", ".png", ".csv", ".log"}
SKIP_LARGE   = {"model_stacking.pkl", "scaler_X.pkl", "pysr_model.pkl"}

# Save to /kaggle/working/ so it appears in Kaggle Output panel
kaggle_out = Path("/kaggle/working")
out_zip = (kaggle_out if kaggle_out.exists() else RESULTSS) / "ALL_RESULTS.zip"

with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for folder in INCLUDE_DIRS:
        d = RESULTSS / folder
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix not in INCLUDE_EXTS:
                continue
            if f.name in SKIP_LARGE:
                continue
            arcname = f.relative_to(ROOT)
            zf.write(f, arcname)
            print(f"  + {arcname}")

size_kb = out_zip.stat().st_size / 1024
print(f"\nSaved → {out_zip}  ({size_kb:.1f} KB)")
print("Download it from Kaggle Output panel on the right.")
