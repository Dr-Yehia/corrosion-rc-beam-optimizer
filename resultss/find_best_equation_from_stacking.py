#!/usr/bin/env python3
"""Convenience entrypoint for Kaggle users.

This wrapper exists to make the main script easy to find inside `resultss/`.
It forwards all CLI arguments to `pysr_stacking_moead_selector.py`.
"""

from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "pysr_stacking_moead_selector.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing target script: {target}")

    # Preserve CLI args exactly for the target script
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
