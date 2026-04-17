#!/usr/bin/env python3
"""Lanceur GUI 100% Python pour Intan Trigger Plotter."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cli import main


if __name__ == "__main__":
    # Force le mode GUI Qt.
    sys.argv = [sys.argv[0], "--gui"]
    main()
