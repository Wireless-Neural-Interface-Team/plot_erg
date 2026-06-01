#!/usr/bin/env python3
"""Pure-Python launcher for the Intan Trigger Plotter Qt GUI."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cli import main


if __name__ == "__main__":
    # Force Qt GUI mode.
    sys.argv = [sys.argv[0], "--gui"]
    main()
