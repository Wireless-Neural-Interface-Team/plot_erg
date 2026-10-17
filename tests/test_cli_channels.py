"""Unit tests for CLI channel selection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import _parse_channels_arg, _resolve_pdf_channels  # noqa: E402


def test_parse_channels_arg_ranges() -> None:
    assert _parse_channels_arg(None) is None
    assert _parse_channels_arg("") is None
    assert _parse_channels_arg("0,3,5-7") == (0, 3, 5, 6, 7)
    assert _parse_channels_arg("8-5") == (5, 6, 7, 8)


def test_resolve_pdf_channels_all_and_subset() -> None:
    assert _resolve_pdf_channels(4, None) == [0, 1, 2, 3]
    assert _resolve_pdf_channels(8, (0, 2, 2, 7, 99, -1)) == [0, 2, 7]


def test_resolve_pdf_channels_empty_raises() -> None:
    with pytest.raises(ValueError, match="No valid"):
        _resolve_pdf_channels(4, (99, -3))
