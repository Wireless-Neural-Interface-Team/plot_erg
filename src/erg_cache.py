"""Hierarchical, content-addressed cache for processed recordings.

Four dependency levels, each with its own key so a parameter change only
invalidates what actually depends on it:

1. ``raw``     — the ``.rhs`` file itself (wideband memmap).
2. ``filter``  — raw + Intan software filter settings (high-pass / low-pass).
3. ``segment`` — raw + segmentation (edges or fixed sections, pre/post window).
4. ``spike``   — filter + segment + spike detection settings.

Derived products (means, RMS profiles) depend on ``filter`` + ``segment``, which
is tracked as the ``derived`` key.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from config import AnalysisConfig

CACHE_DIR_NAME = ".erg_cache"
BUNDLE_MANIFEST_NAME = "manifest.json"
DATASET_SUFFIX = ".ergproc"
CACHE_FORMAT_VERSION = 1


def _stable_hash(payload: Any, *, length: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Identity of a source file without reading its contents."""
    stat = path.stat()
    return {
        "name": path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def filter_params(config: AnalysisConfig) -> dict[str, Any]:
    return {
        "hp_order": int(config.intan_hp_filter_order),
        "hp_type": str(config.intan_hp_filter_type),
        "hp_cutoff_hz": float(config.intan_hp_filter_cutoff_hz),
        "lp_order": int(config.intan_lp_filter_order),
        "lp_type": str(config.intan_lp_filter_type),
        "lp_cutoff_hz": float(config.intan_lp_filter_cutoff_hz),
        "software_notch_hz": int(getattr(config, "software_notch_hz", 0) or 0),
    }


def segmentation_params(config: AnalysisConfig) -> dict[str, Any]:
    if config.edge == "none":
        return {
            "edge": "none",
            "section_count": int(config.section_count),
            "section_duration_s": (
                None if config.section_duration_s is None else float(config.section_duration_s)
            ),
            "section_spec": str(config.section_spec),
            "trigger_start_s": float(config.section_trigger_start_s),
            "trigger_end_s": float(config.section_trigger_end_s),
        }
    return {
        "edge": str(config.edge),
        "threshold": float(config.threshold),
        "pre_s": float(config.pre_s),
        "post_s": float(config.post_s),
    }


def spike_params(config: AnalysisConfig) -> dict[str, Any]:
    return {
        "mode": str(config.spike_threshold_mode),
        "threshold_uv": float(config.spike_threshold_uv),
        "polarity": str(config.spike_threshold_polarity),
        "rms_multiplier": float(config.spike_threshold_rms_multiplier),
        "artifact_threshold_uv": float(config.intan_artifact_threshold_uv),
        "artifact_suppression": bool(config.intan_artifact_suppression_enabled),
        "rms_window_s": float(config.rms_window_s),
    }


def overlay_params(config: AnalysisConfig) -> dict[str, Any]:
    return {
        "pre_ms": float(config.spike_overlay_pre_ms),
        "post_ms": float(config.spike_overlay_post_ms),
    }


@dataclass(frozen=True)
class CacheKeys:
    """All cache keys derived from one :class:`AnalysisConfig`."""

    raw: str
    filtered: str
    segment: str
    derived: str
    spike: str
    overlay: str

    @classmethod
    def from_config(cls, config: AnalysisConfig) -> CacheKeys:
        fingerprint = file_fingerprint(config.rhs_file)
        raw = _stable_hash({"v": CACHE_FORMAT_VERSION, "file": fingerprint})
        filtered = _stable_hash({"raw": raw, "filter": filter_params(config)})
        segment = _stable_hash({"raw": raw, "segment": segmentation_params(config)})
        derived = _stable_hash({"filter": filtered, "segment": segment})
        spike = _stable_hash({"derived": derived, "spike": spike_params(config)})
        overlay = _stable_hash({"spike": spike, "overlay": overlay_params(config)})
        return cls(
            raw=raw,
            filtered=filtered,
            segment=segment,
            derived=derived,
            spike=spike,
            overlay=overlay,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "raw": self.raw,
            "filtered": self.filtered,
            "segment": self.segment,
            "derived": self.derived,
            "spike": self.spike,
            "overlay": self.overlay,
        }


def default_cache_root(config: AnalysisConfig) -> Path:
    """Cache location: explicit ``work_dir``, else next to the output or recording."""
    if config.work_dir is not None:
        return Path(config.work_dir)
    root = config.save_dir if config.save_dir is not None else config.rhs_file.parent
    return Path(root) / CACHE_DIR_NAME


@dataclass(frozen=True)
class RawStreamLayout:
    """Shared, filter-independent cache of one ``.rhs`` file.

    Keyed by the file itself, so changing the Intan filter or the segmentation
    never forces a second read of the recording.
    """

    root: Path

    @property
    def meta_path(self) -> Path:
        return self.root / "raw_meta.json"

    @property
    def amplifier_path(self) -> Path:
        return self.root / "amplifier.npy"

    @property
    def analog_in0_path(self) -> Path:
        return self.root / "analog_in0.npy"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def is_complete(self) -> bool:
        return self.meta_path.exists() and self.amplifier_path.exists()

    def read_meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def write_meta(self, payload: dict[str, Any]) -> None:
        self.ensure()
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.meta_path)


def raw_layout_for(cache_root: Path, keys: CacheKeys) -> RawStreamLayout:
    return RawStreamLayout(root=Path(cache_root) / f"_raw_{keys.raw}")


@dataclass(frozen=True)
class FilteredStreamLayout:
    """Per-filter-settings cache of HP / LP (and optional notch) channel rows.

    Channels are written on first access; a boolean mask tracks which rows are
    ready so cold starts never re-run ``sosfilt`` for a channel already on disk.
    """

    root: Path

    @property
    def meta_path(self) -> Path:
        return self.root / "filter_meta.json"

    @property
    def hp_path(self) -> Path:
        return self.root / "hp.npy"

    @property
    def lp_path(self) -> Path:
        return self.root / "lp.npy"

    @property
    def notch_path(self) -> Path:
        return self.root / "notch.npy"

    @property
    def ready_hp_path(self) -> Path:
        return self.root / "ready_hp.npy"

    @property
    def ready_lp_path(self) -> Path:
        return self.root / "ready_lp.npy"

    @property
    def ready_notch_path(self) -> Path:
        return self.root / "ready_notch.npy"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def is_initialized(self) -> bool:
        return self.meta_path.exists() and self.hp_path.exists() and self.lp_path.exists()

    def read_meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def write_meta(self, payload: dict[str, Any]) -> None:
        self.ensure()
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.meta_path)


def filtered_layout_for(cache_root: Path, keys: CacheKeys) -> FilteredStreamLayout:
    return FilteredStreamLayout(root=Path(cache_root) / f"_filter_{keys.filtered}")


@dataclass(frozen=True)
class BundleLayout:
    """Directory layout of one cached / exported recording bundle."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / BUNDLE_MANIFEST_NAME

    @property
    def streams_dir(self) -> Path:
        return self.root / "streams"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def spikes_dir(self) -> Path:
        return self.root / "spikes"

    @property
    def overlay_dir(self) -> Path:
        return self.root / "overlay"

    def stream_path(self, name: str) -> Path:
        return self.streams_dir / f"{name}.npy"

    def derived_path(self, name: str) -> Path:
        return self.derived_dir / f"{name}.npy"

    def ensure(self) -> None:
        for directory in (
            self.root,
            self.streams_dir,
            self.derived_dir,
            self.spikes_dir,
            self.overlay_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def size_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total


def bundle_for(cache_root: Path, keys: CacheKeys, stem: str) -> BundleLayout:
    """Bundle directory for one (recording, filter, segmentation) combination."""
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:48]
    return BundleLayout(root=Path(cache_root) / f"{safe_stem}_{keys.derived}")


def iter_bundles(cache_root: Path) -> Iterator[BundleLayout]:
    root = Path(cache_root)
    if not root.exists():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / BUNDLE_MANIFEST_NAME).exists():
            yield BundleLayout(root=child)


@dataclass(frozen=True)
class CacheEntryInfo:
    """Summary of one cached bundle, for the cache management UI."""

    root: Path
    source_name: str
    created_at: str
    size_bytes: int
    has_streams: bool
    n_channels: int
    n_trials: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024.0 * 1024.0)


def describe_cache(cache_root: Path) -> list[CacheEntryInfo]:
    entries: list[CacheEntryInfo] = []
    for bundle in iter_bundles(cache_root):
        try:
            manifest = bundle.read_manifest()
        except (OSError, ValueError):
            continue
        meta = manifest.get("meta", {})
        entries.append(
            CacheEntryInfo(
                root=bundle.root,
                source_name=str(meta.get("source_name", bundle.root.name)),
                created_at=str(manifest.get("created_at", "")),
                size_bytes=bundle.size_bytes(),
                has_streams=bool(manifest.get("has_streams", False)),
                n_channels=int(meta.get("n_channels", 0)),
                n_trials=int(meta.get("n_trials", 0)),
            )
        )
    return entries


def iter_raw_layouts(cache_root: Path) -> Iterator[RawStreamLayout]:
    root = Path(cache_root)
    if not root.exists():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("_raw_"):
            yield RawStreamLayout(root=child)


def _tree_size(root: Path) -> int:
    total = 0
    for path in Path(root).rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def cache_size_bytes(cache_root: Path) -> int:
    """Total cache footprint, including the shared raw-stream caches."""
    total = sum(bundle.size_bytes() for bundle in iter_bundles(cache_root))
    total += sum(_tree_size(layout.root) for layout in iter_raw_layouts(cache_root))
    return total


def remove_bundle(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def clear_cache(
    cache_root: Path,
    *,
    keep: set[Path] | None = None,
    include_raw: bool = True,
) -> int:
    """Delete cached bundles (and raw-stream caches). Returns the count removed."""
    protected = {Path(p).resolve() for p in (keep or set())}
    removed = 0
    for bundle in list(iter_bundles(cache_root)):
        if bundle.root.resolve() in protected:
            continue
        remove_bundle(bundle.root)
        removed += 1
    if include_raw:
        for layout in list(iter_raw_layouts(cache_root)):
            if layout.root.resolve() in protected:
                continue
            remove_bundle(layout.root)
            removed += 1
    root = Path(cache_root)
    try:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    except OSError:
        pass
    return removed


def prune_cache(cache_root: Path, *, max_bytes: int, keep: set[Path] | None = None) -> int:
    """Remove the oldest bundles until the cache fits in ``max_bytes``."""
    if max_bytes <= 0:
        return 0
    protected = {Path(p).resolve() for p in (keep or set())}
    bundles = [
        (bundle, bundle.size_bytes(), _bundle_mtime(bundle))
        for bundle in iter_bundles(cache_root)
        if bundle.root.resolve() not in protected
    ]
    total = sum(size for _b, size, _m in bundles)
    bundles.sort(key=lambda item: item[2])
    removed = 0
    for bundle, size, _mtime in bundles:
        if total <= max_bytes:
            break
        remove_bundle(bundle.root)
        total -= size
        removed += 1
    return removed


def _bundle_mtime(bundle: BundleLayout) -> float:
    try:
        return float(bundle.manifest_path.stat().st_mtime)
    except OSError:
        return 0.0


def human_bytes(n_bytes: float) -> str:
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def available_disk_bytes(path: Path) -> int:
    try:
        target = path
        while not target.exists() and target.parent != target:
            target = target.parent
        return int(shutil.disk_usage(target).free)
    except OSError:
        return 0


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
