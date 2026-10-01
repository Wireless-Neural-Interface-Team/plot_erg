# Intan Stimulation Plotter

Python tool to read Intan `.rhs` recordings, detect stimulations on `ANALOG_IN 0`,
extract time windows around each one, and explore every resulting curve
**interactively in the GUI** — with the multi-page PDF report still available.

A recording is processed **once**. Everything the panels need (trial averages,
per-stimulation windows, sliding RMS profiles, spike trains, spike waveforms,
thresholds) is computed, cached on disk, and from then on drawn from the cache.
Display settings therefore apply in real time, and a recording can be exported as
a **reusable processed dataset** that reopens in a few hundred milliseconds,
without the original `.rhs` file.

## Project layout

### Analysis core

- `src/core.py` — RHS I/O, Intan RHX filtering, stimulation detection, memmap stacks
- `src/intan_rhx_dsp.py` — Intan-compatible filters, spike detection, sliding RMS
- `src/probe_layout.py` — MEA probe geometry (probeinterface and mea_editor JSON)
- `src/impedance_tracking.py` — companion impedance CSV exports
- `src/plotting.py` — the panel drawing routines, shared by the screen and the PDF

### Processed data and cache

- `src/erg_cache.py` — staged, content-addressed cache keys and bundle layout
- `src/processed_dataset.py` — the processed-recording model and its on-disk format
- `src/dataset_builder.py` — the processing pipeline, with progress and stage timings

### Display

- `src/view_config.py` — live display settings: legends, panel style, views and tabs
- `src/panel_registry.py` — the catalogue of panels and how each one is drawn
- `src/display_config.py` — panel visibility and legend settings for the PDF
- `src/gui/` — the Qt viewer (PySide6): `main_window.py`, `jobs.py`, `widgets/`
- `src/cli.py` — command-line entry point
- `run_gui.py` — GUI launcher

## Install

```bash
pip install -r requirements.txt
```

## The viewer

```bash
python run_gui.py
```

### Window layout

The window is a set of docks around the views; every dock can be moved, stacked,
or hidden from the **View** menu.

- **Recordings** — the files being compared. Each row has its own legend label,
  curve colour, and two independent toggles (draw it, show it in legends).
- **Channels** — the clickable MEA map plus a filterable channel list; they stay in
  sync, and the map contacts can be shaded by mean RMS or spike count so the
  channels worth looking at stand out.
- **Parameters** — three tabs: *Display* and *Legend & style* apply immediately,
  *Processing* requires reprocessing (and says so).
- **Progress & log** — per-stage progress, a table of loading times, and the full
  pipeline log.

### Views and panels

The centre of the window holds **view tabs**, each one an independent grid of
panels. Use *View → Configure panels* (`Ctrl+P`) to pick which graphs a view
shows, for which temporal section (full view, onset zoom, end zoom), in which
order, and over how many columns. Views can be added, renamed, duplicated,
removed, and the whole layout saved to or loaded from a JSON file.

Each panel has its own header showing how long its last redraw took, a toolbar
for matplotlib zoom/pan/save, and buttons to open it in its own window or remove
it from the view. Panels scrolled far from the viewport are drawn when they come
into reach, which is what keeps a view with dozens of panels responsive.

### Available panels

Every panel of the PDF report is available on screen:

- Trial-averaged raw, high-pass, and low-pass traces
- First and second stimulation, raw / high-pass / low-pass
- Sliding RMS, trial-averaged and per stimulation
- PSTH and firing rate, ISI, rate per trial, raster
- Spike waveform overlay
- MEA layout for the selected channel, impedance evolution
- Summary pages: mean RMS across channels, mean RMS per channel, mean impedance
- All-channel montages (trial-averaged and second stimulation)

Channel-aware panels can be instantiated once per temporal section, so the same
graph can be compared side by side at different zoom levels.

## Processed datasets and cache

Processing is staged, and each stage is keyed by the parameters it actually
depends on. Changing the spike threshold reuses the cached filtered streams;
changing the filter reuses the cached wideband read. The cache lives in
`.erg_cache/` next to the recording (or in the folder set in *Processing → Cache /
work folder*) and can be inspected and trimmed from *Process → Cache manager*.

*File → Export processed dataset* writes a self-contained `.ergproc` folder
(optionally zipped for transport). Reopen it with *File → Open processed dataset*:
every panel works, on any machine, without the `.rhs` file.

## Command line

```bash
python src/cli.py "session01.rhs" --save-dir "plots"
```

### Main CLI options

- `--edge`: `falling`, `rising`, or `none`
- `--threshold`, `--pre`, `--post`
- `--zoom-mode`: `none`, `onset`, `trigger_end`, `both`
- `--zoom-onset-t0-s` / `--zoom-onset-t1-s`, `--zoom-end-t0-s` / `--zoom-end-t1-s`
- `--intan-filter-type`, `--intan-filter-order`, `--intan-filter-cutoff-hz`
  (high-pass and low-pass are always both applied, separately)
- `--spike-threshold-mode`, `--psth-bin-window-s`, `--rms-window-s`
- `--spike-overlay-pre-ms`, `--spike-overlay-post-ms`

Amplifier traces are in **microvolts (µV)**.

## Checks

The scripts in `tools/` run without any recording, on synthetic data:

```bash
python tools/smoke_render.py    # draw every panel of the catalogue
python tools/smoke_dataset.py   # write, reopen and compare a processed dataset
python tools/smoke_gui.py       # build the window offscreen and redraw every panel
```
