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

- `src/core.py` — RHS I/O helpers, Intan RHX filtering, stimulation detection, memmap stacks
- `src/intan_rhx_dsp.py` — Intan-compatible filters, spike detection, sliding RMS
- `src/probe_layout.py` — MEA probe geometry (probeinterface and mea_editor JSON)
- `src/impedance_tracking.py` — companion impedance CSV exports
- `src/plotting.py` — matplotlib PDF composition (Agg); can seed from `ProcessedRecording`
- `src/load_intan_rhs_format.py` / `src/intanutil/` — RHS reader with **stream-to-memmap** path

### Processed data and cache

- `src/erg_cache.py` — staged, content-addressed cache keys and bundle layout
- `src/processed_dataset.py` — the processed-recording model and its on-disk format
- `src/dataset_builder.py` — pipeline (`build_recording` + lazy `LazyFilterBank` + `ensure_channels`)

### Display

- `src/panel_catalog.py` — single catalogue of panel keys, labels and metadata
- `src/view_config.py` — live display settings: legends, panel style, views and tabs
- `src/panel_prepare.py` — backend-neutral panel data specs (screen + future PDF reuse)
- `src/panel_registry.py` — legacy matplotlib panel helpers (PDF / smokes)
- `src/display_config.py` — PDF section visibility and recording styles
- `src/gui/` — Qt viewer (PySide6 + **pyqtgraph** screen plots)
  - `services/` — redraw, pipeline, montage, preview, export, render-request factory
  - `widgets/plot_host.py` — pyqtgraph plot surface
  - `jobs.py` — background build / ensure / filter warm (never on the UI thread)
- `src/cli.py` — command-line entry point (PDF via the same `build_recording` path)
- `run_gui.py` — GUI launcher

### Performance model

- **UI thread**: read ready buffers + draw only (no `sosfilt`, no RHS I/O, no `savefig`)
- **Workers**: F5 build, channel ensure, filter prefetch, PDF/dataset/image export
- **RHS load**: decode blocks straight into a float32 memmap (no full-file double copy)
- **Filters**: lazy HP/LP with RAM LRU + disk store (batch flush)

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `PySide6`, `pyqtgraph`.

## The viewer

```bash
python run_gui.py
```

### Window layout

Channel-first workflow: the centre shows a **light preview of the selected
channel**. Range bars live in the per-channel inspector, not on a global
montage. The all-channel montage is optional (*Vue → Revue montage*, `Ctrl+M`).

The window is a set of docks around the views; every dock can be moved, stacked,
or hidden from the **Vue** menu (`Ctrl+P` toggles Paramètres, `Ctrl+J` the log).

- **Recordings** — the files being compared. Each row has its own legend label,
  curve colour, and two independent toggles (draw it, show it in legends).
- **Channels** — the clickable MEA map plus a filterable channel list; they stay in
  sync, and the map contacts can be shaded by mean RMS or spike count so the
  channels worth looking at stand out. **Double-click** a contact or list row to
  inspect that channel (continuous traces + adjustable range bars).
- **Parameters** — two tabs: *Affichage* applies immediately;
  *Canal → Pipeline* requires reprocessing (and says so).
- **Progress & log** — per-stage progress, a table of loading times, and the full
  pipeline log.

Menus: **Fichier**, **Traitement** (F5 / F6 / F7, cache), **Vue** (montage,
configure panels with `Ctrl+Shift+P`), **Aide**.

### Views and panels

The centre holds the channel preview by default. Use *Vue → Revue montage*
(`Ctrl+M`) for the optional multi-channel montage, then *Vue → Configurer les
panneaux* (`Ctrl+Shift+P`) to customise its grid. In the channel inspector, use
**+ Absolue** / **+ Rel. stim**, then *Appliquer les zooms* for analysis graphs.

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

Filtered HP/LP (and notch) channels are also persisted under
`.erg_cache/_filter_<hash>/` as float32 memmaps, filled channel-by-channel on
first access. That directory can grow to roughly
`8 bytes × n_samples × n_channels` (HP + LP); it is invalidated automatically when
filter settings change. An in-RAM LRU (32 channels) sits in front of the disk
cache so switching between recently viewed MEA contacts stays responsive.

*Fichier → Exporter un dataset traité* writes a self-contained `.ergproc` folder
(optionally zipped for transport). Compute the channels you care about first
(F6 / F7); the export packs their trial averages, RMS, spikes and overlays even
when they were filled on demand. Reopen with *Fichier → Ouvrir traité*: every
panel works, on any machine, without the `.rhs` file.

## Command line

```bash
python src/cli.py "session01.rhs" --save-dir "plots"
python src/cli.py a.rhs b.rhs --save-dir "plots"   # multi-recording PDF
```

### Main CLI options

- `--edge`: `falling`, `rising`, or `none`
- `--threshold`, `--pre`, `--post`
- `--zoom-mode`: `none`, `onset`, `trigger_end`, `both`
- `--zoom-onset-t0-s` / `--zoom-onset-t1-s`, `--zoom-end-t0-s` / `--zoom-end-t1-s`
- `--intan-hp-filter-type`, `--intan-hp-filter-order`, `--intan-hp-filter-cutoff-hz`
  (HIGH / passe-haut)
- `--intan-lp-filter-type`, `--intan-lp-filter-order`, `--intan-lp-filter-cutoff-hz`
  (LOW / passe-bas ; paramètres indépendants du HIGH)
- `--spike-threshold-mode`, `--psth-bin-window-s`, `--rms-window-s`
- `--spike-overlay-pre-ms`, `--spike-overlay-post-ms`

Amplifier traces are in **microvolts (µV)**.

## Checks

The scripts in `tools/` run without any recording, on synthetic data:

```bash
python tools/smoke_render.py    # draw every panel of the catalogue
python tools/smoke_dataset.py   # write, reopen and compare a processed dataset
python tools/smoke_gui.py       # build the window offscreen and redraw every panel
python tools/smoke_perf.py      # timings: means, incremental redraw, filter disk cache
```
