# Intan Stimulation Plotter

Python tool to read Intan `.rhs` files, detect edges on `ANALOG_IN 0`, extract time windows around each stimulation, and generate a **multi-page PDF** per channel.

## Project layout

- `src/core.py` — RHS I/O, Intan RHX filtering, stimulations, mmap stacks
- `src/plotting.py` — multi-panel PDF export
- `src/display_config.py` — panel visibility and legend settings
- `src/gui/` — tabbed Qt UI (PySide6)
- `src/cli.py` — command-line entry point
- `run_gui.py` — GUI launcher

## Install

```bash
pip install -r requirements.txt
```

## Usage

### GUI

```bash
python run_gui.py
```

The UI has 8 tabs: Files, Stimulation, PDF output, Intan filter, Spikes/PSTH, Zoom, Display, Performance.

### Command line

```bash
python src/cli.py "session01.rhs" --save-dir "plots"
```

## Main CLI options

- `--edge`: `falling`, `rising`, or `none`
- `--threshold`, `--pre`, `--post`
- `--zoom-mode`: `none`, `onset`, `trigger_end`, `both`
- `--zoom-onset-t0-s` / `--zoom-onset-t1-s`, `--zoom-end-t0-s` / `--zoom-end-t1-s`
- `--intan-spike-filter`, `--intan-filter-type`, `--intan-filter-order`, `--intan-filter-cutoff-hz`
- `--spike-threshold-mode`, `--psth-bin-window-s`, `--rms-window-s`

Amplifier traces are in **microvolts (µV)**.
