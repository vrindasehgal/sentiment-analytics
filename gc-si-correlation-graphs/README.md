# GC–SI correlation graphs

This folder contains `plot_gc_si.py`, a small CLI that reads gold (GC) and silver (SI) intraday bar CSVs, merges them on timestamp, and writes **four PNG charts** for a calendar day you choose.

## What gets produced (exactly)

All outputs are named with your `--day` as `YYYY-MM-DD`.

| File | Contents |
|------|----------|
| `01_gc_si_overlay_<date>.png` | **Dual y-axes:** GC `SeqClose` vs time (left), SI `SeqClose` vs time (right). Only the chosen day’s bars. |
| `02_ratio_intraday_<date>.png` | **GC/SI ratio** (`close_gc / close_si`) vs time for that day. Y-axis auto-scales unless you pass `--ratio-ylim MIN MAX`. |
| `03_corr_bellcurve_<date>.png` | **Distribution of daily Pearson correlation** between GC and SI **intraday returns**, over a **lookback window** of prior trading days (see below). Histogram (counts) on the left y-axis; optional KDE on a **right** y-axis (density). Vertical lines at μ, ±1σ/±2σ/±3σ, and **today’s** daily correlation. Text box: μ, σ, today’s corr, deviation, z-score. |
| `04_ratio_and_bellcurve_<date>.png` | **Composite:** **Rolling intraday Pearson correlation** (20 consecutive 15m bars ≈ 5h window) vs **time** on the **bottom** x-axis; **y-axis = correlation**. Horizontal **dashed** lines at μ ± 1σ, ±2σ, ±3σ; **solid** lines at μ and **today’s daily** correlation. A **Gaussian KDE** of the **same historical daily correlations** as figure 3 is drawn using a **second x-axis** on the **top** (`twiny`): correlation on that axis, with **density normalized** to a relative height and **mapped into the correlation y-range** (shape only—not literal density units). Stats box matches figure 3 context. |

**Naming note:** Figure 4’s filename still contains `ratio_and_bellcurve` for historical reasons; the chart is about **correlation**, not the GC/SI price ratio.

## How the script works (pipeline)

1. **Load and merge** `GC.csv` and `SI.csv` on  
   `Year, Month, Day, HourOfDay, MinOfHour`  
   **Inner join**—only rows present in both files are kept. Timestamps must align for meaningful plots.

2. **Per calendar day**, compute **one daily Pearson correlation** of **pct-change** returns on GC vs SI closes, using only bars where both returns exist; require at least **8** bars that day (`MIN_BARS_PER_DAY`) or the day is invalid for correlation.

3. **Chosen day (`--day`):** subset merged data to that date for figures 1, 2, and 4.

4. **Historical sample for figures 3 and 4** (and for μ, σ, z on figure 4):  
   - **Default:** all trading days in `--baseline-year` from **Jan 1** through the **day before** `--day`.  
   - **Override:** `--lookback-days N` uses the **last N** trading days **before** `--day` (ignores the YTD window).  
   You need **at least 2** prior days with valid daily correlation or the script exits before drawing figures 3–4.

5. **Figure 4 rolling series:** For the chosen day only, `pct_change` on GC and SI closes, then a **rolling window correlation** with length `ROLLING_CORR_BARS` (20). KDE uses a lightweight **Gaussian KDE** (Silverman bandwidth) implemented in-script—no SciPy.

6. **Styling:** Non-interactive Matplotlib backend (`Agg`), dark theme, 150 DPI PNGs.

## Requirements

- Python 3.10+ recommended (uses `list[str]` typing style).
- Dependencies: see [`requirements-plot.txt`](requirements-plot.txt) (`pandas`, `matplotlib`).

Install (from repo root or this folder):

```bash
pip install -r gc-si-correlation-graphs/requirements-plot.txt
```

## How to run

Default data paths expect **`../gc-si/GC.csv`** and **`../gc-si/SI.csv`** relative to this directory (i.e. `sentiment-analytics/gc-si/` when you run from the repo).

**Minimal example** (writes to `gc-si-correlation-graphs/plots/` by default):

```bash
python gc-si-correlation-graphs/plot_gc_si.py --day 2026-03-10
```

**Explicit paths:**

```bash
python gc-si-correlation-graphs/plot_gc_si.py \
  --day 2026-03-10 \
  --gc path/to/GC.csv \
  --si path/to/SI.csv \
  --out-dir gc-si-correlation-graphs/plots
```

**Use last 60 trading days before `--day` for the bell curve / KDE baseline:**

```bash
python gc-si-correlation-graphs/plot_gc_si.py --day 2026-03-10 --lookback-days 60
```

**Fix ratio y-limits on figure 2:**

```bash
python gc-si-correlation-graphs/plot_gc_si.py --day 2026-03-10 --ratio-ylim 75 95
```

**Useful flags**

| Flag | Role |
|------|------|
| `--day` | Required. Session date `YYYY-MM-DD`. |
| `--gc`, `--si` | CSV paths (defaults: `../gc-si/GC.csv`, `../gc-si/SI.csv`). |
| `--out-dir` | Output directory for PNGs (default: `plots/` under this folder). |
| `--baseline-year` | Year for YTD lookback when `--lookback-days` is omitted (default: 2026). |
| `--lookback-days` | If set, use last N trading days before `--day` instead of YTD. |
| `--ratio-ylim` | Two floats: min and max y for figure 2 only. |

## Troubleshooting

- **`--day` needs at least 8 bars:** That date has too few aligned bars after the merge.
- **Could not compute daily correlation for …:** Not enough valid intraday returns that day.
- **Need at least 2 prior days with valid correlation:** Expand history (earlier `--day`, larger `--lookback-days`, or ensure CSVs cover more dates).
- **`--day` is before Jan 1 of baseline year:** Use `--lookback-days` or a later `--day`.

Run `python gc-si-correlation-graphs/plot_gc_si.py --help` for the full CLI reference.
