# GC–SI correlation graphs

This folder contains `plot_gc_si.py`, a small CLI that reads gold (GC) and silver (SI) intraday bar CSVs, merges them on timestamp, and writes **four PNG charts** for a calendar day you choose.

It also contains `plot_gc_si_weekly.py`, a separate CLI that writes **weekly variants** of the same four charts for the ISO week (Mon-Fri) containing your `--day`.

## What gets produced (exactly)

All outputs are named with your `--day` as `YYYY-MM-DD`.

| File | Contents |
|------|----------|
| `01_gc_si_overlay_<date>.png` | **Dual y-axes:** GC `SeqClose` vs time (left), SI `SeqClose` vs time (right). Only the chosen day’s bars. |
| `02_ratio_intraday_<date>.png` | **GC/SI ratio** (`close_gc / close_si`) vs time for that day. Y-axis auto-scales unless you pass `--ratio-ylim MIN MAX`. |
| `03_corr_bellcurve_<date>.png` | **Distribution of daily Pearson correlation** between GC and SI **intraday returns**, over a **lookback window** of prior trading days (see below). Histogram (counts) on the left y-axis; optional KDE on a **right** y-axis (density). Vertical lines at μ, ±1σ/±2σ/±3σ, and **today’s** daily correlation. Text box: μ, σ, today’s corr, deviation, z-score. |
| `04_ratio_and_bellcurve_<date>.png` | **Composite:** **Rolling intraday Pearson correlation** (20 consecutive 15m bars ≈ 5h window) vs **time** on the **bottom** x-axis; **y-axis = correlation**. Horizontal **dashed** lines at μ ± 1σ, ±2σ, ±3σ; **solid** lines at μ and **today’s daily** correlation. A **Gaussian KDE** of the **same historical daily correlations** as figure 3 is drawn using a **second x-axis** on the **top** (`twiny`): correlation on that axis, with **density normalized** to a relative height and **mapped into the correlation y-range** (shape only—not literal density units). Stats box matches figure 3 context. |

Weekly outputs are named with week token `YYYY-Www_MM-DD_to_MM-DD` (ISO Monday through Friday).

| File | Contents |
|------|----------|
| `01_gc_si_overlay_weekly_<week>.png` | **Dual y-axes:** GC `SeqClose` (left) and SI `SeqClose` (right) using all aligned 15m bars in the ISO Mon-Fri week. X-axis is compressed to remove overnight gaps, with ticks labeled by day (`Mon`, `Tue`, ...). |
| `02_ratio_intraday_weekly_<week>.png` | **GC/SI ratio** (`close_gc / close_si`) across all 15m bars in the ISO Mon-Fri week. Same gapless day-to-day x-axis treatment as weekly figure 01 and optional `--ratio-ylim`. |
| `03_corr_bellcurve_weekly_<week>.png` | **Weekly bell curve view:** histogram + KDE built from **15m rolling-correlation sequence samples** (20-bar rolling Pearson) from the **other days of that week** (excluding chosen `--day`). Vertical markers and stats still reference **today’s daily correlation** and derived μ/σ/z of the weekly sample. |
| `04_ratio_and_bellcurve_weekly_<week>.png` | **Weekly composite:** rolling intraday correlation vs time for the full week plus a top-axis KDE built from the **same 15m rolling-correlation weekly sample** used in weekly figure 03. Sigma bands, mean line, and today-correlation line are computed from that weekly sample. |
| `05_ratio_signal_bands_weekly_<week>.png` | **Weekly ratio signal bands:** GC/SI ratio across the full Mon-Fri week with horizontal lines for `μ` (solid), `μ±1σ` (dashed), and `μ±2σ` (dashed, emphasized). Here `μ` and `σ` come from the **prior 52 weeks of weekly mean ratio values**. Stats box includes `μ`, `σ`, week high/low ratio, ±2σ breach direction (`up`/`down`/`both`/`none`), and whether the week closed back inside ±1σ. |

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

**Weekly script (separate from daily):**

```bash
python gc-si-correlation-graphs/plot_gc_si_weekly.py --day 2026-03-10
```

This generates:

- `01_gc_si_overlay_weekly_<week>.png`
- `02_ratio_intraday_weekly_<week>.png`
- `03_corr_bellcurve_weekly_<week>.png`
- `04_ratio_and_bellcurve_weekly_<week>.png`
- `05_ratio_signal_bands_weekly_<week>.png`

Where `<week>` is `YYYY-Www_MM-DD_to_MM-DD` for the ISO Mon-Fri window containing `--day`.

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

For `plot_gc_si_weekly.py`, only `--day`, `--gc`, `--si`, `--out-dir`, and `--ratio-ylim` apply.

## Weekly behavior details

The weekly script is additive and does not replace daily outputs.

- **Figure 01 and 02 weekly:** still use the original 15-minute bars, but x-axis ticks are day names (`Mon`, `Tue`, ...) placed at the first available bar of each day.
- **Figure 03 weekly:** "today's correlation" is still the chosen day's daily Pearson correlation (from that day's 15-minute returns). The bell curve sample is built from 15m rolling-correlation sequence values from the *other* days in the same Mon-Fri week.
- **Figure 04 weekly:** uses the same weekly 15m rolling-correlation sample semantics as weekly figure 03.
- **Figure 05 weekly:** uses weekly **ratio** (not correlation). Baseline is the prior 52 weekly means (`mean(ratio)` within each prior ISO Mon-Fri week). A signal is flagged if any 15m bar breaches `μ±2σ`; the close condition reports whether the final bar is back inside `μ±1σ`.
- **Edge cases:** if the week has missing/sparse data, the script still completes. It renders fallback visuals/annotations when correlations cannot be computed with normal sample sizes.

## Troubleshooting

- **`--day` needs at least 8 bars:** That date has too few aligned bars after the merge.
- **Could not compute daily correlation for …:** Not enough valid intraday returns that day.
- **Need at least 2 prior days with valid correlation:** Expand history (earlier `--day`, larger `--lookback-days`, or ensure CSVs cover more dates).
- **`--day` is before Jan 1 of baseline year:** Use `--lookback-days` or a later `--day`.

Run `python gc-si-correlation-graphs/plot_gc_si.py --help` for the full CLI reference.
