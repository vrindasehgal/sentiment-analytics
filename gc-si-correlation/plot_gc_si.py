#!/usr/bin/env python3
"""
GC–SI intraday charts from merged 15m bars.

CSV inputs live in ../gc-si/ by default (GC.csv, SI.csv).

Install: pip install -r requirements-plot.txt  (pandas, matplotlib).

  python gc-si-correlation-graphs/plot_gc_si.py --day 2026-01-02 --out-dir gc-si-correlation-graphs/plots

Bell-curve baseline: by default, all trading days in --baseline-year from Jan 1 up to (not including)
--day. Override with --lookback-days N for the last N days before --day.

Use a --day that exists in both GC.csv and SI.csv after the inner merge (timestamps must align).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

MIN_BARS_PER_DAY = 8
ROLLING_CORR_BARS = 20  # 15m bars (~5h) for intraday rolling Pearson on fig 4 overlay
KEY = ["Year", "Month", "Day", "HourOfDay", "MinOfHour"]
BASELINE_YEAR_DEFAULT = 2026


def _default_paths() -> tuple[Path, Path]:
    """(project_root_gc_si_correlation_graphs_dir, gc_si_data_dir)."""
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    return here, data_dir


def parse_args() -> argparse.Namespace:
    root, data_dir = _default_paths()
    p = argparse.ArgumentParser(description="Plot GC/SI ratio and correlation figures.")
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument("--day", type=str, required=True, help="Chosen day YYYY-MM-DD (figs 1–4)")
    p.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="If set, use last N trading days before --day (ignores baseline-year window)",
    )
    p.add_argument(
        "--baseline-year",
        type=int,
        default=BASELINE_YEAR_DEFAULT,
        help=f"With default lookback: use trading days from Jan 1 this year through day before --day (default {BASELINE_YEAR_DEFAULT})",
    )
    p.add_argument("--out-dir", type=Path, default=root / "plots")
    p.add_argument(
        "--ratio-ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Fixed ratio y-axis (min max); default auto-scale from data",
    )
    return p.parse_args()


def load_merge(gc_path: Path, si_path: Path) -> pd.DataFrame:
    gc = pd.read_csv(gc_path)
    si = pd.read_csv(si_path)
    g = gc[KEY + ["SeqClose"]].rename(columns={"SeqClose": "close_gc"})
    s = si[KEY + ["SeqClose"]].rename(columns={"SeqClose": "close_si"})
    m = g.merge(s, on=KEY, how="inner")
    if len(m) < len(g):
        print(f"merge: inner join rows={len(m)} (gc rows={len(g)}; unmatched gc bars dropped={len(g) - len(m)})")
    m["ts"] = pd.to_datetime(
        dict(year=m["Year"], month=m["Month"], day=m["Day"])
    ) + pd.to_timedelta(m["HourOfDay"], unit="h") + pd.to_timedelta(m["MinOfHour"], unit="m")
    m = m.sort_values("ts").reset_index(drop=True)
    m["cal_day"] = m["ts"].dt.date
    m["ratio"] = m["close_gc"] / m["close_si"]
    return m


def _daily_pearson_returns(group: pd.DataFrame) -> float:
    r_gc = group["close_gc"].pct_change()
    r_si = group["close_si"].pct_change()
    df = pd.DataFrame({"rg": r_gc, "rs": r_si}).dropna()
    if len(df) < MIN_BARS_PER_DAY:
        return np.nan
    return float(df["rg"].corr(df["rs"]))


def compute_daily_correlations(merged: pd.DataFrame) -> pd.Series:
    rows: list[tuple[date, float]] = []
    for day, g in merged.groupby("cal_day", sort=True):
        rows.append((day, _daily_pearson_returns(g)))
    s = pd.Series({d: v for d, v in rows}, dtype=float)
    s.name = "daily_corr"
    return s.sort_index()


def prior_trading_days(all_days: list[date], chosen: date, n: int) -> list[date]:
    before = [d for d in all_days if d < chosen]
    return before[-n:] if len(before) > n else before


def prior_days_ytd(all_days: list[date], chosen: date, year: int) -> list[date]:
    start = date(year, 1, 1)
    return [d for d in all_days if start <= d < chosen]


def apply_dark_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#1a1a1a",
            "axes.facecolor": "#1a1a1a",
            "axes.edgecolor": "#666666",
            "axes.labelcolor": "#e0e0e0",
            "text.color": "#e0e0e0",
            "xtick.color": "#cccccc",
            "ytick.color": "#cccccc",
            "grid.color": "#444444",
            "grid.alpha": 0.5,
        }
    )


def _apply_ratio_ylim(ax, ratio_series: pd.Series, ratio_ylim: tuple[float, float] | None) -> None:
    if ratio_ylim is not None:
        ax.set_ylim(ratio_ylim[0], ratio_ylim[1])
        return
    lo, hi = float(ratio_series.min()), float(ratio_series.max())
    pad = max((hi - lo) * 0.06, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)


def _format_intraday_time_axis(ax) -> None:
    """X-axis as HH:MM only; hide matplotlib date offset text next to ticks."""
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Time")


def _corr_ylim_from_hist(
    hist_vals: np.ndarray, mu: float, sigma: float, today_corr: float
) -> tuple[float, float]:
    corr_min = float(np.min(hist_vals)) if len(hist_vals) else today_corr
    corr_max = float(np.max(hist_vals)) if len(hist_vals) else today_corr
    corr_min = min(corr_min, mu - 3 * sigma, today_corr)
    corr_max = max(corr_max, mu + 3 * sigma, today_corr)
    pad = max((corr_max - corr_min) * 0.06, 1e-4)
    return corr_min - pad, corr_max + pad


def _rolling_intraday_corr(day_df: pd.DataFrame) -> pd.Series:
    rg = day_df["close_gc"].pct_change()
    rs = day_df["close_si"].pct_change()
    return rg.rolling(ROLLING_CORR_BARS, min_periods=MIN_BARS_PER_DAY).corr(rs)


def _gaussian_kde_density(samples: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Lightweight Gaussian KDE (no scipy dependency).

    Uses a Gaussian kernel with bandwidth via Silverman's rule of thumb.
    Returns density values for each `points[i]`.
    """
    x = np.asarray(samples, dtype=float)
    t = np.asarray(points, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros_like(t)
    if n == 1:
        # Degenerate: treat as a narrow peak; density is effectively ~0 for plotting.
        return np.zeros_like(t)

    std = float(np.std(x, ddof=1))
    if std < 1e-12:
        return np.zeros_like(t)

    # Silverman's rule of thumb
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    h = max(h, 1e-12)
    z = (t[:, None] - x[None, :]) / h
    density = np.exp(-0.5 * z**2).sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi))
    return density


def fig1_overlay(day_df: pd.DataFrame, out_path: Path, day_str: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(day_df["ts"], day_df["close_gc"], label="GC close (SeqClose)", color="#ffd700", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("GC close", color="#ffd700")
    ax.tick_params(axis="y", labelcolor="#ffd700")
    ax.spines["left"].set_color("#ffd700")
    ax.grid(True, alpha=0.35)

    ax2 = ax.twinx()
    ax2.plot(day_df["ts"], day_df["close_si"], label="SI close (SeqClose)", color="#c0c0c0", linewidth=1.2)
    ax2.set_ylabel("SI close", color="#c0c0c0")
    ax2.tick_params(axis="y", labelcolor="#c0c0c0")
    ax2.spines["right"].set_color("#c0c0c0")

    ax.set_title(f"Gold & silver — raw SeqClose (dual axis) — {day_str}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_ratio(
    day_df: pd.DataFrame, out_path: Path, day_str: str, ratio_ylim: tuple[float, float] | None
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(day_df["ts"], day_df["ratio"], color="#6ec1ff", linewidth=1.2)
    _apply_ratio_ylim(ax, day_df["ratio"], ratio_ylim)
    ax.set_title(f"GC/SI ratio (intraday) — {day_str}")
    ax.set_ylabel("Ratio")
    ax.grid(True)
    _format_intraday_time_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sigma_line(ax, x: float, color: str, linestyle: str, lw: float) -> None:
    ax.axvline(x, color=color, linestyle=linestyle, linewidth=lw, alpha=0.85)


def _label_at_top(ax, x: float, text: str, color: str, y_axes: float) -> None:
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x, y_axes, text, transform=trans, ha="center", va="bottom", fontsize=7, color=color, clip_on=False)


def _bellcurve_axes(
    ax,
    hist_vals: np.ndarray,
    mu: float,
    sigma: float,
    today_corr: float,
    z: float,
    day_str: str,
    lookback_used: int,
    lookback_caption: str,
    *,
    hist_legend_label: str = "Day count",
    y_label: str = "Trading days (count)",
    x_label: str = "Daily Pearson corr (GC vs SI intraday returns)",
    title_prefix: str = "Historical daily correlation vs",
    title_count_unit: str = "days",
) -> None:
    n_bins = min(40, max(10, len(hist_vals) // 5))
    ax.hist(
        hist_vals,
        bins=n_bins,
        density=False,
        alpha=0.55,
        color="#4a90d9",
        edgecolor="#333",
        label=hist_legend_label,
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    ax2 = ax.twinx()
    ax2.set_ylabel("Density (KDE)", color="#ff9f43")
    ax2.tick_params(axis="y", labelcolor="#ff9f43")
    ax2.spines["right"].set_color("#ff9f43")
    if len(hist_vals) >= 5:
        x_min = float(np.min(hist_vals))
        x_max = float(np.max(hist_vals))
        x_pad = max((x_max - x_min) * 0.08, 1e-4)
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 400)
        kde_density = _gaussian_kde_density(hist_vals, x_grid)
        if np.max(kde_density) > 0:
            ax2.plot(x_grid, kde_density, color="#ff9f43", linewidth=1.8, label="KDE")
    ax2.grid(False)

    colors_sigma = ["#888888", "#aaaaaa", "#cccccc"]
    for k, c in zip([1, 2, 3], colors_sigma):
        _sigma_line(ax, mu - k * sigma, c, "--", 1)
        _sigma_line(ax, mu + k * sigma, c, "--", 1)
    _sigma_line(ax, mu, "#ffffff", "-", 1.2)
    _sigma_line(ax, today_corr, "#ff6b6b", "-", 2.2)

    _label_at_top(ax, mu, "Mean μ", "#ffffff", 0.995)
    for k, c in zip([1, 2, 3], colors_sigma):
        _label_at_top(ax, mu - k * sigma, f"−{k}σ", c, 0.995)
        _label_at_top(ax, mu + k * sigma, f"+{k}σ", c, 0.995)
    _label_at_top(ax, today_corr, f"Today's corr\n{today_corr:.3f}", "#ff6b6b", 1.035)

    ax.set_title(
        f"{title_prefix} {day_str}\n{lookback_caption} (n={lookback_used} {title_count_unit})",
        pad=36,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    dev = today_corr - mu
    text = (
        f"μ = {mu:.6f}\n"
        f"σ = {sigma:.6f}\n"
        f"Today's corr = {today_corr:.6f}\n"
        f"Deviation = {dev:+.6f}\n"
        f"z-score = {z:+.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#2a2a2a", edgecolor="#555", alpha=0.95),
    )


def fig3_bellcurve(
    hist_vals: np.ndarray,
    mu: float,
    sigma: float,
    today_corr: float,
    z: float,
    out_path: Path,
    day_str: str,
    lookback_used: int,
    lookback_caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _bellcurve_axes(ax, hist_vals, mu, sigma, today_corr, z, day_str, lookback_used, lookback_caption)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_composite(
    day_df: pd.DataFrame,
    hist_vals: np.ndarray,
    mu: float,
    sigma: float,
    today_corr: float,
    z: float,
    out_path: Path,
    day_str: str,
    lookback_used: int,
    lookback_caption: str,
) -> None:
    """
    One shared y-axis (correlation): rolling intraday correlation vs time (bottom x).
    twiny() top x = correlation; KDE of historical daily correlations with density
    normalized to [0,1] and mapped into the same y-limits (shape only, not density units).
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    roll = _rolling_intraday_corr(day_df)
    ax.plot(
        day_df["ts"],
        roll,
        color="#ff9f43",
        linewidth=1.8,
        label=f"Rolling corr ({ROLLING_CORR_BARS} bars)",
        zorder=3,
    )
    ax.set_ylabel("Correlation", color="#ff9f43")
    ax.tick_params(axis="y", labelcolor="#ff9f43")
    ax.spines["left"].set_color("#ff9f43")
    ax.grid(True, alpha=0.35)
    _format_intraday_time_axis(ax)

    y_lo, y_hi = _corr_ylim_from_hist(hist_vals, mu, sigma, today_corr)
    ax.set_ylim(y_lo, y_hi)

    colors_sigma = ["#666666", "#888888", "#aaaaaa"]
    for k, c in zip([1, 2, 3], colors_sigma):
        ax.axhline(mu - k * sigma, color=c, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
        ax.axhline(mu + k * sigma, color=c, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
    ax.axhline(
        mu,
        color="#ffffff",
        linestyle="-",
        linewidth=1.2,
        alpha=0.95,
        label="Mean μ",
        zorder=2,
    )
    ax.axhline(
        today_corr,
        color="#ff6b6b",
        linestyle="-",
        linewidth=1.6,
        alpha=0.95,
        label="Today's daily corr",
        zorder=2,
    )

    # KDE on correlation grid; map normalized density into a vertical band of shared y-limits
    span = y_hi - y_lo
    kde_band = 0.62
    band_base = y_lo + 0.12 * span
    corr_grid = np.linspace(y_lo, y_hi, 400)
    kde_density = _gaussian_kde_density(hist_vals, corr_grid)
    d_max = float(np.max(kde_density)) if kde_density.size else 0.0
    if d_max > 0:
        d_norm = kde_density / d_max
    else:
        d_norm = np.zeros_like(kde_density)
    y_kde = band_base + d_norm * span * kde_band

    ax2 = ax.twiny()
    ax2.plot(
        corr_grid,
        y_kde,
        color="#7fe38b",
        linewidth=1.7,
        alpha=0.92,
        label="KDE (hist. daily corr; height scaled to panel)",
        zorder=2,
    )
    ax2.set_xlim(y_lo, y_hi)
    ax2.tick_params(axis="x", colors="#7fe38b", labelsize=9)
    ax2.spines["top"].set_color("#7fe38b")
    ax2.set_xlabel("Daily correlation (historical)", color="#7fe38b", labelpad=8)

    ax.set_title(
        f"Rolling intraday correlation (time) + daily-corr KDE (top axis, y = relative shape) — {day_str}",
        pad=14,
        fontsize=11,
    )

    dev = today_corr - mu
    ax.text(
        0.99,
        0.98,
        f"today corr = {today_corr:.6f}\nμ={mu:.6f} σ={sigma:.6f}\nDeviation={dev:+.6f}\nz={z:+.2f}\n"
        f"{lookback_caption} (n={lookback_used})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#2a2a2a", edgecolor="#555", alpha=0.95),
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    chosen = datetime.strptime(args.day, "%Y-%m-%d").date()
    ratio_ylim = tuple(args.ratio_ylim) if args.ratio_ylim is not None else None

    apply_dark_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merge(args.gc, args.si)
    daily_corr = compute_daily_correlations(merged)

    all_days = sorted(merged["cal_day"].unique())
    day_df = merged[merged["cal_day"] == chosen].copy()
    if len(day_df) < MIN_BARS_PER_DAY:
        raise SystemExit(f"--day {chosen}: need at least {MIN_BARS_PER_DAY} bars, got {len(day_df)}")

    day_str = chosen.isoformat()

    fig1_overlay(day_df, args.out_dir / f"01_gc_si_overlay_{day_str}.png", day_str)
    fig2_ratio(day_df, args.out_dir / f"02_ratio_intraday_{day_str}.png", day_str, ratio_ylim)

    today_corr = daily_corr.get(chosen, np.nan)
    if np.isnan(today_corr):
        raise SystemExit(f"Could not compute daily correlation for {chosen} (insufficient returns?)")

    if args.lookback_days is not None:
        prior_days = prior_trading_days(all_days, chosen, args.lookback_days)
        lookback_caption = f"Last {args.lookback_days} trading days before {day_str}"
    else:
        y0 = date(args.baseline_year, 1, 1)
        if chosen < y0:
            raise SystemExit(
                f"--day {chosen} is before {y0}; no {args.baseline_year} YTD baseline. "
                "Use --lookback-days N or a later --day."
            )
        prior_days = prior_days_ytd(all_days, chosen, args.baseline_year)
        lookback_caption = f"{args.baseline_year} YTD (Jan 1 → day before {day_str})"

    hist_series = daily_corr.reindex(prior_days).dropna()
    hist_vals = hist_series.values

    if len(hist_vals) < 2:
        raise SystemExit(
            f"Need at least 2 prior days with valid correlation for bell curve; got {len(hist_vals)}. "
            f"Window: {lookback_caption}. Try --lookback-days with a larger N."
        )

    mu = float(np.mean(hist_vals))
    sigma = float(np.std(hist_vals, ddof=1))
    if sigma < 1e-12:
        z = 0.0 if abs(today_corr - mu) < 1e-12 else float("inf")
    else:
        z = (today_corr - mu) / sigma

    lookback_used = len(hist_vals)

    fig3_bellcurve(
        hist_vals,
        mu,
        sigma,
        today_corr,
        z,
        args.out_dir / f"03_corr_bellcurve_{day_str}.png",
        day_str,
        lookback_used,
        lookback_caption,
    )

    fig4_composite(
        day_df,
        hist_vals,
        mu,
        sigma,
        today_corr,
        z,
        args.out_dir / f"04_ratio_and_bellcurve_{day_str}.png",
        day_str,
        lookback_used,
        lookback_caption,
    )

    print(f"Wrote figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
