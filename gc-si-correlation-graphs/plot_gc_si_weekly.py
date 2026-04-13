#!/usr/bin/env python3
"""
Weekly GC-SI charts (Mon-Fri ISO week) from merged 15m bars.

This script is additive and does not change daily outputs.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_gc_si import (
    MIN_BARS_PER_DAY,
    ROLLING_CORR_BARS,
    _apply_ratio_ylim,
    _bellcurve_axes,
    _corr_ylim_from_hist,
    _daily_pearson_returns,
    _gaussian_kde_density,
    _rolling_intraday_corr,
    _sigma_line,
    apply_dark_style,
    load_merge,
)


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    return here, data_dir


def parse_args() -> argparse.Namespace:
    root, data_dir = _default_paths()
    p = argparse.ArgumentParser(description="Plot weekly GC/SI figures for ISO Mon-Fri window.")
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument("--day", type=str, required=True, help="Anchor day YYYY-MM-DD (selects containing ISO week)")
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


def iso_week_bounds(chosen: date) -> tuple[date, date]:
    monday = chosen - timedelta(days=chosen.weekday())
    friday = monday + timedelta(days=4)
    return monday, friday


def _week_tag(week_start: date, week_end: date) -> str:
    iso_year, iso_week, _ = week_start.isocalendar()
    return f"{iso_year}-W{iso_week:02d}_{week_start:%m-%d}_to_{week_end:%m-%d}"


def _format_weekday_axis(ax, week_df: pd.DataFrame) -> None:
    first_bars = week_df.sort_values("ts").groupby("cal_day", sort=True).head(1)
    ticks = list(first_bars["ts"])
    labels = [d.strftime("%a") for d in first_bars["cal_day"]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Day")


def _week_bar_index_and_ticks(week_df: pd.DataFrame) -> tuple[np.ndarray, list[int], list[str]]:
    ordered = week_df.sort_values("ts")
    x = np.arange(len(ordered), dtype=float)
    first_bars = ordered.groupby("cal_day", sort=True).head(1)
    tick_positions = [int(i) for i in first_bars.index]
    tick_labels = [d.strftime("%a") for d in first_bars["cal_day"]]
    return x, tick_positions, tick_labels


def _render_info_figure(out_path: Path, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#2a2a2a", edgecolor="#555", alpha=0.95),
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1_overlay_weekly(week_df: pd.DataFrame, out_path: Path, week_label: str) -> None:
    ordered = week_df.sort_values("ts").reset_index(drop=True)
    x, tick_positions, tick_labels = _week_bar_index_and_ticks(ordered)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, ordered["close_gc"], label="GC close (SeqClose)", color="#ffd700", linewidth=1.2)
    ax.set_ylabel("GC close", color="#ffd700")
    ax.tick_params(axis="y", labelcolor="#ffd700")
    ax.spines["left"].set_color("#ffd700")
    ax.grid(True, alpha=0.35)

    ax2 = ax.twinx()
    ax2.plot(x, ordered["close_si"], label="SI close (SeqClose)", color="#c0c0c0", linewidth=1.2)
    ax2.set_ylabel("SI close", color="#c0c0c0")
    ax2.tick_params(axis="y", labelcolor="#c0c0c0")
    ax2.spines["right"].set_color("#c0c0c0")

    ax.set_title(f"Gold & silver — raw SeqClose (dual axis) — Weekly {week_label}")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Day")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_ratio_weekly(
    week_df: pd.DataFrame, out_path: Path, week_label: str, ratio_ylim: tuple[float, float] | None
) -> None:
    ordered = week_df.sort_values("ts").reset_index(drop=True)
    x, tick_positions, tick_labels = _week_bar_index_and_ticks(ordered)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, ordered["ratio"], color="#6ec1ff", linewidth=1.2)
    _apply_ratio_ylim(ax, ordered["ratio"], ratio_ylim)
    ax.set_title(f"GC/SI ratio (intraday) — Weekly {week_label}")
    ax.set_ylabel("Ratio")
    ax.grid(True)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_bellcurve_weekly(
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


def fig4_composite_weekly(
    week_df: pd.DataFrame,
    hist_vals: np.ndarray,
    mu: float,
    sigma: float,
    today_corr: float,
    z: float,
    out_path: Path,
    week_label: str,
    lookback_used: int,
    lookback_caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    roll = _rolling_intraday_corr(week_df)
    ax.plot(
        week_df["ts"],
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
    _format_weekday_axis(ax, week_df)

    y_lo, y_hi = _corr_ylim_from_hist(hist_vals, mu, sigma, today_corr)
    ax.set_ylim(y_lo, y_hi)

    colors_sigma = ["#666666", "#888888", "#aaaaaa"]
    for k, c in zip([1, 2, 3], colors_sigma):
        ax.axhline(mu - k * sigma, color=c, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
        ax.axhline(mu + k * sigma, color=c, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
    ax.axhline(mu, color="#ffffff", linestyle="-", linewidth=1.2, alpha=0.95, label="Mean μ", zorder=2)
    ax.axhline(
        today_corr,
        color="#ff6b6b",
        linestyle="-",
        linewidth=1.6,
        alpha=0.95,
        label="Today's daily corr",
        zorder=2,
    )

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
        label="KDE (other weekdays; height scaled to panel)",
        zorder=2,
    )
    ax2.set_xlim(y_lo, y_hi)
    ax2.tick_params(axis="x", colors="#7fe38b", labelsize=9)
    ax2.spines["top"].set_color("#7fe38b")
    ax2.set_xlabel("Daily correlation (historical)", color="#7fe38b", labelpad=8)

    ax.set_title(
        f"Rolling intraday correlation (time) + daily-corr KDE (top axis) — Weekly {week_label}",
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


def _safe_stats(
    chosen: date,
    week_days: list[date],
    week_df: pd.DataFrame,
    daily_corr: pd.Series,
) -> tuple[float, np.ndarray, float, float, float, str]:
    today_corr = daily_corr.get(chosen, np.nan)
    if np.isnan(today_corr):
        today_corr = 0.0
        today_note = (
            f"Could not compute chosen-day corr for {chosen.isoformat()} "
            f"(need >= {MIN_BARS_PER_DAY} return bars)."
        )
    else:
        today_note = ""

    other_days = [d for d in week_days if d != chosen]
    intraday_samples: list[np.ndarray] = []
    for d in other_days:
        g = week_df[week_df["cal_day"] == d].sort_values("ts")
        if len(g) < MIN_BARS_PER_DAY:
            continue
        roll = _rolling_intraday_corr(g).dropna()
        if len(roll) > 0:
            intraday_samples.append(roll.values.astype(float))
    if intraday_samples:
        hist_vals = np.concatenate(intraday_samples)
    else:
        hist_vals = np.array([], dtype=float)

    if len(hist_vals) == 0:
        hist_vals = np.array([today_corr, today_corr], dtype=float)
        lookback_note = "No valid 15m sequence samples from other weekdays; fallback uses chosen-day corr only."
    elif len(hist_vals) == 1:
        hist_vals = np.array([hist_vals[0], hist_vals[0]], dtype=float)
        lookback_note = "Only one valid 15m sequence sample from other weekdays; duplicated for stable rendering."
    else:
        lookback_note = ""

    mu = float(np.mean(hist_vals))
    sigma = float(np.std(hist_vals, ddof=1)) if len(hist_vals) > 1 else 0.0
    if sigma < 1e-12:
        z = 0.0 if abs(today_corr - mu) < 1e-12 else float("inf")
    else:
        z = (today_corr - mu) / sigma

    notes = [n for n in [today_note, lookback_note] if n]
    return today_corr, hist_vals, mu, sigma, z, " ".join(notes)


def main() -> None:
    args = parse_args()
    chosen = datetime.strptime(args.day, "%Y-%m-%d").date()
    ratio_ylim = tuple(args.ratio_ylim) if args.ratio_ylim is not None else None

    apply_dark_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merge(args.gc, args.si)
    monday, friday = iso_week_bounds(chosen)
    week_days = [monday + timedelta(days=i) for i in range(5)]
    week_tag = _week_tag(monday, friday)
    week_label = f"{week_tag} ({monday.isoformat()} to {friday.isoformat()})"
    week_df = merged[merged["cal_day"].isin(week_days)].copy()

    out1 = args.out_dir / f"01_gc_si_overlay_weekly_{week_tag}.png"
    out2 = args.out_dir / f"02_ratio_intraday_weekly_{week_tag}.png"
    out3 = args.out_dir / f"03_corr_bellcurve_weekly_{week_tag}.png"
    out4 = args.out_dir / f"04_ratio_and_bellcurve_weekly_{week_tag}.png"

    if week_df.empty:
        msg = [
            f"No aligned GC/SI data found for ISO week {week_tag}.",
            "Figures are placeholders so the run still completes.",
        ]
        _render_info_figure(out1, f"Weekly GC/SI overlay — {week_tag}", msg)
        _render_info_figure(out2, f"Weekly GC/SI ratio — {week_tag}", msg)
        _render_info_figure(out3, f"Weekly correlation bell curve — {week_tag}", msg)
        _render_info_figure(out4, f"Weekly rolling corr + KDE — {week_tag}", msg)
        print(f"Wrote weekly placeholder figures to {args.out_dir.resolve()}")
        return

    fig1_overlay_weekly(week_df, out1, week_label)
    fig2_ratio_weekly(week_df, out2, week_label, ratio_ylim)

    daily_corr = pd.Series(
        {
            d: _daily_pearson_returns(g)
            for d, g in week_df.groupby("cal_day", sort=True)
        },
        dtype=float,
    ).sort_index()

    today_corr, hist_vals, mu, sigma, z, notes = _safe_stats(chosen, week_days, week_df, daily_corr)
    lookback_caption = (
        "15m rolling-correlation samples from other weekdays "
        f"in same ISO week ({monday.isoformat()} to {friday.isoformat()})"
    )
    if notes:
        lookback_caption = f"{lookback_caption}. {notes}"
    lookback_used = len(hist_vals)

    fig3_bellcurve_weekly(
        hist_vals,
        mu,
        sigma,
        today_corr,
        z,
        out3,
        chosen.isoformat(),
        lookback_used,
        lookback_caption,
    )

    fig4_composite_weekly(
        week_df,
        hist_vals,
        mu,
        sigma,
        today_corr,
        z,
        out4,
        week_label,
        lookback_used,
        lookback_caption,
    )

    print(f"Wrote weekly figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
