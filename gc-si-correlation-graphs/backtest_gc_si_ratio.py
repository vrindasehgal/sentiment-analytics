#!/usr/bin/env python3
"""
Backtest a GC strategy driven by GC/SI ratio +/- 2 sigma signals.

Signal baseline:
- For each ISO week, mu/sigma are computed from the prior ISO week's
  intraday ratio bars only (Mon-Fri).

Strategy:
- Short 1 GC when ratio > mu + 2*sigma
- Long 1 GC when ratio < mu - 2*sigma
- Exit when PnL >= +2% account value or <= -0.5% account value
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from plot_gc_si import load_merge

GC_MULTIPLIER = 100.0
DEFAULT_ACCOUNT_VALUE = 100_000.0
PROFIT_TARGET_PCT = 0.02
STOP_LOSS_PCT = 0.005
@dataclass
class Position:
    side: int  # +1 long, -1 short
    entry_ts: pd.Timestamp
    entry_gc: float
    entry_si: float


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    return here, data_dir


def _iso_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _weekly_mean_ratio_series(merged: pd.DataFrame) -> pd.Series:
    tmp = merged[["cal_day", "ratio"]].copy()
    tmp["week_start"] = tmp["cal_day"].map(_iso_week_start)
    return tmp.groupby("week_start", sort=True)["ratio"].mean().astype(float)


def parse_args() -> argparse.Namespace:
    root, data_dir = _default_paths()
    p = argparse.ArgumentParser(description="Backtest GC/SI ratio +/-2 sigma strategy.")
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument("--out-dir", type=Path, default=root / "plots")
    p.add_argument("--account-value", type=float, default=DEFAULT_ACCOUNT_VALUE)
    return p.parse_args()


def attach_baseline(merged: pd.DataFrame) -> pd.DataFrame:
    weekdays = merged[merged["cal_day"].map(lambda d: d.weekday() <= 4)].copy()
    weekdays["week_start"] = weekdays["cal_day"].map(_iso_week_start)
    baseline_map: dict[date, tuple[float, float]] = {}

    ratio_by_week: dict[date, np.ndarray] = {}
    for wk, g in weekdays.groupby("week_start", sort=True):
        ratio_by_week[wk] = g["ratio"].astype(float).values

    week_starts = sorted(ratio_by_week.keys())
    for i in range(1, len(week_starts)):
        wk = week_starts[i]
        prev_wk = week_starts[i - 1]
        prev_vals = ratio_by_week[prev_wk]
        if len(prev_vals) >= 2:
            mu = float(np.mean(prev_vals))
            sigma = float(np.std(prev_vals, ddof=1))
            baseline_map[wk] = (mu, sigma)

    weekdays["ratio_mu"] = weekdays["week_start"].map(
        lambda w: baseline_map.get(w, (np.nan, np.nan))[0]
    )
    weekdays["ratio_sigma"] = weekdays["week_start"].map(
        lambda w: baseline_map.get(w, (np.nan, np.nan))[1]
    )
    return weekdays.sort_values("ts").reset_index(drop=True)


def backtest(df: pd.DataFrame, account_value: float) -> pd.DataFrame:
    profit_target = account_value * PROFIT_TARGET_PCT
    stop_loss = account_value * STOP_LOSS_PCT

    trades: list[dict[str, object]] = []
    pos: Position | None = None

    for row in df.itertuples(index=False):
        mu = float(row.ratio_mu)
        sigma = float(row.ratio_sigma)
        ratio = float(row.ratio)
        gc_price = float(row.close_gc)
        si_price = float(row.close_si)
        ts = pd.Timestamp(row.ts)

        if np.isnan(mu) or np.isnan(sigma) or sigma <= 0.0:
            continue

        upper = mu + 2.0 * sigma
        lower = mu - 2.0 * sigma

        if pos is None:
            if ratio > upper:
                pos = Position(side=-1, entry_ts=ts, entry_gc=gc_price, entry_si=si_price)
            elif ratio < lower:
                pos = Position(side=1, entry_ts=ts, entry_gc=gc_price, entry_si=si_price)
            continue

        pnl = (gc_price - pos.entry_gc) * GC_MULTIPLIER * pos.side

        if pnl >= profit_target or pnl <= -stop_loss:
            trades.append(
                {
                    "Date of Trade": pos.entry_ts,
                    "GC price": pos.entry_gc,
                    "Si Price": pos.entry_si,
                    "Date of Close": ts,
                    "Profit or Loss": round(pnl, 2),
                }
            )
            pos = None

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
        }

    pnl = trades["Profit or Loss"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak

    return {
        "trades": float(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "net_pnl": float(pnl.sum()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merge(args.gc, args.si)
    model_df = attach_baseline(merged)
    trades = backtest(model_df, account_value=float(args.account_value))

    out_csv = args.out_dir / "gc_si_ratio_backtest_results.csv"
    out_summary_csv = args.out_dir / "gc_si_ratio_backtest_summary.csv"
    trades.to_csv(out_csv, index=False)

    stats = summarize(trades)
    closing_account_value = float(args.account_value) + stats["net_pnl"]
    summary_row = pd.DataFrame(
        [
            {
                "StartingAccount": float(args.account_value),
                "Trades": int(stats["trades"]),
                "WinRate": stats["win_rate"],
                "NetPnL": stats["net_pnl"],
                "AvgWin": stats["avg_win"],
                "AvgLoss": stats["avg_loss"],
                "MaxDrawdown": stats["max_drawdown"],
                "ClosingAccount": closing_account_value,
            }
        ]
    )
    summary_row.to_csv(out_summary_csv, index=False)
    print(f"Wrote trade results: {out_csv.resolve()}")
    print(f"Wrote summary results: {out_summary_csv.resolve()}")
    print(
        "Summary | "
        f"Trades={int(stats['trades'])} | "
        f"WinRate={stats['win_rate']:.2%} | "
        f"NetPnL=${stats['net_pnl']:,.2f} | "
        f"AvgWin=${stats['avg_win']:,.2f} | "
        f"AvgLoss=${stats['avg_loss']:,.2f} | "
        f"MaxDrawdown=${stats['max_drawdown']:,.2f} | "
        f"ClosingAccount=${closing_account_value:,.2f}"
    )


if __name__ == "__main__":
    main()
