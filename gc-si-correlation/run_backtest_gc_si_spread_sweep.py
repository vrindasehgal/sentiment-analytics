#!/usr/bin/env python3
"""Sweep GC/SI spread backtests (probable_duration and wait_entry modes)."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from backtest_gc_si_ratio_daily import DEFAULT_ACCOUNT_VALUE, aggregate_daily_closes, attach_baseline_daily
from backtest_gc_si_spread import (
    REWARD_RISK_PCTS,
    Mode,
    run_spread_backtest,
    summarize_by_year_with_params,
    summarize_sweep,
)
from plot_gc_si import load_merge

TRIGGER_SIGMAS = (1.5, 2.0)
PROBABLE_DURATION_N = (1, 2, 3, 5)
WAIT_ENTRY_LAGS = (1, 2)
WAIT_COMPLEMENTARY_LAGS = (1, 2)


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    return here, data_dir


def _param_grid(mode: Mode) -> list[tuple[float, int, int]]:
    """(trigger_sigma, entry_lag, calendar_n) per run."""
    if mode == "probable_duration":
        return [(k, 0, n) for k, n in product(TRIGGER_SIGMAS, PROBABLE_DURATION_N)]
    return [
        (k, entry_lag, comp_lag)
        for k, entry_lag, comp_lag in product(
            TRIGGER_SIGMAS, WAIT_ENTRY_LAGS, WAIT_COMPLEMENTARY_LAGS
        )
    ]


def parse_args() -> argparse.Namespace:
    root, data_dir = _default_paths()
    p = argparse.ArgumentParser(description="Sweep GC/SI spread backtest parameter grids.")
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument("--out-dir", type=Path, default=root / "results" / "spread_backtest")
    p.add_argument("--account-value", type=float, default=DEFAULT_ACCOUNT_VALUE)
    p.add_argument(
        "--mode",
        choices=("probable_duration", "wait_entry"),
        required=True,
        help="Backtest A (probable_duration) or B (wait_entry).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.gc, "GC"), (args.si, "SI")):
        if not path.is_file():
            raise SystemExit(f"Missing {label} CSV: {path.resolve()}")

    mode: Mode = args.mode
    out_dir = args.out_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merge(args.gc, args.si)
    daily = attach_baseline_daily(aggregate_daily_closes(merged))
    account_value = float(args.account_value)

    sweep_rows: list[dict[str, object]] = []
    all_trades: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    all_by_year: list[pd.DataFrame] = []

    for trigger_sigma, entry_lag, calendar_n in _param_grid(mode):
        for reward_risk in REWARD_RISK_PCTS:
            params = {
                "TriggerSigma": trigger_sigma,
                "Mode": mode,
                "EntryLag": entry_lag,
                "CalendarN": calendar_n,
                "RewardRisk": reward_risk,
            }
            result = run_spread_backtest(
                daily,
                account_value,
                mode=mode,
                trigger_sigma=trigger_sigma,
                entry_lag=entry_lag,
                calendar_n=calendar_n,
                reward_risk=reward_risk,
            )

            stats = summarize_sweep(
                result.trades,
                result.skipped_signals,
                result.equity_series,
                result.daily_returns,
            )
            sweep_rows.append({**params, **stats})

            if not result.trades.empty:
                t = result.trades.copy()
                for k, v in params.items():
                    t[k] = v
                all_trades.append(t)

            if not result.skipped_signals.empty:
                all_skipped.append(result.skipped_signals)

            by_yr = summarize_by_year_with_params(
                result.trades,
                result.equity_series,
                result.daily_returns,
                params,
            )
            if not by_yr.empty:
                all_by_year.append(by_yr)

    sweep_df = pd.DataFrame(sweep_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    by_year_df = pd.concat(all_by_year, ignore_index=True) if all_by_year else pd.DataFrame()

    sweep_path = out_dir / "sweep_summary.csv"
    trades_path = out_dir / "trades.csv"
    skipped_path = out_dir / "skipped_signals.csv"
    by_year_path = out_dir / "by_year.csv"

    sweep_df.to_csv(sweep_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    by_year_df.to_csv(by_year_path, index=False)

    print(f"Wrote sweep summary: {sweep_path.resolve()}")
    print(f"Wrote trades: {trades_path.resolve()} ({len(trades_df)} rows)")
    print(f"Wrote skipped signals: {skipped_path.resolve()} ({len(skipped_df)} rows)")
    print(f"Wrote by-year: {by_year_path.resolve()}")

    if not sweep_df.empty:
        for k in sorted(sweep_df["TriggerSigma"].unique()):
            sub = sweep_df[sweep_df["TriggerSigma"] == k].sort_values("NetPnL", ascending=False)
            best = sub.iloc[0]
            print(
                f"  k={k:g} best NetPnL=${best['NetPnL']:,.0f} | "
                f"N={int(best['CalendarN'])} lag={int(best['EntryLag'])} "
                f"RR={best['RewardRisk']} | trades={int(best['Trades'])} "
                f"skipped={int(best['SignalsSkipped'])}"
            )


if __name__ == "__main__":
    main()
