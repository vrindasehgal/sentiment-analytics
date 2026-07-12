#!/usr/bin/env python3
"""Extended-delay sweep for GC/SI spread backtests (Phase 1)."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from backtest_gc_si_ratio_daily import DEFAULT_ACCOUNT_VALUE, aggregate_daily_closes, attach_baseline_daily
from backtest_gc_si_spread import (
    REWARD_RISK_PCTS,
    Mode,
    build_results_summary,
    run_spread_backtest,
    select_top_configs,
    summarize_by_month,
    summarize_by_year_with_params,
    summarize_sweep,
)
from plot_gc_si import load_merge

TRIGGER_SIGMAS = (1.5, 2.0)
PROBABLE_ENTRY_LAGS = (0, 1, 2, 3, 5)
PROBABLE_CALENDAR_N = (1, 2, 3, 5, 6, 7, 8, 10)
WAIT_ENTRY_LAGS = (1, 2, 3, 4, 5, 6)
WAIT_CALENDAR_N = (1, 2, 3, 4, 5, 6)


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    return here, data_dir


def _param_grid(mode: Mode) -> list[tuple[float, int, int]]:
    if mode == "probable_duration":
        return [
            (k, entry_lag, cal_n)
            for k, entry_lag, cal_n in product(
                TRIGGER_SIGMAS, PROBABLE_ENTRY_LAGS, PROBABLE_CALENDAR_N
            )
        ]
    return [
        (k, entry_lag, cal_n)
        for k, entry_lag, cal_n in product(TRIGGER_SIGMAS, WAIT_ENTRY_LAGS, WAIT_CALENDAR_N)
    ]


def parse_args() -> argparse.Namespace:
    root, data_dir = _default_paths()
    p = argparse.ArgumentParser(description="Extended-delay GC/SI spread backtest sweep (Phase 1).")
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "results" / "spread_backtest_extended",
    )
    p.add_argument("--account-value", type=float, default=DEFAULT_ACCOUNT_VALUE)
    p.add_argument(
        "--mode",
        choices=("probable_duration", "wait_entry", "both"),
        default="both",
    )
    p.add_argument("--top-n", type=int, default=3, help="Top timing configs per mode for Phase 2.")
    return p.parse_args()


def run_mode(
    mode: Mode,
    daily: pd.DataFrame,
    account_value: float,
    out_root: Path,
    top_n: int,
) -> None:
    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict[str, object]] = []
    all_trades: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    all_by_year: list[pd.DataFrame] = []
    all_by_month: list[pd.DataFrame] = []

    grid = _param_grid(mode)
    print(f"[{mode}] Running {len(grid) * len(REWARD_RISK_PCTS)} backtests...")

    for trigger_sigma, entry_lag, calendar_n in grid:
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
                account_value=account_value,
            )
            sweep_rows.append({**params, **stats})

            if not result.trades.empty:
                t = result.trades.copy()
                for k, v in params.items():
                    t[k] = v
                all_trades.append(t)

            if not result.skipped_signals.empty:
                all_skipped.append(result.skipped_signals)

            all_by_year.append(
                summarize_by_year_with_params(
                    result.trades,
                    result.equity_series,
                    result.daily_returns,
                    params,
                )
            )
            by_m = summarize_by_month(result.trades, params)
            if not by_m.empty:
                all_by_month.append(by_m)

    sweep_df = pd.DataFrame(sweep_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    by_year_df = pd.concat(all_by_year, ignore_index=True) if all_by_year else pd.DataFrame()
    by_month_df = pd.concat(all_by_month, ignore_index=True) if all_by_month else pd.DataFrame()

    top_df = select_top_configs(sweep_df, mode=mode, top_n=top_n)

    sweep_df.to_csv(out_dir / "sweep_summary.csv", index=False)
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    skipped_df.to_csv(out_dir / "skipped_signals.csv", index=False)
    by_year_df.to_csv(out_dir / "by_year.csv", index=False)
    by_month_df.to_csv(out_dir / "by_month.csv", index=False)
    top_df.to_csv(out_dir / "top_configs.csv", index=False)

    print(f"Wrote {out_dir / 'sweep_summary.csv'} ({len(sweep_df)} rows)")
    print(f"Wrote {out_dir / 'by_month.csv'} ({len(by_month_df)} rows)")
    print(f"Wrote {out_dir / 'top_configs.csv'} ({len(top_df)} rows)")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    for path, label in ((args.gc, "GC"), (args.si, "SI")):
        if not path.is_file():
            raise SystemExit(f"Missing {label} CSV: {path.resolve()}")

    merged = load_merge(args.gc, args.si)
    daily = attach_baseline_daily(aggregate_daily_closes(merged))
    account_value = float(args.account_value)

    modes: list[Mode] = (
        ["probable_duration", "wait_entry"] if args.mode == "both" else [args.mode]  # type: ignore[list-item]
    )
    for mode in modes:
        run_mode(mode, daily, account_value, args.out_dir, args.top_n)

    summary = build_results_summary(
        args.out_dir,
        adaptive_dir=None,
        reversion_path=root / "results" / "reversion_duration_backtest" / "gc_si_ratio_reversion_daily_summary.csv",
        account_value=account_value,
        month_patterns_dir=args.out_dir,
    )
    if not summary.empty:
        summary_path = args.out_dir / "results_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path.resolve()} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
