#!/usr/bin/env python3
"""Adaptive risk backtest on extended-delay best configs (Phase 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_gc_si_ratio_daily import DEFAULT_ACCOUNT_VALUE, aggregate_daily_closes, attach_baseline_daily
from backtest_gc_si_spread import (
    REWARD_RISK_PCTS,
    DEFAULT_RISK_CONTROLS,
    Mode,
    build_results_summary,
    get_extended_delay_best_configs,
    print_results_summary_narrative,
    run_spread_backtest,
    summarize_by_month,
    summarize_sweep,
)
from plot_gc_si import load_merge


def _default_paths() -> tuple[Path, Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    results_root = here / "results" / "spread_delays_adaptive_risk"
    return here, data_dir, results_root


def parse_args() -> argparse.Namespace:
    _, data_dir, results_root = _default_paths()
    p = argparse.ArgumentParser(description="Adaptive GC/SI spread backtest on Phase-1 configs.")
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument(
        "--extended-dir",
        type=Path,
        default=results_root / "phase1_extended",
        help="Directory with Phase-1 sweep_summary.csv per mode.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=results_root / "phase2_adaptive",
    )
    p.add_argument(
        "--summary-out-dir",
        type=Path,
        default=results_root,
        help="Directory for consolidated results_summary.csv and month_patterns_summary.csv",
    )
    p.add_argument(
        "--reversion-path",
        type=Path,
        default=None,
        help="Reversion study summary CSV for median-duration alignment.",
    )
    p.add_argument("--account-value", type=float, default=DEFAULT_ACCOUNT_VALUE)
    p.add_argument(
        "--config-source",
        choices=("extended_delay_best", "top_configs"),
        default="extended_delay_best",
        help="Timing configs: best per trigger from full sweep (default) or top_configs.csv.",
    )
    p.add_argument(
        "--mode",
        choices=("probable_duration", "wait_entry", "both"),
        default="both",
    )
    return p.parse_args()


def _load_timing_configs(
    mode: Mode,
    extended_dir: Path,
    config_source: str,
) -> pd.DataFrame:
    if config_source == "top_configs":
        top_path = extended_dir / mode / "top_configs.csv"
        if not top_path.is_file():
            raise SystemExit(f"Missing {top_path}. Run Phase 1 extended sweep first.")
        configs = pd.read_csv(top_path)
    else:
        sweep_path = extended_dir / mode / "sweep_summary.csv"
        if not sweep_path.is_file():
            raise SystemExit(f"Missing {sweep_path}. Run Phase 1 extended sweep first.")
        sweep = pd.read_csv(sweep_path)
        configs = get_extended_delay_best_configs(sweep, mode)

    if configs.empty:
        raise SystemExit(f"No timing configs for {mode} from {config_source}")

    timing_cols = ["TriggerSigma", "EntryLag", "CalendarN"]
    return configs.drop_duplicates(subset=timing_cols).reset_index(drop=True)


def run_mode(
    mode: Mode,
    daily: pd.DataFrame,
    account_value: float,
    extended_dir: Path,
    out_root: Path,
    config_source: str,
) -> None:
    unique_configs = _load_timing_configs(mode, extended_dir, config_source)

    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict[str, object]] = []
    all_trades: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    all_by_month: list[pd.DataFrame] = []

    print(
        f"[{mode}] {len(unique_configs)} timing configs x {len(REWARD_RISK_PCTS)} RR "
        f"x 2 (baseline/adaptive) [{config_source}]"
    )

    for _, cfg in unique_configs.iterrows():
        trigger_sigma = float(cfg["TriggerSigma"])
        entry_lag = int(cfg["EntryLag"])
        calendar_n = int(cfg["CalendarN"])

        for reward_risk in REWARD_RISK_PCTS:
            base_params = {
                "TriggerSigma": trigger_sigma,
                "Mode": mode,
                "EntryLag": entry_lag,
                "CalendarN": calendar_n,
                "RewardRisk": reward_risk,
            }

            for variant, risk in (("baseline", None), ("adaptive", DEFAULT_RISK_CONTROLS)):
                result = run_spread_backtest(
                    daily,
                    account_value,
                    mode=mode,
                    trigger_sigma=trigger_sigma,
                    entry_lag=entry_lag,
                    calendar_n=calendar_n,
                    reward_risk=reward_risk,
                    risk_controls=risk,
                )
                stats = summarize_sweep(
                    result.trades,
                    result.skipped_signals,
                    result.equity_series,
                    result.daily_returns,
                    account_value=account_value,
                )
                sweep_rows.append({**base_params, "Variant": variant, **stats})

                if not result.trades.empty:
                    t = result.trades.copy()
                    t["Variant"] = variant
                    for k, v in base_params.items():
                        t[k] = v
                    all_trades.append(t)

                if not result.skipped_signals.empty:
                    s = result.skipped_signals.copy()
                    s["Variant"] = variant
                    all_skipped.append(s)

                by_m = summarize_by_month(result.trades, {**base_params, "Variant": variant})
                if not by_m.empty:
                    all_by_month.append(by_m)

    sweep_df = pd.DataFrame(sweep_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    by_month_df = pd.concat(all_by_month, ignore_index=True) if all_by_month else pd.DataFrame()

    sweep_df.to_csv(out_dir / "sweep_summary_adaptive.csv", index=False)
    trades_df.to_csv(out_dir / "trades_adaptive.csv", index=False)
    skipped_df.to_csv(out_dir / "skipped_signals_adaptive.csv", index=False)
    by_month_df.to_csv(out_dir / "by_month_adaptive.csv", index=False)

    print(f"Wrote {out_dir / 'sweep_summary_adaptive.csv'} ({len(sweep_df)} rows)")

    if not sweep_df.empty:
        for variant in ("baseline", "adaptive"):
            sub = sweep_df[sweep_df["Variant"] == variant]
            if sub.empty:
                continue
            best = sub.sort_values("Sharpe", ascending=False).iloc[0]
            print(
                f"  {variant}: best Sharpe={best['Sharpe']:.2f} "
                f"NetPnL=${best['NetPnL']:,.0f} MDD=${best['MaxDrawdown']:,.0f} "
                f"k={best['TriggerSigma']} N={int(best['CalendarN'])} lag={int(best['EntryLag'])}"
            )


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    reversion_path = args.reversion_path or (
        here / "results" / "reversion_duration_backtest" / "gc_si_ratio_reversion_daily_summary.csv"
    )

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
        run_mode(
            mode,
            daily,
            account_value,
            args.extended_dir,
            args.out_dir,
            args.config_source,
        )

    summary = build_results_summary(
        args.extended_dir,
        adaptive_dir=args.out_dir,
        reversion_path=reversion_path,
        account_value=account_value,
        month_patterns_dir=args.summary_out_dir,
    )
    if not summary.empty:
        args.summary_out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.summary_out_dir / "results_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote consolidated summary: {summary_path.resolve()} ({len(summary)} rows)")
        print_results_summary_narrative(summary)


if __name__ == "__main__":
    main()
