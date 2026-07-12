#!/usr/bin/env python3
"""Rebuild consolidated spread backtest results_summary.csv from existing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest_gc_si_ratio_daily import DEFAULT_ACCOUNT_VALUE
from backtest_gc_si_spread import build_results_summary, print_results_summary_narrative


def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    results_root = root / "results" / "spread_delays_adaptive_risk"
    return root, results_root


def parse_args() -> argparse.Namespace:
    root, results_root = _default_paths()
    p = argparse.ArgumentParser(
        description="Build spread_backtest results_summary.csv from Phase 1/2 CSVs."
    )
    p.add_argument(
        "--extended-dir",
        type=Path,
        default=results_root / "phase1_extended",
    )
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=results_root / "phase2_adaptive",
        help="Optional Phase 2 dir; omit with --phase1-only for delay sections only.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=results_root,
    )
    p.add_argument(
        "--reversion-path",
        type=Path,
        default=root / "results" / "reversion_duration_backtest" / "gc_si_ratio_reversion_daily_summary.csv",
    )
    p.add_argument("--account-value", type=float, default=DEFAULT_ACCOUNT_VALUE)
    p.add_argument(
        "--phase1-only",
        action="store_true",
        help="Write Phase 1 sections only (no adaptive_vs_baseline).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adaptive_dir = None if args.phase1_only else args.adaptive_dir

    summary = build_results_summary(
        args.extended_dir,
        adaptive_dir=adaptive_dir,
        reversion_path=args.reversion_path,
        account_value=args.account_value,
        month_patterns_dir=args.out_dir,
    )
    if summary.empty:
        raise SystemExit(
            f"No summary rows built. Check sweep files under {args.extended_dir.resolve()}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "results_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path.resolve()} ({len(summary)} rows)")
    print(f"Sections: {sorted(summary['SummarySection'].unique().tolist())}")
    print_results_summary_narrative(summary)


if __name__ == "__main__":
    main()
