#!/usr/bin/env python3
"""
Daily-close GC/SI ratio reversion timing study.

For each trigger k (default 1.5 and 2.0):
- Baseline: prior ISO week's daily ratios (mu, sigma) — same as backtest_gc_si_ratio_daily.
- Event starts when ratio exits mu +/- k*sigma (inside -> outside).
- Event ends on first close back inside the same band.
- DurationDays = revert_idx - cross_idx (same convention as daily backtest).

Outputs: events ledger, full-sample summary, yearly summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_gc_si_ratio_daily import aggregate_daily_closes, attach_baseline_daily
from plot_gc_si import load_merge

DEFAULT_TRIGGER_SIGMAS = (1.5, 2.0)


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    return here, data_dir


@dataclass
class OpenExcursion:
    cross_idx: int
    cross_date: date
    direction: str
    ratio_at_cross: float
    mu: float
    sigma: float
    trigger_sigma: float


def _is_inside(ratio: float, lower: float, upper: float) -> bool:
    return lower <= ratio <= upper


def scan_excursions(daily: pd.DataFrame, trigger_sigma: float) -> pd.DataFrame:
    """Detect excursions for one trigger width k."""
    k = trigger_sigma
    events: list[dict[str, object]] = []
    seen_inside = False
    inside = True
    open_exc: OpenExcursion | None = None
    n = len(daily)

    for i in range(n):
        row = daily.iloc[i]
        cal_day = row["cal_day"]
        ratio = float(row["ratio"])
        mu = float(row["ratio_mu"])
        sigma = float(row["ratio_sigma"])

        if np.isnan(mu) or np.isnan(sigma) or sigma <= 0.0:
            continue

        lower = mu - k * sigma
        upper = mu + k * sigma
        currently_inside = _is_inside(ratio, lower, upper)

        if not seen_inside:
            if currently_inside:
                seen_inside = True
                inside = True
            continue

        if open_exc is not None:
            if currently_inside:
                duration = i - open_exc.cross_idx
                events.append(
                    {
                        "TriggerSigma": open_exc.trigger_sigma,
                        "Direction": open_exc.direction,
                        "CrossDate": open_exc.cross_date,
                        "RevertDate": cal_day,
                        "DurationDays": duration,
                        "Censored": False,
                        "RatioAtCross": open_exc.ratio_at_cross,
                        "RatioAtRevert": ratio,
                        "Mu": open_exc.mu,
                        "Sigma": open_exc.sigma,
                        "DaysToLastBar": duration,
                    }
                )
                open_exc = None
            inside = currently_inside
            continue

        if inside and not currently_inside:
            direction = "upper" if ratio > upper else "lower"
            open_exc = OpenExcursion(
                cross_idx=i,
                cross_date=cal_day,
                direction=direction,
                ratio_at_cross=ratio,
                mu=mu,
                sigma=sigma,
                trigger_sigma=k,
            )

        inside = currently_inside

    if open_exc is not None:
        last_idx = n - 1
        events.append(
            {
                "TriggerSigma": open_exc.trigger_sigma,
                "Direction": open_exc.direction,
                "CrossDate": open_exc.cross_date,
                "RevertDate": pd.NaT,
                "DurationDays": np.nan,
                "Censored": True,
                "RatioAtCross": open_exc.ratio_at_cross,
                "RatioAtRevert": np.nan,
                "Mu": open_exc.mu,
                "Sigma": open_exc.sigma,
                "DaysToLastBar": last_idx - open_exc.cross_idx,
            }
        )

    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events)


def _duration_stats(durations: pd.Series) -> dict[str, float]:
    d = durations.dropna().astype(float)
    if len(d) == 0:
        return {
            "MeanDurationDays": float("nan"),
            "MedianDurationDays": float("nan"),
            "MinDurationDays": float("nan"),
            "MaxDurationDays": float("nan"),
            "P25DurationDays": float("nan"),
            "P75DurationDays": float("nan"),
            "P90DurationDays": float("nan"),
            "StdDurationDays": float("nan"),
        }
    return {
        "MeanDurationDays": float(d.mean()),
        "MedianDurationDays": float(d.median()),
        "MinDurationDays": float(d.min()),
        "MaxDurationDays": float(d.max()),
        "P25DurationDays": float(d.quantile(0.25)),
        "P75DurationDays": float(d.quantile(0.75)),
        "P90DurationDays": float(d.quantile(0.90)),
        "StdDurationDays": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
    }


def summarize_events(events: pd.DataFrame) -> dict[str, object]:
    """Aggregate counts and duration stats for one event subset."""
    if events.empty:
        return {
            "Events": 0,
            "Reverted": 0,
            "Censored": 0,
            "PctReverted": float("nan"),
            **_duration_stats(pd.Series(dtype=float)),
        }

    censored = events["Censored"].astype(bool)
    n_events = len(events)
    n_censored = int(censored.sum())
    n_reverted = n_events - n_censored
    pct = (n_reverted / n_events) if n_events else float("nan")
    dur = events.loc[~censored, "DurationDays"]

    return {
        "Events": n_events,
        "Reverted": n_reverted,
        "Censored": n_censored,
        "PctReverted": pct,
        **_duration_stats(dur),
    }


def build_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    triggers = sorted(events["TriggerSigma"].unique())
    for k in triggers:
        sub_k = events[events["TriggerSigma"] == k]
        for direction in ("upper", "lower"):
            sub = sub_k[sub_k["Direction"] == direction]
            if sub.empty:
                continue
            rows.append(
                {"TriggerSigma": k, "Direction": direction, **summarize_events(sub)}
            )
        rows.append(
            {"TriggerSigma": k, "Direction": "all", **summarize_events(sub_k)}
        )

    rows.append(
        {
            "TriggerSigma": "all",
            "Direction": "all",
            **summarize_events(events),
        }
    )
    return pd.DataFrame(rows)


def _event_year(row: pd.Series) -> int:
    if bool(row["Censored"]):
        return int(pd.Timestamp(row["CrossDate"]).year)
    return int(pd.Timestamp(row["RevertDate"]).year)


def build_summary_by_year(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    ev = events.copy()
    ev["Year"] = ev.apply(_event_year, axis=1)
    rows: list[dict[str, object]] = []

    for year in sorted(ev["Year"].unique()):
        sub_y = ev[ev["Year"] == year]
        for k in sorted(sub_y["TriggerSigma"].unique()):
            sub_k = sub_y[sub_y["TriggerSigma"] == k]
            for direction in ("upper", "lower", "all"):
                if direction == "all":
                    sub = sub_k
                else:
                    sub = sub_k[sub_k["Direction"] == direction]
                if sub.empty:
                    continue
                rows.append(
                    {
                        "Year": int(year),
                        "TriggerSigma": k,
                        "Direction": direction,
                        **summarize_events(sub),
                    }
                )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    root, data_dir = _default_paths()
    p = argparse.ArgumentParser(
        description="Daily GC/SI ratio time-to-reversion inside mu +/- k*sigma band."
    )
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument("--out-dir", type=Path, default=root / "results")
    p.add_argument(
        "--trigger-sigmas",
        type=float,
        nargs="+",
        default=list(DEFAULT_TRIGGER_SIGMAS),
        help="Band half-widths in sigma units (default: 1.5 2.0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.gc, "GC"), (args.si, "SI")):
        if not path.is_file():
            raise SystemExit(f"Missing {label} CSV: {path.resolve()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merge(args.gc, args.si)
    daily = attach_baseline_daily(aggregate_daily_closes(merged))

    frames: list[pd.DataFrame] = []
    for k in args.trigger_sigmas:
        part = scan_excursions(daily, trigger_sigma=float(k))
        if not part.empty:
            frames.append(part)

    if frames:
        events = pd.concat(frames, ignore_index=True).sort_values(
            ["TriggerSigma", "CrossDate", "Direction"]
        )
    else:
        events = pd.DataFrame()

    events_path = args.out_dir / "gc_si_ratio_reversion_daily_events.csv"
    summary_path = args.out_dir / "gc_si_ratio_reversion_daily_summary.csv"
    by_year_path = args.out_dir / "gc_si_ratio_reversion_daily_by_year.csv"

    events.to_csv(events_path, index=False)
    build_summary(events).to_csv(summary_path, index=False)
    build_summary_by_year(events).to_csv(by_year_path, index=False)

    print(f"Wrote events: {events_path.resolve()}")
    print(f"Wrote summary: {summary_path.resolve()}")
    print(f"Wrote by-year summary: {by_year_path.resolve()}")
    print(f"Total excursions: {len(events)}")

    if not events.empty:
        for k in sorted(events["TriggerSigma"].unique()):
            sub = events[events["TriggerSigma"] == k]
            rev = sub[~sub["Censored"].astype(bool)]
            pct = len(rev) / len(sub) if len(sub) else 0.0
            med = rev["DurationDays"].median() if len(rev) else float("nan")
            print(
                f"  k={k:g} | events={len(sub)} | "
                f"reverted={len(rev)} ({pct:.1%}) | median_days={med}"
            )

        bad = events[~events["Censored"].astype(bool)]
        bad = bad[bad["DurationDays"].astype(float) <= 0]
        if len(bad):
            raise SystemExit(f"Invariant failed: {len(bad)} reverted events with DurationDays <= 0")


if __name__ == "__main__":
    main()
