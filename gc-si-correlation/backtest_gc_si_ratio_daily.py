#!/usr/bin/env python3
"""
Daily-close backtest: GC/SI ratio +/- 2 sigma signals.

Baseline: prior ISO week's daily ratio samples (GC close / SI close), Mon–Fri.

Mean reversion (default): arm when ratio crosses band; enter on the *next* trading
day only if still beyond the same threshold (short if still > upper, long if
still < lower).

Continuation (--continuation): long at the same close when ratio > upper; short
when ratio < lower (no confirmation delay).

Exit: +3% / -1% of starting account value (1 GC, $100 multiplier).
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
PROFIT_TARGET_PCT = 0.03
STOP_LOSS_PCT = 0.01
TRADING_DAYS_PER_YEAR = 252.0


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "gc-si"
    return here, data_dir


def _iso_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def aggregate_daily_closes(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: last merged bar of that day (session close proxy)."""
    m = merged.sort_values("ts")
    idx = m.groupby("cal_day", sort=True)["ts"].idxmax()
    d = m.loc[idx].copy()
    d["ratio"] = d["close_gc"].astype(float) / d["close_si"].astype(float)
    d["cal_day"] = pd.to_datetime(d["cal_day"]).dt.date
    d = d.reset_index(drop=True)
    return d[
        ["cal_day", "close_gc", "close_si", "ratio"]
    ].sort_values("cal_day").reset_index(drop=True)


def attach_baseline_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Map each row's ISO week to mu/sigma from the *prior* week's daily ratios."""
    df = daily[daily["cal_day"].map(lambda d: d.weekday() <= 4)].copy()
    df["week_start"] = df["cal_day"].map(_iso_week_start)
    baseline_map: dict[date, tuple[float, float]] = {}

    ratio_by_week: dict[date, np.ndarray] = {}
    for wk, g in df.groupby("week_start", sort=True):
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

    df["ratio_mu"] = df["week_start"].map(lambda w: baseline_map.get(w, (np.nan, np.nan))[0])
    df["ratio_sigma"] = df["week_start"].map(lambda w: baseline_map.get(w, (np.nan, np.nan))[1])
    return df.sort_values("cal_day").reset_index(drop=True)


@dataclass
class OpenPosition:
    side: int  # +1 long, -1 short
    entry_idx: int
    entry_date: date
    entry_gc: float
    entry_si: float


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_series: pd.Series  # index: cal_day, values: equity
    daily_returns: pd.Series


def run_backtest(
    daily: pd.DataFrame, account_value: float, *, continuation: bool = False
) -> BacktestResult:
    profit_target = account_value * PROFIT_TARGET_PCT
    stop_loss = account_value * STOP_LOSS_PCT

    trades: list[dict[str, object]] = []
    pos: OpenPosition | None = None
    pending_side: int | None = None
    cum_closed_pnl = 0.0

    equity_by_day: list[tuple[date, float]] = []

    n = len(daily)
    for i in range(n):
        skip_arm_today = False
        row = daily.iloc[i]
        cal_day = row["cal_day"]
        gc_price = float(row["close_gc"])
        si_price = float(row["close_si"])
        ratio = float(row["ratio"])
        mu = float(row["ratio_mu"])
        sigma = float(row["ratio_sigma"])
        valid_baseline = not (np.isnan(mu) or np.isnan(sigma) or sigma <= 0.0)
        upper = mu + 2.0 * sigma if valid_baseline else float("nan")
        lower = mu - 2.0 * sigma if valid_baseline else float("nan")

        if pos is not None:
            pnl = (gc_price - pos.entry_gc) * GC_MULTIPLIER * pos.side
            if pnl >= profit_target or pnl <= -stop_loss:
                trades.append(
                    {
                        "EntryDate": pos.entry_date,
                        "ExitDate": cal_day,
                        "Side": "long" if pos.side == 1 else "short",
                        "EntryGC": pos.entry_gc,
                        "ExitGC": gc_price,
                        "EntrySI": pos.entry_si,
                        "ExitSI": si_price,
                        "ProfitOrLoss": round(pnl, 2),
                        "DurationDays": i - pos.entry_idx,
                    }
                )
                cum_closed_pnl += pnl
                pos = None
                pending_side = None

        if pos is None:
            if continuation:
                if valid_baseline:
                    if ratio > upper:
                        pos = OpenPosition(
                            side=1,
                            entry_idx=i,
                            entry_date=cal_day,
                            entry_gc=gc_price,
                            entry_si=si_price,
                        )
                    elif ratio < lower:
                        pos = OpenPosition(
                            side=-1,
                            entry_idx=i,
                            entry_date=cal_day,
                            entry_gc=gc_price,
                            entry_si=si_price,
                        )
            else:
                if pending_side is not None:
                    if valid_baseline:
                        if pending_side == -1:
                            if ratio > upper:
                                pos = OpenPosition(
                                    side=-1,
                                    entry_idx=i,
                                    entry_date=cal_day,
                                    entry_gc=gc_price,
                                    entry_si=si_price,
                                )
                            elif ratio < lower:
                                pending_side = None
                                skip_arm_today = True
                            else:
                                pending_side = None
                        else:
                            if ratio < lower:
                                pos = OpenPosition(
                                    side=1,
                                    entry_idx=i,
                                    entry_date=cal_day,
                                    entry_gc=gc_price,
                                    entry_si=si_price,
                                )
                            elif ratio > upper:
                                pending_side = None
                                skip_arm_today = True
                            else:
                                pending_side = None
                    else:
                        pending_side = None
                    if pos is not None:
                        pending_side = None

                if pos is None and pending_side is None and valid_baseline and not skip_arm_today:
                    if ratio > upper:
                        pending_side = -1
                    elif ratio < lower:
                        pending_side = 1

        if pos is not None:
            unreal = (gc_price - pos.entry_gc) * GC_MULTIPLIER * pos.side
            equity = account_value + cum_closed_pnl + unreal
        else:
            equity = account_value + cum_closed_pnl

        equity_by_day.append((cal_day, equity))

    equity_series = pd.Series(
        [e for _, e in equity_by_day], index=[d for d, _ in equity_by_day], name="equity"
    )
    eq_vals = equity_series.values.astype(float)
    prev = np.roll(eq_vals, 1)
    prev[0] = eq_vals[0]
    rets = np.where(prev != 0.0, eq_vals / prev - 1.0, 0.0)
    rets[0] = 0.0
    daily_returns = pd.Series(rets, index=equity_series.index, name="daily_return")

    return BacktestResult(
        trades=pd.DataFrame(trades),
        equity_series=equity_series,
        daily_returns=daily_returns,
    )


def _max_drawdown_fraction(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0.0, np.nan)
    return float(abs(dd.min())) if dd.notna().any() else 0.0


def _max_drawdown_dollars(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = equity - peak
    return float(abs(dd.min())) if len(dd) else 0.0


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or float(r.std()) == 0.0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0.0]
    if len(downside) < 1:
        return float("inf") if float(r.mean()) > 0 else float("nan")
    dstd = float(downside.std(ddof=0))
    if dstd == 0.0:
        return float("nan")
    return float(r.mean() / dstd * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize_by_year(
    trades: pd.DataFrame,
    equity_series: pd.Series,
    daily_returns: pd.Series,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    t = trades.copy()
    t["ExitYear"] = pd.to_datetime(t["ExitDate"]).dt.year
    rows: list[dict[str, object]] = []

    for year in sorted(t["ExitYear"].unique()):
        ty = t[t["ExitYear"] == year].sort_values("ExitDate")
        pnl = ty["ProfitOrLoss"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        n_w, n_l = int((pnl > 0).sum()), int((pnl < 0).sum())
        win_loss_ratio = (n_w / n_l) if n_l > 0 else (float("inf") if n_w > 0 else float("nan"))

        win_dur = ty.loc[pnl > 0, "DurationDays"].astype(float)
        loss_dur = ty.loc[pnl < 0, "DurationDays"].astype(float)

        is_win = (pnl > 0).values
        max_cons_w = _max_consecutive(is_win)
        max_cons_l = _max_consecutive(~is_win)

        mask_yr = equity_series.index.map(lambda d: d.year == year)
        eq_y = equity_series.loc[mask_yr]
        ret_y = daily_returns.loc[mask_yr]

        mdd_frac = _max_drawdown_fraction(eq_y) if len(eq_y) > 1 else 0.0
        mdd_usd = _max_drawdown_dollars(eq_y) if len(eq_y) > 1 else 0.0
        sharpe = _sharpe(ret_y)
        sortino = _sortino(ret_y)

        if len(eq_y) >= 2:
            start_eq = float(eq_y.iloc[0])
            end_eq = float(eq_y.iloc[-1])
            yr_ret = (end_eq / start_eq - 1.0) if start_eq > 0 else float("nan")
        else:
            yr_ret = float("nan")

        if mdd_frac > 1e-12:
            calmar = yr_ret / mdd_frac
        else:
            calmar = float("inf") if yr_ret > 0 else float("nan")

        rows.append(
            {
                "Year": int(year),
                "Trades": len(ty),
                "Wins": n_w,
                "Losses": n_l,
                "WinLossRatio": win_loss_ratio,
                "AvgGain": float(wins.mean()) if len(wins) else float("nan"),
                "AvgLoss": float(losses.mean()) if len(losses) else float("nan"),
                "AvgDurationWinDays": float(win_dur.mean()) if len(win_dur) else float("nan"),
                "AvgDurationLossDays": float(loss_dur.mean()) if len(loss_dur) else float("nan"),
                "MaxConsecutiveWins": max_cons_w,
                "MaxConsecutiveLosses": max_cons_l,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Calmar": calmar,
                "MaxDrawdown": mdd_usd,
                "MaxDrawdownFraction": mdd_frac,
                "NetPnL": float(pnl.sum()),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    root, data_dir = _default_paths()
    p = argparse.ArgumentParser(
        description="Daily GC/SI ratio backtest (mean reversion or continuation breakout)."
    )
    p.add_argument("--gc", type=Path, default=data_dir / "GC.csv")
    p.add_argument("--si", type=Path, default=data_dir / "SI.csv")
    p.add_argument("--out-dir", type=Path, default=root / "results")
    p.add_argument("--account-value", type=float, default=DEFAULT_ACCOUNT_VALUE)
    p.add_argument(
        "--continuation",
        action="store_true",
        help="Breakout mode: long above +2σ / short below −2σ at same close (no 1-day delay).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.gc, "GC"), (args.si, "SI")):
        if not path.is_file():
            raise SystemExit(f"Missing {label} CSV: {path.resolve()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merge(args.gc, args.si)
    daily_raw = aggregate_daily_closes(merged)
    daily = attach_baseline_daily(daily_raw)

    result = run_backtest(
        daily,
        account_value=float(args.account_value),
        continuation=bool(args.continuation),
    )

    if args.continuation:
        trades_path = args.out_dir / "gc_si_ratio_daily_continuation_trades.csv"
        yearly_path = args.out_dir / "gc_si_ratio_daily_continuation_by_year.csv"
    else:
        trades_path = args.out_dir / "gc_si_ratio_daily_trades.csv"
        yearly_path = args.out_dir / "gc_si_ratio_daily_by_year.csv"

    result.trades.to_csv(trades_path, index=False)
    by_year = summarize_by_year(result.trades, result.equity_series, result.daily_returns)
    by_year.to_csv(yearly_path, index=False)

    print(f"Wrote trades: {trades_path.resolve()}")
    print(f"Wrote yearly summary: {yearly_path.resolve()}")
    print(f"Total trades: {len(result.trades)}")


if __name__ == "__main__":
    main()
