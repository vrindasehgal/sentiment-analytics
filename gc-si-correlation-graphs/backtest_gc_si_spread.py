#!/usr/bin/env python3
"""
Daily GC/SI mean-reversion spread backtest.

On envelope breach (ratio exits mu +/- k*sigma), open a dollar-neutral spread at
entry day close (both GC and SI legs). Exit flat when combined PnL hits
profit/stop or at entry + N trading days (calendar backstop), whichever comes
first (profit/stop checked before calendar on the same day).

Ratio HIGH -> sell GC / buy SI. Ratio LOW -> buy GC / sell SI (mirror).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from backtest_gc_si_ratio_daily import (
    TRADING_DAYS_PER_YEAR,
    _max_drawdown_dollars,
    _max_drawdown_fraction,
    _sharpe,
    _sortino,
    aggregate_daily_closes,
    attach_baseline_daily,
)

GC_MULTIPLIER = 100.0
SI_MULTIPLIER = 5000.0
GC_CONTRACTS = 1.0
DEFAULT_ACCOUNT_VALUE = 100_000.0

Mode = Literal["probable_duration", "wait_entry"]

REWARD_RISK_PCTS: dict[str, tuple[float, float]] = {
    "1:3": (0.03, 0.01),
    "1:2": (0.02, 0.01),
    "1:1": (0.02, 0.02),
    "1:5": (0.05, 0.01),
}


def _is_inside(ratio: float, lower: float, upper: float) -> bool:
    return lower <= ratio <= upper


def si_contracts_dollar_equiv(gc_price: float, si_price: float) -> float:
    return (gc_price * GC_MULTIPLIER) / (si_price * SI_MULTIPLIER)


def spread_pnl(
    gc: float,
    si: float,
    *,
    spread_dir: int,
    entry_gc: float,
    entry_si: float,
    si_contracts: float,
) -> float:
    """spread_dir +1 = long GC / short SI; -1 = short GC / long SI."""
    gc_pnl = spread_dir * (gc - entry_gc) * GC_MULTIPLIER * GC_CONTRACTS
    si_pnl = -spread_dir * (si - entry_si) * SI_MULTIPLIER * si_contracts
    return float(gc_pnl + si_pnl)


@dataclass
class OpenSpread:
    signal_idx: int
    signal_date: date
    entry_idx: int
    entry_date: date
    direction: str  # "high" | "low"
    spread_dir: int
    entry_gc: float
    entry_si: float
    si_contracts: float
    calendar_exit_idx: int


@dataclass
class PendingEntry:
    signal_idx: int
    signal_date: date
    entry_idx: int
    direction: str
    spread_dir: int
    ratio_at_signal: float


@dataclass
class SpreadBacktestResult:
    trades: pd.DataFrame
    skipped_signals: pd.DataFrame
    equity_series: pd.Series
    daily_returns: pd.Series


def _bands(mu: float, sigma: float, trigger_sigma: float) -> tuple[float, float] | None:
    if np.isnan(mu) or np.isnan(sigma) or sigma <= 0.0:
        return None
    return mu - trigger_sigma * sigma, mu + trigger_sigma * sigma


def run_spread_backtest(
    daily: pd.DataFrame,
    account_value: float,
    *,
    mode: Mode,
    trigger_sigma: float,
    entry_lag: int,
    calendar_n: int,
    reward_risk: str,
) -> SpreadBacktestResult:
    if reward_risk not in REWARD_RISK_PCTS:
        raise ValueError(f"Unknown reward_risk {reward_risk!r}")
    profit_pct, stop_pct = REWARD_RISK_PCTS[reward_risk]
    profit_target = account_value * profit_pct
    stop_loss = account_value * stop_pct

    trades: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    position: OpenSpread | None = None
    pending: PendingEntry | None = None
    cum_closed_pnl = 0.0
    equity_by_day: list[tuple[date, float]] = []

    seen_inside = False
    inside = True
    n = len(daily)

    def log_skip(signal_idx: int, signal_date: date, direction: str, ratio: float) -> None:
        skipped.append(
            {
                "SignalDate": signal_date,
                "Direction": direction,
                "RatioAtSignal": ratio,
                "TriggerSigma": trigger_sigma,
                "Mode": mode,
                "EntryLag": entry_lag,
                "CalendarN": calendar_n,
                "RewardRisk": reward_risk,
                "SkipReason": "position_open",
            }
        )

    def close_position(pos: OpenSpread, exit_idx: int, exit_reason: str, censored: bool = False) -> None:
        nonlocal position, cum_closed_pnl
        row = daily.iloc[exit_idx]
        exit_gc = float(row["close_gc"])
        exit_si = float(row["close_si"])
        pnl = spread_pnl(
            exit_gc,
            exit_si,
            spread_dir=pos.spread_dir,
            entry_gc=pos.entry_gc,
            entry_si=pos.entry_si,
            si_contracts=pos.si_contracts,
        )
        trades.append(
            {
                "TriggerSigma": trigger_sigma,
                "Mode": mode,
                "EntryLag": entry_lag,
                "CalendarN": calendar_n,
                "RewardRisk": reward_risk,
                "SignalDate": pos.signal_date,
                "EntryDate": pos.entry_date,
                "ExitDate": row["cal_day"],
                "Direction": pos.direction,
                "SpreadDir": pos.spread_dir,
                "EntryGC": pos.entry_gc,
                "EntrySI": pos.entry_si,
                "ExitGC": exit_gc,
                "ExitSI": exit_si,
                "SiContracts": pos.si_contracts,
                "ProfitOrLoss": round(pnl, 2),
                "DurationDays": exit_idx - pos.entry_idx,
                "ExitReason": exit_reason,
                "Censored": censored,
            }
        )
        cum_closed_pnl += pnl
        position = None

    def open_position(pend: PendingEntry, entry_idx: int) -> None:
        nonlocal position, pending
        row = daily.iloc[entry_idx]
        entry_gc = float(row["close_gc"])
        entry_si = float(row["close_si"])
        si_qty = si_contracts_dollar_equiv(entry_gc, entry_si)
        position = OpenSpread(
            signal_idx=pend.signal_idx,
            signal_date=pend.signal_date,
            entry_idx=entry_idx,
            entry_date=row["cal_day"],
            direction=pend.direction,
            spread_dir=pend.spread_dir,
            entry_gc=entry_gc,
            entry_si=entry_si,
            si_contracts=si_qty,
            calendar_exit_idx=entry_idx + calendar_n,
        )
        pending = None

    for i in range(n):
        row = daily.iloc[i]
        cal_day = row["cal_day"]
        gc_price = float(row["close_gc"])
        si_price = float(row["close_si"])
        ratio = float(row["ratio"])
        mu = float(row["ratio_mu"])
        sigma = float(row["ratio_sigma"])

        band = _bands(mu, sigma, trigger_sigma)
        signal_cross = False
        cross_direction: str | None = None
        cross_spread_dir = 0

        if band is not None:
            lower, upper = band
            currently_inside = _is_inside(ratio, lower, upper)
            if seen_inside and inside and not currently_inside:
                signal_cross = True
                if ratio > upper:
                    cross_direction = "high"
                    cross_spread_dir = -1
                else:
                    cross_direction = "low"
                    cross_spread_dir = 1
            if not seen_inside and currently_inside:
                seen_inside = True
            inside = currently_inside if seen_inside else inside

        if position is not None:
            pnl = spread_pnl(
                gc_price,
                si_price,
                spread_dir=position.spread_dir,
                entry_gc=position.entry_gc,
                entry_si=position.entry_si,
                si_contracts=position.si_contracts,
            )
            exit_reason: str | None = None
            if pnl >= profit_target:
                exit_reason = "profit"
            elif pnl <= -stop_loss:
                exit_reason = "stop"
            elif i >= position.calendar_exit_idx:
                exit_reason = "calendar"
            if exit_reason is not None:
                close_position(position, i, exit_reason)

        if signal_cross and cross_direction is not None:
            if position is not None:
                log_skip(i, cal_day, cross_direction, ratio)
            elif pending is not None:
                log_skip(i, cal_day, cross_direction, ratio)
            else:
                entry_idx = i + entry_lag
                if entry_idx < n:
                    pending = PendingEntry(
                        signal_idx=i,
                        signal_date=cal_day,
                        entry_idx=entry_idx,
                        direction=cross_direction,
                        spread_dir=cross_spread_dir,
                        ratio_at_signal=ratio,
                    )
                else:
                    skipped.append(
                        {
                            "SignalDate": cal_day,
                            "Direction": cross_direction,
                            "RatioAtSignal": ratio,
                            "TriggerSigma": trigger_sigma,
                            "Mode": mode,
                            "EntryLag": entry_lag,
                            "CalendarN": calendar_n,
                            "RewardRisk": reward_risk,
                            "SkipReason": "entry_beyond_data",
                        }
                    )

        if position is None and pending is not None and i == pending.entry_idx:
            open_position(pending, i)

        if position is not None:
            unreal = spread_pnl(
                gc_price,
                si_price,
                spread_dir=position.spread_dir,
                entry_gc=position.entry_gc,
                entry_si=position.entry_si,
                si_contracts=position.si_contracts,
            )
            equity = account_value + cum_closed_pnl + unreal
        else:
            equity = account_value + cum_closed_pnl
        equity_by_day.append((cal_day, equity))

    if position is not None:
        close_position(position, n - 1, "censored", censored=True)

    equity_series = pd.Series(
        [e for _, e in equity_by_day], index=[d for d, _ in equity_by_day], name="equity"
    )
    eq_vals = equity_series.values.astype(float)
    prev = np.roll(eq_vals, 1)
    prev[0] = eq_vals[0]
    rets = np.where(prev != 0.0, eq_vals / prev - 1.0, 0.0)
    rets[0] = 0.0
    daily_returns = pd.Series(rets, index=equity_series.index, name="daily_return")

    return SpreadBacktestResult(
        trades=pd.DataFrame(trades),
        skipped_signals=pd.DataFrame(skipped),
        equity_series=equity_series,
        daily_returns=daily_returns,
    )


def summarize_sweep(
    trades: pd.DataFrame,
    skipped_signals: pd.DataFrame,
    equity_series: pd.Series,
    daily_returns: pd.Series,
) -> dict[str, float | int]:
    n_trades = len(trades)
    n_skipped = len(skipped_signals)
    if n_trades == 0:
        return {
            "Trades": 0,
            "SignalsSkipped": n_skipped,
            "WinRate": float("nan"),
            "AvgWin": float("nan"),
            "AvgLoss": float("nan"),
            "NetPnL": 0.0,
            "MaxDrawdown": 0.0,
            "PctExitEarly": float("nan"),
            "PctExitCalendar": float("nan"),
            "Sharpe": float("nan"),
        }

    pnl = trades["ProfitOrLoss"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    early = trades["ExitReason"].isin(["profit", "stop"]).sum()
    calendar = (trades["ExitReason"] == "calendar").sum()
    completed = trades[~trades["Censored"].astype(bool)]

    peak = equity_series.cummax()
    mdd = float(abs((equity_series - peak).min())) if len(equity_series) > 1 else 0.0

    return {
        "Trades": n_trades,
        "SignalsSkipped": n_skipped,
        "WinRate": float((pnl > 0).mean()),
        "AvgWin": float(wins.mean()) if len(wins) else float("nan"),
        "AvgLoss": float(losses.mean()) if len(losses) else float("nan"),
        "NetPnL": float(pnl.sum()),
        "MaxDrawdown": mdd,
        "PctExitEarly": float(early / len(completed)) if len(completed) else float("nan"),
        "PctExitCalendar": float(calendar / len(completed)) if len(completed) else float("nan"),
        "Sharpe": _sharpe(daily_returns),
    }


def summarize_by_year_with_params(
    trades: pd.DataFrame,
    equity_series: pd.Series,
    daily_returns: pd.Series,
    params: dict[str, object],
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    t = trades.copy()
    t["ExitYear"] = pd.to_datetime(t["ExitDate"]).dt.year
    rows: list[dict[str, object]] = []

    for year in sorted(t["ExitYear"].unique()):
        ty = t[t["ExitYear"] == year]
        pnl = ty["ProfitOrLoss"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        mask_yr = equity_series.index.map(lambda d: d.year == year)
        eq_y = equity_series.loc[mask_yr]
        ret_y = daily_returns.loc[mask_yr]
        mdd_usd = _max_drawdown_dollars(eq_y) if len(eq_y) > 1 else 0.0
        mdd_frac = _max_drawdown_fraction(eq_y) if len(eq_y) > 1 else 0.0

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
                **params,
                "Year": int(year),
                "Trades": len(ty),
                "Wins": int((pnl > 0).sum()),
                "Losses": int((pnl < 0).sum()),
                "AvgGain": float(wins.mean()) if len(wins) else float("nan"),
                "AvgLoss": float(losses.mean()) if len(losses) else float("nan"),
                "Sharpe": _sharpe(ret_y),
                "Sortino": _sortino(ret_y),
                "Calmar": calmar,
                "MaxDrawdown": mdd_usd,
                "NetPnL": float(pnl.sum()),
            }
        )

    return pd.DataFrame(rows)
