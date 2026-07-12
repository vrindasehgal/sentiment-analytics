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
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from backtest_gc_si_ratio_daily import (
    TRADING_DAYS_PER_YEAR,
    _max_consecutive,
    _max_drawdown_dollars,
    _max_drawdown_fraction,
    _sharpe,
    _sortino,
    aggregate_daily_closes,
    attach_baseline_daily,
)

GC_MULTIPLIER = 100.0
SI_MULTIPLIER = 5000.0
GC_CONTRACTS_BASE = 1.0
DEFAULT_ACCOUNT_VALUE = 100_000.0

Mode = Literal["probable_duration", "wait_entry"]

REWARD_RISK_PCTS: dict[str, tuple[float, float]] = {
    "1:3": (0.03, 0.01),
    "1:2": (0.02, 0.01),
    "1:1": (0.02, 0.02),
    "1:5": (0.05, 0.01),
}

PARAM_KEYS = ("TriggerSigma", "Mode", "EntryLag", "CalendarN", "RewardRisk")
SWEEP_METRIC_COLS = (
    "Trades",
    "WinRate",
    "AvgWin",
    "AvgLoss",
    "AvgWinPctOfAccount",
    "AvgLossPctOfAccount",
    "NetPnL",
    "MaxDrawdown",
    "Sharpe",
    "PctExitEarly",
    "PctExitCalendar",
    "SignalsSkipped",
    "SignalsSkippedCoolingOff",
    "PctCensored",
)

ORIGINAL_PROBABLE_ENTRY_LAGS = {0}
ORIGINAL_PROBABLE_CALENDAR_N = {1, 2, 3, 5}
ORIGINAL_WAIT_ENTRY_LAGS = {1, 2}
ORIGINAL_WAIT_CALENDAR_N = {1, 2}

NEW_PROBABLE_CALENDAR_N = {6, 7, 8, 10}
NEW_PROBABLE_ENTRY_LAGS = {1, 2, 3, 5}
NEW_WAIT_ENTRY_LAGS = {3, 4, 5, 6}
NEW_WAIT_CALENDAR_N = {3, 4, 5, 6}

# Representative entry lags for side-by-side delay curve tables (best CalendarN per lag).
DELAY_CURVE_ENTRY_LAGS: dict[str, tuple[int, ...]] = {
    "probable_duration": (0, 1, 2, 5),
    "wait_entry": (1, 2, 4, 6),
}


@dataclass(frozen=True)
class RiskControls:
    """Fixed adaptive sizing rules for Phase 2."""

    scale_up_mult: float = 1.5
    scale_down_mult: float = 0.5
    win_rate_threshold: float = 0.5
    min_trades_for_scale_up: int = 2
    consecutive_losses_for_cooldown: int = 2


DEFAULT_RISK_CONTROLS = RiskControls()


def _is_inside(ratio: float, lower: float, upper: float) -> bool:
    return lower <= ratio <= upper


def si_contracts_dollar_equiv(
    gc_price: float, si_price: float, *, gc_contracts: float = GC_CONTRACTS_BASE
) -> float:
    return (gc_price * GC_MULTIPLIER * gc_contracts) / (si_price * SI_MULTIPLIER)


def spread_pnl(
    gc: float,
    si: float,
    *,
    spread_dir: int,
    entry_gc: float,
    entry_si: float,
    si_contracts: float,
    gc_contracts: float = GC_CONTRACTS_BASE,
) -> float:
    """spread_dir +1 = long GC / short SI; -1 = short GC / long SI."""
    gc_pnl = spread_dir * (gc - entry_gc) * GC_MULTIPLIER * gc_contracts
    si_pnl = -spread_dir * (si - entry_si) * SI_MULTIPLIER * si_contracts
    return float(gc_pnl + si_pnl)


@dataclass
class OpenSpread:
    signal_idx: int
    signal_date: date
    entry_idx: int
    entry_date: date
    direction: str
    spread_dir: int
    entry_gc: float
    entry_si: float
    si_contracts: float
    gc_contracts: float
    size_mult: float
    calendar_exit_idx: int
    month_win_rate_at_entry: float
    consecutive_losses_at_entry: int


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


def _year_month(d: date) -> tuple[int, int]:
    return d.year, d.month


def _month_win_rate(month_stats: dict[tuple[int, int], dict[str, int]], ym: tuple[int, int]) -> float:
    s = month_stats.get(ym, {"wins": 0, "losses": 0})
    total = s["wins"] + s["losses"]
    if total == 0:
        return float("nan")
    return s["wins"] / total


def _compute_size_mult(
    consecutive_losses: int,
    month_stats: dict[tuple[int, int], dict[str, int]],
    entry_ym: tuple[int, int],
    risk: RiskControls,
) -> float:
    mult = 1.0
    wr = _month_win_rate(month_stats, entry_ym)
    s = month_stats.get(entry_ym, {"wins": 0, "losses": 0})
    mtd_trades = s["wins"] + s["losses"]
    if mtd_trades >= risk.min_trades_for_scale_up and wr > risk.win_rate_threshold:
        mult = max(mult, risk.scale_up_mult)
    if consecutive_losses == 1:
        mult = min(mult, risk.scale_down_mult)
    return mult


def _in_cooling_off(cal_day: date, cooling_off_month: tuple[int, int] | None) -> bool:
    if cooling_off_month is None:
        return False
    return _year_month(cal_day) == cooling_off_month


def run_spread_backtest(
    daily: pd.DataFrame,
    account_value: float,
    *,
    mode: Mode,
    trigger_sigma: float,
    entry_lag: int,
    calendar_n: int,
    reward_risk: str,
    risk_controls: RiskControls | None = None,
) -> SpreadBacktestResult:
    if reward_risk not in REWARD_RISK_PCTS:
        raise ValueError(f"Unknown reward_risk {reward_risk!r}")
    profit_pct, stop_pct = REWARD_RISK_PCTS[reward_risk]
    profit_target = account_value * profit_pct
    stop_loss = account_value * stop_pct
    risk = risk_controls

    trades: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    position: OpenSpread | None = None
    pending: PendingEntry | None = None
    cum_closed_pnl = 0.0
    equity_by_day: list[tuple[date, float]] = []

    consecutive_losses = 0
    cooling_off_month: tuple[int, int] | None = None
    month_stats: dict[tuple[int, int], dict[str, int]] = {}

    seen_inside = False
    inside = True
    n = len(daily)

    def skip_row(
        signal_date: date,
        direction: str,
        ratio: float,
        reason: str,
    ) -> None:
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
                "SkipReason": reason,
            }
        )

    def close_position(pos: OpenSpread, exit_idx: int, exit_reason: str, censored: bool = False) -> None:
        nonlocal position, cum_closed_pnl, consecutive_losses, cooling_off_month
        row = daily.iloc[exit_idx]
        exit_gc = float(row["close_gc"])
        exit_si = float(row["close_si"])
        exit_day = row["cal_day"]
        pnl = spread_pnl(
            exit_gc,
            exit_si,
            spread_dir=pos.spread_dir,
            entry_gc=pos.entry_gc,
            entry_si=pos.entry_si,
            si_contracts=pos.si_contracts,
            gc_contracts=pos.gc_contracts,
        )
        trade_row: dict[str, object] = {
            "TriggerSigma": trigger_sigma,
            "Mode": mode,
            "EntryLag": entry_lag,
            "CalendarN": calendar_n,
            "RewardRisk": reward_risk,
            "SignalDate": pos.signal_date,
            "EntryDate": pos.entry_date,
            "ExitDate": exit_day,
            "Direction": pos.direction,
            "SpreadDir": pos.spread_dir,
            "EntryGC": pos.entry_gc,
            "EntrySI": pos.entry_si,
            "ExitGC": exit_gc,
            "ExitSI": exit_si,
            "SiContracts": pos.si_contracts,
            "GcContracts": pos.gc_contracts,
            "SizeMult": pos.size_mult,
            "ProfitOrLoss": round(pnl, 2),
            "DurationDays": exit_idx - pos.entry_idx,
            "ExitReason": exit_reason,
            "Censored": censored,
        }
        if risk is not None:
            trade_row["MonthWinRateAtEntry"] = pos.month_win_rate_at_entry
            trade_row["ConsecutiveLossesAtEntry"] = pos.consecutive_losses_at_entry
        trades.append(trade_row)
        cum_closed_pnl += pnl

        if not censored and risk is not None:
            exit_ym = _year_month(exit_day)
            if exit_ym not in month_stats:
                month_stats[exit_ym] = {"wins": 0, "losses": 0}
            if pnl > 0:
                month_stats[exit_ym]["wins"] += 1
                consecutive_losses = 0
            elif pnl < 0:
                month_stats[exit_ym]["losses"] += 1
                consecutive_losses += 1
                if consecutive_losses >= risk.consecutive_losses_for_cooldown:
                    cooling_off_month = exit_ym
            else:
                consecutive_losses = 0

        position = None

    def open_position(pend: PendingEntry, entry_idx: int) -> None:
        nonlocal position, pending
        row = daily.iloc[entry_idx]
        cal_day = row["cal_day"]
        entry_gc = float(row["close_gc"])
        entry_si = float(row["close_si"])

        if risk is not None and _in_cooling_off(cal_day, cooling_off_month):
            skip_row(cal_day, pend.direction, pend.ratio_at_signal, "cooling_off")
            pending = None
            return

        entry_ym = _year_month(cal_day)
        wr_at_entry = _month_win_rate(month_stats, entry_ym)
        cons_at_entry = consecutive_losses
        size_mult = (
            _compute_size_mult(consecutive_losses, month_stats, entry_ym, risk)
            if risk is not None
            else 1.0
        )
        gc_qty = GC_CONTRACTS_BASE * size_mult
        si_qty = si_contracts_dollar_equiv(entry_gc, entry_si, gc_contracts=gc_qty)

        position = OpenSpread(
            signal_idx=pend.signal_idx,
            signal_date=pend.signal_date,
            entry_idx=entry_idx,
            entry_date=cal_day,
            direction=pend.direction,
            spread_dir=pend.spread_dir,
            entry_gc=entry_gc,
            entry_si=entry_si,
            si_contracts=si_qty,
            gc_contracts=gc_qty,
            size_mult=size_mult,
            calendar_exit_idx=entry_idx + calendar_n,
            month_win_rate_at_entry=wr_at_entry,
            consecutive_losses_at_entry=cons_at_entry,
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

        ym = _year_month(cal_day)
        if risk is not None and cooling_off_month is not None and ym > cooling_off_month:
            cooling_off_month = None

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
                gc_contracts=position.gc_contracts,
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
                skip_row(cal_day, cross_direction, ratio, "position_open")
            elif pending is not None:
                skip_row(cal_day, cross_direction, ratio, "position_open")
            elif risk is not None and _in_cooling_off(cal_day, cooling_off_month):
                skip_row(cal_day, cross_direction, ratio, "cooling_off")
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
                    skip_row(cal_day, cross_direction, ratio, "entry_beyond_data")

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
                gc_contracts=position.gc_contracts,
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
    account_value: float = DEFAULT_ACCOUNT_VALUE,
) -> dict[str, float | int]:
    n_trades = len(trades)
    n_skipped = len(skipped_signals)
    n_cooling = (
        int((skipped_signals["SkipReason"] == "cooling_off").sum())
        if n_skipped and "SkipReason" in skipped_signals.columns
        else 0
    )
    if n_trades == 0:
        return {
            "Trades": 0,
            "SignalsSkipped": n_skipped,
            "SignalsSkippedCoolingOff": n_cooling,
            "WinRate": float("nan"),
            "AvgWin": float("nan"),
            "AvgLoss": float("nan"),
            "AvgWinPctOfAccount": float("nan"),
            "AvgLossPctOfAccount": float("nan"),
            "NetPnL": 0.0,
            "MaxDrawdown": 0.0,
            "PctCensored": float("nan"),
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
    censored = trades["Censored"].astype(bool).sum()

    peak = equity_series.cummax()
    mdd = float(abs((equity_series - peak).min())) if len(equity_series) > 1 else 0.0
    acct = float(account_value) if account_value > 0 else DEFAULT_ACCOUNT_VALUE

    return {
        "Trades": n_trades,
        "SignalsSkipped": n_skipped,
        "SignalsSkippedCoolingOff": n_cooling,
        "WinRate": float((pnl > 0).mean()),
        "AvgWin": float(wins.mean()) if len(wins) else float("nan"),
        "AvgLoss": float(losses.mean()) if len(losses) else float("nan"),
        "AvgWinPctOfAccount": float(wins.mean() / acct) if len(wins) else float("nan"),
        "AvgLossPctOfAccount": float(abs(losses.mean()) / acct) if len(losses) else float("nan"),
        "NetPnL": float(pnl.sum()),
        "MaxDrawdown": mdd,
        "PctCensored": float(censored / n_trades),
        "PctExitEarly": float(early / len(completed)) if len(completed) else float("nan"),
        "PctExitCalendar": float(calendar / len(completed)) if len(completed) else float("nan"),
        "Sharpe": _sharpe(daily_returns),
    }


def summarize_by_month(
    trades: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    t = trades[~trades["Censored"].astype(bool)].copy()
    if t.empty:
        return pd.DataFrame()

    t["ExitMonth"] = pd.to_datetime(t["ExitDate"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []

    for month in sorted(t["ExitMonth"].unique()):
        tm = t[t["ExitMonth"] == month].sort_values("ExitDate")
        pnl = tm["ProfitOrLoss"].astype(float)
        is_win = (pnl > 0).values
        rows.append(
            {
                **params,
                "Month": month,
                "Trades": len(tm),
                "Wins": int((pnl > 0).sum()),
                "Losses": int((pnl < 0).sum()),
                "WinRate": float((pnl > 0).mean()),
                "NetPnL": float(pnl.sum()),
                "MaxConsecutiveWins": _max_consecutive(is_win),
                "MaxConsecutiveLosses": _max_consecutive(~is_win),
                "EndingConsecutiveLosses": _ending_consecutive_losses(is_win),
            }
        )

    return pd.DataFrame(rows)


def _ending_consecutive_losses(is_win: np.ndarray) -> int:
    streak = 0
    for v in reversed(is_win.tolist()):
        if v:
            break
        streak += 1
    return streak


def select_top_configs(
    sweep_summary: pd.DataFrame,
    *,
    mode: Mode,
    top_n: int = 3,
    min_trades: int = 50,
    max_pct_censored: float = 0.05,
) -> pd.DataFrame:
    if sweep_summary.empty:
        return pd.DataFrame()

    df = sweep_summary[sweep_summary["Mode"] == mode].copy()
    df = df[df["Trades"] >= min_trades]
    if "PctCensored" in df.columns:
        df = df[df["PctCensored"].fillna(1.0) <= max_pct_censored]

    timing_cols = ["TriggerSigma", "Mode", "EntryLag", "CalendarN"]
    best_per_timing = (
        df.sort_values(["Sharpe", "NetPnL"], ascending=[False, False])
        .groupby(timing_cols, as_index=False)
        .first()
    )
    top = best_per_timing.sort_values(["Sharpe", "NetPnL"], ascending=[False, False]).head(top_n)
    return top.reset_index(drop=True)


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


def _is_original_delay_row(row: pd.Series) -> bool:
    mode = row["Mode"]
    entry_lag = int(row["EntryLag"])
    cal_n = int(row["CalendarN"])
    if mode == "probable_duration":
        return entry_lag in ORIGINAL_PROBABLE_ENTRY_LAGS and cal_n in ORIGINAL_PROBABLE_CALENDAR_N
    return entry_lag in ORIGINAL_WAIT_ENTRY_LAGS and cal_n in ORIGINAL_WAIT_CALENDAR_N


def _is_new_delay_row(row: pd.Series) -> bool:
    mode = row["Mode"]
    entry_lag = int(row["EntryLag"])
    cal_n = int(row["CalendarN"])
    if mode == "probable_duration":
        return entry_lag in NEW_PROBABLE_ENTRY_LAGS or cal_n in NEW_PROBABLE_CALENDAR_N
    return entry_lag in NEW_WAIT_ENTRY_LAGS or cal_n in NEW_WAIT_CALENDAR_N


def _best_per_trigger(sweep: pd.DataFrame, subset: pd.DataFrame | None = None) -> pd.DataFrame:
    df = subset if subset is not None else sweep
    if df.empty:
        return pd.DataFrame()
    return (
        df.sort_values(["Sharpe", "NetPnL"], ascending=[False, False])
        .groupby(["Mode", "TriggerSigma"], as_index=False)
        .first()
    )


def _row_to_summary(row: pd.Series, summary_section: str, *, variant: str = "") -> dict[str, object]:
    out: dict[str, object] = {
        "SummarySection": summary_section,
        "Mode": row["Mode"],
        "TriggerSigma": row["TriggerSigma"],
        "RewardRisk": row["RewardRisk"],
        "EntryLag": int(row["EntryLag"]),
        "CalendarN": int(row["CalendarN"]),
        "Variant": variant,
    }
    for col in SWEEP_METRIC_COLS:
        if col in row.index:
            out[col] = row[col]
    return out


def summarize_monthly_patterns(by_month: pd.DataFrame, params: dict[str, object]) -> dict[str, float | int]:
    if by_month.empty:
        return {
            "AvgMonthlyWinRate": float("nan"),
            "MonthsAbove50PctWinRate": 0,
            "MonthsAllWins": 0,
            "MonthsAllLosses": 0,
            "MonthsMixed": 0,
            "WorstMonthNetPnL": float("nan"),
            "BestMonthNetPnL": float("nan"),
            "MaxMonthlyConsecutiveLosses": 0,
            "AvgTradesPerMonth": float("nan"),
        }

    mask = pd.Series(True, index=by_month.index)
    for key in PARAM_KEYS:
        if key in params and key in by_month.columns:
            mask &= by_month[key] == params[key]
    sub = by_month.loc[mask]
    if sub.empty:
        return summarize_monthly_patterns(pd.DataFrame(), params)

    win_rates = sub["WinRate"].astype(float)
    net_pnls = sub["NetPnL"].astype(float)
    wins = sub["Wins"].astype(int)
    losses = sub["Losses"].astype(int)
    trades = sub["Trades"].astype(int)
    return {
        "AvgMonthlyWinRate": float(win_rates.mean()),
        "MonthsAbove50PctWinRate": int((win_rates > 0.5).sum()),
        "MonthsAllWins": int(((losses == 0) & (trades > 0)).sum()),
        "MonthsAllLosses": int(((wins == 0) & (trades > 0)).sum()),
        "MonthsMixed": int(((wins > 0) & (losses > 0)).sum()),
        "WorstMonthNetPnL": float(net_pnls.min()),
        "BestMonthNetPnL": float(net_pnls.max()),
        "MaxMonthlyConsecutiveLosses": int(sub["MaxConsecutiveLosses"].max()),
        "AvgTradesPerMonth": float(sub["Trades"].mean()),
    }


def _month_pattern_stats(sub: pd.DataFrame) -> dict[str, float | int]:
    if sub.empty:
        return summarize_monthly_patterns(pd.DataFrame(), {})
    win_rates = sub["WinRate"].astype(float)
    net_pnls = sub["NetPnL"].astype(float)
    wins = sub["Wins"].astype(int)
    losses = sub["Losses"].astype(int)
    trades = sub["Trades"].astype(int)
    return {
        "MonthsWithTrades": int(len(sub)),
        "AvgMonthlyWinRate": float(win_rates.mean()),
        "MonthsAbove50PctWinRate": int((win_rates > 0.5).sum()),
        "MonthsAllWins": int(((losses == 0) & (trades > 0)).sum()),
        "MonthsAllLosses": int(((wins == 0) & (trades > 0)).sum()),
        "MonthsMixed": int(((wins > 0) & (losses > 0)).sum()),
        "WorstMonthNetPnL": float(net_pnls.min()),
        "BestMonthNetPnL": float(net_pnls.max()),
        "MaxMonthlyConsecutiveLosses": int(sub["MaxConsecutiveLosses"].max()),
        "AvgTradesPerMonth": float(sub["Trades"].mean()),
    }


def build_month_patterns_summary(by_month: pd.DataFrame) -> pd.DataFrame:
    if by_month.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for keys, grp in by_month.groupby(list(PARAM_KEYS), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        params = dict(zip(PARAM_KEYS, keys))
        rows.append({**params, **_month_pattern_stats(grp)})
    return pd.DataFrame(rows)


def load_reversion_medians(reversion_path: Path) -> dict[float, float]:
    if not reversion_path.is_file():
        return {}
    df = pd.read_csv(reversion_path)
    if df.empty or "MedianDurationDays" not in df.columns:
        return {}
    sub = df[df["Direction"] == "all"] if "Direction" in df.columns else df
    out: dict[float, float] = {}
    for _, row in sub.iterrows():
        try:
            k = float(row["TriggerSigma"])
            out[k] = float(row["MedianDurationDays"])
        except (TypeError, ValueError):
            continue
    return out


def enrich_sweep_from_trades(
    sweep: pd.DataFrame,
    trades: pd.DataFrame | None,
    account_value: float = DEFAULT_ACCOUNT_VALUE,
) -> pd.DataFrame:
    if sweep.empty:
        return sweep

    out = sweep.copy()
    acct = float(account_value) if account_value > 0 else DEFAULT_ACCOUNT_VALUE
    if trades is None or trades.empty:
        if "AvgWinPctOfAccount" not in out.columns:
            out["AvgWinPctOfAccount"] = float("nan")
        if "AvgLossPctOfAccount" not in out.columns:
            out["AvgLossPctOfAccount"] = float("nan")
        return out

    pnl_stats: list[dict[str, object]] = []
    for keys, grp in trades.groupby(list(PARAM_KEYS), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pnl = grp["ProfitOrLoss"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        pnl_stats.append(
            {
                **dict(zip(PARAM_KEYS, keys)),
                "AvgWinPctOfAccount": float(wins.mean() / acct) if len(wins) else float("nan"),
                "AvgLossPctOfAccount": float(abs(losses.mean()) / acct) if len(losses) else float("nan"),
            }
        )
    stats_df = pd.DataFrame(pnl_stats)
    out = out.drop(columns=["AvgWinPctOfAccount", "AvgLossPctOfAccount"], errors="ignore")
    return out.merge(stats_df, on=list(PARAM_KEYS), how="left")


def get_extended_delay_best_configs(sweep: pd.DataFrame, mode: Mode) -> pd.DataFrame:
    sub = sweep[sweep["Mode"] == mode].copy()
    if sub.empty:
        return pd.DataFrame()
    best = _best_per_trigger(sub)
    cols = ["TriggerSigma", "Mode", "EntryLag", "CalendarN"]
    return best[cols].drop_duplicates().reset_index(drop=True)


def build_delay_curve_summary(
    sweep: pd.DataFrame,
    *,
    mode: Mode,
    reversion_medians: dict[float, float] | None = None,
) -> pd.DataFrame:
    """Best CalendarN per representative EntryLag, using extended-best RewardRisk per trigger."""
    if sweep.empty:
        return pd.DataFrame()

    sub = sweep[sweep["Mode"] == mode].copy()
    if sub.empty:
        return pd.DataFrame()

    curve_lags = DELAY_CURVE_ENTRY_LAGS.get(mode, ())
    if not curve_lags:
        return pd.DataFrame()

    extended_best = _best_per_trigger(sub)
    rows: list[dict[str, object]] = []

    for _, best_row in extended_best.iterrows():
        trigger_sigma = float(best_row["TriggerSigma"])
        reward_risk = best_row["RewardRisk"]
        best_lag = int(best_row["EntryLag"])
        best_n = int(best_row["CalendarN"])

        lag_sub = sub[
            (sub["TriggerSigma"] == trigger_sigma) & (sub["RewardRisk"] == reward_risk)
        ]
        curve_points: list[pd.Series] = []
        for lag in curve_lags:
            at_lag = lag_sub[lag_sub["EntryLag"] == lag]
            if at_lag.empty:
                continue
            curve_points.append(
                at_lag.sort_values(["Sharpe", "NetPnL"], ascending=[False, False]).iloc[0]
            )

        if not curve_points:
            continue

        peak_sharpe = max(float(p["Sharpe"]) for p in curve_points if pd.notna(p["Sharpe"]))

        for point in curve_points:
            r = _row_to_summary(point, "delay_curve")
            lag = int(point["EntryLag"])
            cal_n = int(point["CalendarN"])
            sharpe = float(point["Sharpe"])
            r["CurveEntryLag"] = lag
            r["IsExtendedBest"] = lag == best_lag and cal_n == best_n
            r["IsCurvePeak"] = pd.notna(point["Sharpe"]) and abs(sharpe - peak_sharpe) < 1e-9
            r["SharpeVsCurvePeak"] = sharpe - peak_sharpe if pd.notna(point["Sharpe"]) else float("nan")
            if reversion_medians and trigger_sigma in reversion_medians:
                median_days = reversion_medians[trigger_sigma]
                r["MedianReversionDays"] = median_days
                r["CalendarNMinusMedianReversion"] = cal_n - median_days
            rows.append(r)

    return pd.DataFrame(rows)


def build_delay_comparison_summary(
    sweep: pd.DataFrame,
    by_month: pd.DataFrame,
    *,
    mode: Mode,
    reversion_medians: dict[float, float] | None = None,
) -> pd.DataFrame:
    if sweep.empty:
        return pd.DataFrame()

    sub = sweep[sweep["Mode"] == mode].copy()
    if sub.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    original = sub[sub.apply(_is_original_delay_row, axis=1)]
    new_only = sub[sub.apply(_is_new_delay_row, axis=1)]

    extended_best = _best_per_trigger(sub)
    original_best = _best_per_trigger(sub, original)
    new_best = _best_per_trigger(sub, new_only)

    orig_lookup = {
        (float(r["TriggerSigma"]),): (float(r["Sharpe"]), float(r["NetPnL"]))
        for _, r in original_best.iterrows()
    }

    for _, row in extended_best.iterrows():
        r = _row_to_summary(row, "extended_delay_best")
        key = (float(row["TriggerSigma"]),)
        if key in orig_lookup:
            r["SharpeVsOriginal"] = float(row["Sharpe"]) - orig_lookup[key][0]
            r["NetPnLVsOriginal"] = float(row["NetPnL"]) - orig_lookup[key][1]
        if reversion_medians:
            k = float(row["TriggerSigma"])
            if k in reversion_medians:
                median_days = reversion_medians[k]
                r["MedianReversionDays"] = median_days
                r["CalendarNMinusMedianReversion"] = int(row["CalendarN"]) - median_days
        rows.append(r)

    for _, row in original_best.iterrows():
        rows.append(_row_to_summary(row, "original_delay_best"))

    for _, row in new_best.iterrows():
        rows.append(_row_to_summary(row, "extended_delay_new_only"))

    for _, row in extended_best.iterrows():
        params = {k: row[k] for k in PARAM_KEYS}
        month_stats = summarize_monthly_patterns(by_month, params)
        r = _row_to_summary(row, "monthly_patterns")
        r.update(month_stats)
        rows.append(r)

    overall = sub.sort_values(["Sharpe", "NetPnL"], ascending=[False, False]).head(1)
    for _, row in overall.iterrows():
        rows.append(_row_to_summary(row, "overall_best"))

    return pd.DataFrame(rows)


def build_adaptive_comparison_summary(sweep_adaptive: pd.DataFrame, *, mode: Mode) -> pd.DataFrame:
    if sweep_adaptive.empty or "Variant" not in sweep_adaptive.columns:
        return pd.DataFrame()

    sub = sweep_adaptive[sweep_adaptive["Mode"] == mode].copy()
    if sub.empty:
        return pd.DataFrame()

    join_keys = ["TriggerSigma", "EntryLag", "CalendarN", "RewardRisk"]
    baseline = sub[sub["Variant"] == "baseline"].copy()
    adaptive = sub[sub["Variant"] == "adaptive"].copy()

    rows: list[dict[str, object]] = []
    for _, base_row in baseline.iterrows():
        match = adaptive[
            (adaptive["TriggerSigma"] == base_row["TriggerSigma"])
            & (adaptive["EntryLag"] == base_row["EntryLag"])
            & (adaptive["CalendarN"] == base_row["CalendarN"])
            & (adaptive["RewardRisk"] == base_row["RewardRisk"])
        ]
        if match.empty:
            continue
        adapt_row = match.iloc[0]

        for variant, row in (("baseline", base_row), ("adaptive", adapt_row)):
            r = _row_to_summary(row, "adaptive_vs_baseline", variant=variant)
            if variant == "adaptive":
                r["DeltaNetPnL"] = float(adapt_row["NetPnL"]) - float(base_row["NetPnL"])
                r["DeltaMaxDrawdown"] = float(adapt_row["MaxDrawdown"]) - float(base_row["MaxDrawdown"])
                r["DeltaSharpe"] = float(adapt_row["Sharpe"]) - float(base_row["Sharpe"])
                r["DeltaWinRate"] = float(adapt_row["WinRate"]) - float(base_row["WinRate"])
            else:
                r["DeltaNetPnL"] = 0.0
                r["DeltaMaxDrawdown"] = 0.0
                r["DeltaSharpe"] = 0.0
                r["DeltaWinRate"] = 0.0
            rows.append(r)

    return pd.DataFrame(rows)


def build_results_summary(
    extended_dir: Path,
    adaptive_dir: Path | None = None,
    *,
    reversion_path: Path | None = None,
    account_value: float = DEFAULT_ACCOUNT_VALUE,
    month_patterns_dir: Path | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    reversion_medians = load_reversion_medians(reversion_path) if reversion_path else {}

    for mode in ("probable_duration", "wait_entry"):
        sweep_path = extended_dir / mode / "sweep_summary.csv"
        by_month_path = extended_dir / mode / "by_month.csv"
        trades_path = extended_dir / mode / "trades.csv"
        if not sweep_path.is_file():
            continue
        sweep = pd.read_csv(sweep_path)
        trades = pd.read_csv(trades_path) if trades_path.is_file() else pd.DataFrame()
        sweep = enrich_sweep_from_trades(sweep, trades, account_value=account_value)
        by_month = pd.read_csv(by_month_path) if by_month_path.is_file() else pd.DataFrame()

        if month_patterns_dir is not None and not by_month.empty:
            month_summary = build_month_patterns_summary(by_month)
            if not month_summary.empty:
                out_mp = month_patterns_dir / mode
                out_mp.mkdir(parents=True, exist_ok=True)
                month_summary.to_csv(out_mp / "month_patterns_summary.csv", index=False)

        delay_df = build_delay_comparison_summary(
            sweep,
            by_month,
            mode=mode,  # type: ignore[arg-type]
            reversion_medians=reversion_medians or None,
        )
        if not delay_df.empty:
            frames.append(delay_df)

        curve_df = build_delay_curve_summary(
            sweep,
            mode=mode,  # type: ignore[arg-type]
            reversion_medians=reversion_medians or None,
        )
        if not curve_df.empty:
            if month_patterns_dir is not None:
                out_curve = month_patterns_dir / mode
                out_curve.mkdir(parents=True, exist_ok=True)
                curve_df.to_csv(out_curve / "delay_curve_summary.csv", index=False)
            frames.append(curve_df)

        if adaptive_dir is not None:
            adapt_path = adaptive_dir / mode / "sweep_summary_adaptive.csv"
            adapt_trades_path = adaptive_dir / mode / "trades_adaptive.csv"
            if adapt_path.is_file():
                adapt_sweep = pd.read_csv(adapt_path)
                adapt_trades = (
                    pd.read_csv(adapt_trades_path) if adapt_trades_path.is_file() else pd.DataFrame()
                )
                adapt_sweep = enrich_sweep_from_trades(
                    adapt_sweep, adapt_trades, account_value=account_value
                )
                adapt_df = build_adaptive_comparison_summary(adapt_sweep, mode=mode)  # type: ignore[arg-type]
                if not adapt_df.empty:
                    frames.append(adapt_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    section_order = [
        "extended_delay_best",
        "original_delay_best",
        "extended_delay_new_only",
        "delay_curve",
        "monthly_patterns",
        "overall_best",
        "adaptive_vs_baseline",
    ]
    out["_section_ord"] = out["SummarySection"].map({s: i for i, s in enumerate(section_order)})
    out = out.sort_values(
        ["_section_ord", "Mode", "TriggerSigma", "CurveEntryLag", "EntryLag", "Variant"]
    ).drop(columns=["_section_ord"])
    return out.reset_index(drop=True)


def print_results_summary_narrative(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    for mode in ("probable_duration", "wait_entry"):
        ext = summary[
            (summary["SummarySection"] == "extended_delay_best") & (summary["Mode"] == mode)
        ]
        orig = summary[
            (summary["SummarySection"] == "original_delay_best") & (summary["Mode"] == mode)
        ]
        new = summary[
            (summary["SummarySection"] == "extended_delay_new_only") & (summary["Mode"] == mode)
        ]
        for _, row in ext.iterrows():
            k = row["TriggerSigma"]
            orig_row = orig[orig["TriggerSigma"] == k]
            new_row = new[new["TriggerSigma"] == k]
            orig_sharpe = float(orig_row.iloc[0]["Sharpe"]) if len(orig_row) else float("nan")
            new_sharpe = float(new_row.iloc[0]["Sharpe"]) if len(new_row) else float("nan")
            print(
                f"Extended delays ({mode}, k={k:g}): best Sharpe={row['Sharpe']:.2f} "
                f"lag={int(row['EntryLag'])} N={int(row['CalendarN'])} | "
                f"original best={orig_sharpe:.2f} | new-delay best={new_sharpe:.2f}"
                + (
                    f" | median reversion={row['MedianReversionDays']:.1f}d "
                    f"(N−median={row['CalendarNMinusMedianReversion']:.1f})"
                    if "MedianReversionDays" in row.index and pd.notna(row["MedianReversionDays"])
                    else ""
                )
            )

        curve = summary[
            (summary["SummarySection"] == "delay_curve") & (summary["Mode"] == mode)
        ]
        for k in sorted(curve["TriggerSigma"].unique()):
            pts = curve[curve["TriggerSigma"] == k].sort_values("CurveEntryLag")
            if pts.empty:
                continue
            parts = []
            for _, p in pts.iterrows():
                marker = ""
                if "IsCurvePeak" in p.index and bool(p["IsCurvePeak"]):
                    marker = "*"
                if "IsExtendedBest" in p.index and bool(p["IsExtendedBest"]):
                    marker += "†"
                parts.append(
                    f"lag={int(p['CurveEntryLag'])} N={int(p['CalendarN'])} "
                    f"Sharpe={p['Sharpe']:.2f}{marker}"
                )
            print(f"Delay curve ({mode}, k={k:g}): " + " | ".join(parts))

    adapt = summary[
        (summary["SummarySection"] == "adaptive_vs_baseline") & (summary["Variant"] == "adaptive")
    ]
    base = summary[
        (summary["SummarySection"] == "adaptive_vs_baseline") & (summary["Variant"] == "baseline")
    ]
    if not adapt.empty and not base.empty:
        best_adapt = adapt.sort_values("Sharpe", ascending=False).iloc[0]
        key = (
            best_adapt["Mode"],
            best_adapt["TriggerSigma"],
            best_adapt["EntryLag"],
            best_adapt["CalendarN"],
            best_adapt["RewardRisk"],
        )
        base_match = base[
            (base["Mode"] == key[0])
            & (base["TriggerSigma"] == key[1])
            & (base["EntryLag"] == key[2])
            & (base["CalendarN"] == key[3])
            & (base["RewardRisk"] == key[4])
        ]
        if len(base_match):
            b = base_match.iloc[0]
            mdd_pct = (
                (best_adapt["MaxDrawdown"] - b["MaxDrawdown"]) / b["MaxDrawdown"] * 100
                if b["MaxDrawdown"]
                else float("nan")
            )
            print(
                f"Adaptive vs baseline ({key[0]}): MDD ${b['MaxDrawdown']:,.0f} -> "
                f"${best_adapt['MaxDrawdown']:,.0f} ({mdd_pct:+.0f}%) | "
                f"NetPnL ${b['NetPnL']:,.0f} -> ${best_adapt['NetPnL']:,.0f}"
            )
