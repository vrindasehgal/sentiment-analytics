#!/usr/bin/env python3
"""
Retry only the failed collection slices recorded in coverage_log.csv.

This script uses coverage_log.csv as the source of truth for which windows failed,
then re-fetches only those slices from GDELT and appends recovered rows to
corpus_a.csv. Matching, exclusions, and append behavior are reused from
headline_extraction.py so retries stay identical to the main extraction pipeline.
"""
import argparse
import time
from datetime import datetime

import pandas as pd
import requests

from headline_extraction import (
    START_DATE,
    END_DATE,
    fetch_page,
    normalize_and_group,
    flag_keywords,
    append_corpus,
    append_coverage,
)


DEFAULT_RETRY_START = "2026-05-10"
DEFAULT_RETRY_END = END_DATE


def latest_failed_slices(coverage_path, start_date, end_date):
    """Return the latest failed slices within the inclusive date window."""
    coverage = pd.read_csv(coverage_path)
    coverage["date"] = pd.to_datetime(coverage["date"]).dt.date
    mask = (coverage["date"] >= start_date.date()) & (coverage["date"] <= end_date.date())
    scoped = coverage.loc[mask].copy()
    if scoped.empty:
        return scoped.iloc[0:0]

    # Keep the latest status for each original window so previously recovered
    # slices are not retried again.
    scoped = scoped.reset_index(names="_row_order")
    latest = (
        scoped.sort_values("_row_order")
        .groupby("window_start", as_index=False)
        .tail(1)
        .sort_values("date")
        .reset_index(drop=True)
    )
    return latest.loc[latest["status"] == "failed"].copy()


def retry_failed_slices(failed, sleep_s):
    """Retry each failed slice and return raw rows plus new coverage rows."""
    session = requests.Session()
    session.headers.update({"User-Agent": "gold-headline-extraction/1.0"})

    all_rows = []
    coverage_rows = []
    recovered_dates = []
    still_failed_dates = []

    for idx, row in failed.reset_index(drop=True).iterrows():
        win_start = datetime.strptime(row["window_start"], "%Y-%m-%d %H:%M")
        win_end = datetime.strptime(row["window_end"], "%Y-%m-%d %H:%M")
        page, n, ok = fetch_page(win_start, win_end, session)
        coverage_rows.append({
            "date": row["date"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "status": "ok" if ok else "failed",
            "raw_count": n,
        })
        status = "ok" if ok else "FAILED"
        print(f"  retry {win_start:%Y-%m-%d %H:%M}: {n} rows [{status}]")
        if ok:
            all_rows.extend(page)
            recovered_dates.append(str(row["date"]))
        else:
            still_failed_dates.append(str(row["date"]))

        if idx < len(failed) - 1:
            time.sleep(sleep_s)

    return pd.DataFrame(all_rows), pd.DataFrame(coverage_rows), recovered_dates, still_failed_dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_RETRY_START,
                    help=f"inclusive first calendar day to inspect in coverage_log.csv (default {DEFAULT_RETRY_START})")
    ap.add_argument("--end", default=DEFAULT_RETRY_END,
                    help=f"inclusive last calendar day to inspect in coverage_log.csv (default {DEFAULT_RETRY_END})")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="seconds to wait between GDELT requests (default 5)")
    ap.add_argument("--out", default="corpus_a.csv")
    ap.add_argument("--coverage-out", default="coverage_log.csv")
    a = ap.parse_args()

    start = datetime.strptime(a.start, "%Y-%m-%d")
    end = datetime.strptime(a.end, "%Y-%m-%d")

    print(f"[Retry] Inspecting failed slices in {start:%Y-%m-%d} -> {end:%Y-%m-%d} (inclusive)")
    failed = latest_failed_slices(a.coverage_out, start, end)
    if failed.empty:
        print("No failed slices found in that window. Nothing to retry.")
        return

    print(f"  found {len(failed)} failed slice(s) to retry")
    for _, row in failed.iterrows():
        print(f"    {row['date']}  {row['window_start']} -> {row['window_end']}")

    raw, new_coverage, recovered_dates, still_failed_dates = retry_failed_slices(failed, a.sleep)
    all_coverage = append_coverage(a.coverage_out, new_coverage)
    print(f"  coverage log -> {a.coverage_out}  (+{len(new_coverage)} rows, {len(all_coverage)} total)")

    if raw.empty:
        print("No successful retries produced rows.")
    else:
        print("[Retry] Normalize + tag duplicates ...")
        df, excluded, dup_rows = normalize_and_group(raw)
        print("[Retry] Flagging Rev15 keyword terms ...")
        df = flag_keywords(df)
        combined, n_added = append_corpus(a.out, df)
        combined.to_csv(a.out, index=False)
        print(f"  corpus -> {a.out}  (+{n_added} new rows, {len(combined)} total)")
        print(f"  retry verification: raw={len(raw)}, kept={len(df)}, excluded={excluded}, dup_rows={dup_rows}, kw_flag={int(df['kw_flag'].sum())}")

    if recovered_dates:
        print("  recovered dates: " + ", ".join(recovered_dates))
    if still_failed_dates:
        print("  still failed: " + ", ".join(still_failed_dates))


if __name__ == "__main__":
    main()
