#!/usr/bin/env python3
"""
GDELT Gold-Price Headline Extraction  —  Research Task 1
========================================================
OBJECTIVE
  Produce a verified dataset of English-language gold-PRICE news headlines over a
  ~3-month window (Corpus A), each paired to its exact publication timestamp and
  source, then flag the subset containing Rev 15 keyword terms (Corpus B) via an
  added column. Outputs: corpus_a.csv + coverage_log.csv. (No sentiment scoring
  in this task -- that comes after this output is verified.)
 
SOURCE
  GDELT DOC 2.0 API, ArtList (article-list) mode. Free, no key. ArtList reliably
  covers the most recent ~3 months, which is exactly this window -- so no BigQuery
  is needed. ArtList returns the EXACT headline (title) per article, so no
  URL-to-title recovery is needed.
 
PIPELINE (steps map to the research plan)
  Step 2  compose + call ArtList (fetch layer)
  Step 3  page day-by-day (maxrecords caps at 250/call)
  Step 4  collect raw rows + coverage log (quiet vs failed dates)
  Step 5  Corpus A: exclude Indian gold news (headline tokens + Indian outlets); tag duplicates (no drop)
  Step 6  normalize timestamps + sort
  Corpus B  flag headlines containing Rev 15 terms (kw_flag/kw_terms columns)
  Step 7  write corpus_a.csv
          [date, timestamp, headline, domain, url, dupe_group, kw_flag, kw_terms, article_text]
  Step 8  verification report (coverage, daily counts, exclusions, Corpus B size)
 
USAGE
  pip install requests pandas
  python headline_et.py                                   # default: END_DATE window
  python headline_et.py --start 2026-04-13 --end 2026-07-13
  python headline_et.py --step-hours 12                   # finer slices if a day caps out
"""
import argparse, time, sys, re, os
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
 
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Extraction window (inclusive calendar dates, UTC). Change these to move the window.
START_DATE = "2026-08-09"
END_DATE = "2026-08-16"

QUERY = '("gold price" OR "gold prices" OR "gold bullion") sourcelang:english'

GOLD_TERMS = ["gold", "bullion", "xau"]      # for Step-8 keyword sanity check

# ---- Rev 15 keyword dictionary (Memo_Vrinda_HCGSS_Rev15.docx): 23 bullish + 23 bearish ----
REV15_TERMS = [
    # bullish
    "dovish fed", "weakening dollar", "weak dollar", "weaker dollar", "quantitative easing",
    "safe haven", "stimulus", "weakening us dollar", "recession", "inverted yield curve",
    "falling real yields", "falling yields", "hard landing", "rate cut", "falling real 10-year yields",
    "jobs slowdown", "hiring slowdown", "labor market cooling", "weak payrolls", "rising jobless claims",
    "falling energy costs", "falling gasoline prices", "lower inflation",
    # bearish
    "hawkish fed", "central bank selling", "tightening", "strong dollar", "rate hike",
    "rising yields", "higher yields", "soft landing", "rising real 10-year yields", "economy overheating",
    "strong payrolls", "strong jobs report", "falling jobless claims", "hiring surge", "strong hiring",
    "tight labor market", "strong gdp growth", "resilient economy", "resilient labor market",
    "accelerating wage growth", "rising energy costs", "rising gasoline prices", "higher inflation",
]

# Corpus A exclusions: Indian gold news (headline tokens + Indian outlets).
# Word-boundary except ₹. indians closes the Indian/Indians hole; rupee/INR/₹/Sensex/
# Nifty/Mumbai/Delhi/Chennai/Pune/Hyderabad catch leftover India-price and India-market copy.
EXCLUDE_HEADLINE_RE = re.compile(
    r"(?:\b(?:india|indians?|rs|mcx|rupee|inr|sensex|nifty|mumbai|delhi|chennai|pune|hyderabad)\b|₹)",
    re.IGNORECASE,
)
# Memo 2.4 domains plus named leftovers whose host does not contain "india".
# Hosts containing "india" and *.in / *.co.in are also dropped in is_excluded_domain.
EXCLUDE_DOMAINS = (
    "indiatimes", "moneycontrol", "thehindubusinessline", "livemint",
    "businesstoday", "financialexpress", "zeebiz",
    "thehindu.com", "deccanchronicle", "business-standard", "hindustantimes",
    "freepressjournal", "navhindtimes", "heraldgoa",
)


def is_excluded_domain(domain):
    """True if the source is an Indian outlet (2.4 list, named leftovers, .in, or 'india' in host)."""
    d = (domain or "").lower()
    if not d:
        return False
    if "india" in d:
        return True
    if d.endswith(".in") or ".co.in" in d:
        return True
    return any(token in d for token in EXCLUDE_DOMAINS)


# ---- Shared matching rule (memo section 3): exact-phrase, word-boundary, case-insensitive ----
# This is the SINGLE source of truth for term matching. The future scorer MUST reuse
# match_rev15_terms so a flagged headline can never score zero from a rule mismatch.
_REV15_RES = [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)) for t in REV15_TERMS]


def match_rev15_terms(headline):
    """Return the Rev15 terms present in the headline as exact whole phrases.

    A term matches only as a complete phrase ("rate hike" does not match "hike the
    rate"). Reused by both Corpus B flagging and any downstream scoring step.
    """
    h = headline or ""
    return [t for t, rx in _REV15_RES if rx.search(h)]


# ----------------------------- Step 2/3: fetch -----------------------------
def fetch_page(win_start, win_end, session, retries=8):
    params = {
        "query": QUERY, "mode": "ArtList", "format": "json",
        "maxrecords": 250, "sort": "DateAsc",
        "startdatetime": win_start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   win_end.strftime("%Y%m%d%H%M%S"),
    }
    for attempt in range(retries):
        try:
            r = session.get(GDELT, params=params, timeout=60)
            if r.status_code == 429:
                # GDELT rate limit: honor Retry-After if present, else exponential backoff
                wait = int(r.headers.get("Retry-After", 0)) or min(60, 5 * (2 ** attempt))
                print(f"  ~ {win_start:%Y-%m-%d %H:%M} rate-limited (429), "
                      f"sleeping {wait}s (attempt {attempt + 1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            arts = r.json().get("articles", [])
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! {win_start:%Y-%m-%d %H:%M} failed: {e}", file=sys.stderr)
                return [], 0, False
            wait = min(60, 3 * (2 ** attempt))
            time.sleep(wait)
            continue
        rows = [{
            "seendate": a.get("seendate"), "headline": a.get("title"),
            "domain": a.get("domain"), "url": a.get("url"),
            "language": a.get("language"), "sourcecountry": a.get("sourcecountry"),
        } for a in arts]
        return rows, len(arts), True
    print(f"  ! {win_start:%Y-%m-%d %H:%M} failed: still rate-limited after {retries} retries",
          file=sys.stderr)
    return [], 0, False
 
 
def fetch_all(start, end, step_hours=24, sleep_s=5.0):
    session = requests.Session()
    session.headers.update({"User-Agent": "gold-headline-extraction/1.0"})
    all_rows, cur, capped_slices, coverage = [], start, [], []
    step = timedelta(hours=step_hours)
    while cur < end:
        nxt = min(cur + step, end)
        page, n, ok = fetch_page(cur, nxt, session)
        if n >= 250:                                     # Step 3 guard: possible truncation
            capped_slices.append(cur.strftime("%Y-%m-%d %H:%M"))
        # Step 2.7 coverage log: record every slice queried so a genuinely quiet day
        # (status=ok, raw_count=0) is distinguishable from a failed/missing one.
        coverage.append({
            "date": cur.date(),
            "window_start": cur.strftime("%Y-%m-%d %H:%M"),
            "window_end": nxt.strftime("%Y-%m-%d %H:%M"),
            "status": "ok" if ok else "failed",
            "raw_count": n,
        })
        status = "ok" if ok else "FAILED"
        print(f"  {cur:%Y-%m-%d %H:%M}: {n} rows [{status}]"
              + ("  [CAPPED -> use finer --step-hours]" if n >= 250 else ""))
        all_rows += page
        cur = nxt
        time.sleep(sleep_s)                              # GDELT throttles aggressive polling
    if capped_slices:
        print(f"\n  WARNING: {len(capped_slices)} slice(s) hit the 250 cap and may be truncated:")
        for s in capped_slices[:10]:
            print(f"    {s}")
        print("  Re-run those windows with a smaller --step-hours (e.g. 6) to capture all.")
    return pd.DataFrame(all_rows), pd.DataFrame(coverage)
 
 
# --------------------- Step 5/6: normalize + group -------------------------
def _norm_key(h):
    if not isinstance(h, str):
        return ""
    return re.sub(r"[^a-z0-9 ]", "", h.lower()).strip()
 
 
def normalize_and_group(raw):
    """Build Corpus A: drop Indian gold news, normalize timestamps, and tag syndicated
    duplicates with a shared dupe_group id WITHOUT dropping any rows (memo 2.6)."""
    df = raw.copy()
    # Step 6: parse GDELT 'seendate' (e.g. 20260415T131500Z) to real UTC datetime
    df["timestamp"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ",
                                     utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "headline"])
    df["date"] = df["timestamp"].dt.date

    # Corpus A exclusions: Indian gold news (headline tokens + Indian outlets)
    domain_mask = df["domain"].fillna("").map(is_excluded_domain)
    head_mask = df["headline"].fillna("").map(lambda h: bool(EXCLUDE_HEADLINE_RE.search(h)))
    excluded = int((domain_mask | head_mask).sum())
    df = df[~(domain_mask | head_mask)]

    df["_key"] = df["headline"].map(_norm_key)
    df = df[df["_key"] != ""].sort_values("timestamp")   # earliest first

    # Step 5 (memo 2.6): keep syndicated duplicates as separate rows; only TAG them.
    df["dupe_group"] = df.groupby(["date", "_key"]).ngroup()
    dup_rows = int(df.duplicated(subset=["date", "_key"], keep="first").sum())

    df = df.drop(columns=["_key", "seendate"]).reset_index(drop=True)
    return df, excluded, dup_rows


def flag_keywords(df):
    """Populate Corpus B columns via the shared exact-phrase matcher.

    kw_flag == 1 marks the Corpus B subset. article_text is left empty in this
    extraction pass (ArtList returns no body text; collected in a later step).
    """
    hits = df["headline"].map(match_rev15_terms)
    df["kw_terms"] = hits.map(lambda xs: "; ".join(xs))
    df["kw_flag"] = hits.map(lambda xs: 1 if xs else 0)
    df["article_text"] = pd.NA
    return df
 
 
# --------------------------- Step 8: verify -------------------------------
def verify(df, raw_n, excluded, dup_rows, coverage, start, end):
    print("\n================ STEP 8 — VERIFICATION ================")
    print(f"  raw rows fetched          : {raw_n}")
    print(f"  Corpus A excluded rows    : {excluded}  (Indian gold news: headline tokens + Indian outlets)")
    print(f"  Corpus A rows (kept)      : {len(df)}")
    print(f"  syndicated dup rows kept  : {dup_rows}  across {df['dupe_group'].nunique()} dupe_groups")
    print(f"  Corpus B flagged (kw_flag): {int(df['kw_flag'].sum())}"
          + (f"  ({df['kw_flag'].mean():.1%})" if len(df) else ""))

    # (b) coverage: distinguish genuinely quiet days from failed/missing collection
    all_days = pd.date_range(start.date(), (end - timedelta(days=1)).date(), freq="D").date
    have = set(df["date"])
    missing = [d for d in all_days if d not in have]
    print(f"  calendar days in window   : {len(all_days)}")
    print(f"  days WITH headlines       : {len(have)}")
    if len(coverage):
        failed_days = sorted(set(coverage.loc[coverage["status"] == "failed", "date"]))
        quiet_days = sorted(set(coverage.loc[(coverage["status"] == "ok") & (coverage["raw_count"] == 0), "date"]))
        print(f"  days FAILED (collection)  : {len(failed_days)}  (missing != quiet; re-run these)")
        if failed_days:
            print("     e.g. " + ", ".join(str(d) for d in failed_days[:8]))
        print(f"  days genuinely QUIET (0)  : {len(quiet_days)}  (status=ok, raw_count=0)")
        if quiet_days:
            print("     e.g. " + ", ".join(str(d) for d in quiet_days[:8]))
    elif missing:
        print(f"  days with ZERO headlines  : {len(missing)}  (check weekends/holidays vs gaps)")
        print("     e.g. " + ", ".join(str(d) for d in missing[:8]))

    # (c) daily counts sanity (weekday vs weekend)
    per_day = df.groupby("date").size()
    if len(per_day):
        wd = df[df["timestamp"].dt.dayofweek < 5].groupby("date").size().mean()
        we = df[df["timestamp"].dt.dayofweek >= 5].groupby("date").size().mean()
        print(f"  avg rows/weekday          : {wd:.1f}")
        print(f"  avg rows/weekend-day      : {we:.1f}   (weekday should exceed weekend)")

    # (d) keyword sanity: share of headlines that actually mention a gold term
    hit = df["headline"].str.lower().str.contains("|".join(GOLD_TERMS), na=False).mean()
    print(f"  headlines mentioning gold : {hit:.1%}  (low => tighten the query)")
    print("=======================================================")
    print("  MANUAL CHECK still required: eyeball ~20 headlines for genuine gold-PRICE")
    print("  relevance (not just any 'gold'), and confirm Corpus B flags look right.")
 
 
OUT_COLS = ["date", "timestamp", "headline", "domain", "url",
            "dupe_group", "kw_flag", "kw_terms", "article_text"]


def append_corpus(path, new_df):
    """Append new Corpus A rows onto an existing file without overwriting.

    Existing rows are kept as-is. New rows whose URL already exists are skipped.
    dupe_group ids on new rows are offset so they do not collide with the old file.
    """
    new_df = new_df[OUT_COLS].copy()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return new_df, len(new_df)
    old = pd.read_csv(path)
    existing_urls = set(old["url"].dropna().astype(str))
    added = new_df[~new_df["url"].astype(str).isin(existing_urls)].copy()
    offset = int(old["dupe_group"].max()) + 1 if len(old) and old["dupe_group"].notna().any() else 0
    if len(added):
        added["dupe_group"] = added["dupe_group"] + offset
    combined = pd.concat([old, added], ignore_index=True)
    return combined, len(added)


def append_coverage(path, new_cov):
    """Append coverage slices; never overwrite an existing coverage log."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        old = pd.read_csv(path)
        combined = pd.concat([old, new_cov], ignore_index=True)
    else:
        combined = new_cov
    combined.to_csv(path, index=False)
    return combined


# --------------------------------- main -----------------------------------
def main():
    default_start = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    default_end = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=default_start.strftime("%Y-%m-%d"))
    ap.add_argument("--end",   default=default_end.strftime("%Y-%m-%d"),
                    help="inclusive last calendar day to collect (UTC)")
    ap.add_argument("--step-hours", type=int, default=24)
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="seconds to wait between GDELT requests (default 5; raise if 429s persist)")
    ap.add_argument("--out", default="corpus_a.csv")
    ap.add_argument("--coverage-out", default="coverage_log.csv",
                    help="per-slice collection log (distinguishes quiet vs failed dates)")
    ap.add_argument("--keep-raw", action="store_true", help="also write headlines_raw.csv")
    a = ap.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%d")
    last_day = datetime.strptime(a.end, "%Y-%m-%d")
    end = last_day + timedelta(days=1)                   # fetch_all is exclusive of end
 
    print(f"[Steps 2-4] Fetching ArtList  {start:%Y-%m-%d} -> {last_day:%Y-%m-%d} (inclusive)  (sleep={a.sleep}s)")
    raw, coverage = fetch_all(start, end, step_hours=a.step_hours, sleep_s=a.sleep)
    cov_all = append_coverage(a.coverage_out, coverage)
    print(f"  coverage log -> {a.coverage_out}  ({len(coverage)} new slices, {len(cov_all)} total)")
    if a.keep_raw:
        raw.to_csv("headlines_raw.csv", index=False)
    if raw.empty:
        print("No rows returned. Check the window is within the last ~3 months (ArtList limit).")
        return
 
    print("[Steps 5-6] Normalize + tag duplicates (Corpus A) ...")
    df, excluded, dup_rows = normalize_and_group(raw)
 
    print("[Corpus B] Flagging Rev15 keyword terms ...")
    df = flag_keywords(df)
 
    print("[Step 7] Appending to existing corpus (not overwriting) ...")
    combined, n_added = append_corpus(a.out, df)
    combined.to_csv(a.out, index=False)
 
    verify(df, len(raw), excluded, dup_rows, coverage, start, end)
    print(f"\nDONE -> {a.out}  (+{n_added} new rows this run, {len(combined)} Corpus A total, "
          f"{int(df['kw_flag'].sum())} of this run flagged as Corpus B)")
 
if __name__ == "__main__":
    main()
 
