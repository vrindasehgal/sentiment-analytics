# Gold headline extraction

This folder builds a checked set of English-language gold-price headlines (Corpus A) and flags the subset that contains a fixed list of bullish and bearish phrases (Corpus B). Corpus B is not a second file. It is `kw_flag == 1` on `corpus_a.csv`.

There is no sentiment score here. The output is the headline, the time GDELT recorded it, the source, and the phrase flag.

The committed corpus has **6,557 headlines** from **10 May 2026 through 18 Oct 2026**, across **821 domains**. **223** of them (3.4%) match at least one phrase. **5,064** duplicate-groups cover those rows: syndicated copies of the same headline on the same day are kept and share an id.

## Source

[GDELT DOC 2.0](https://www.gdeltproject.org/) ArtList, JSON, no API key. The query is:

```text
("gold price" OR "gold prices" OR "gold bullion") sourcelang:english
```

ArtList returns the headline, domain, URL, and a `seendate`. It does not return the article body, so `article_text` is empty. Each call is capped at 250 rows. The free endpoint wants about one request every 5 seconds. Faster polling returns `429 Too Many Requests`. Both scripts sleep between calls (`--sleep`, default 5) and back off on 429.

ArtList is a recent window, on the order of the last few months. The file in this folder was built by appending many runs. Running the script again does not rebuild that file from scratch, and a date far outside the current ArtList window may come back empty.

## What gets dropped, and what gets kept

Corpus A drops Indian gold-market copy and then keeps everything else, including syndicated duplicates.

Headline exclusions are case-insensitive. `india`, `indian`, `indians`, `rs` as a whole word, `mcx`, `rupee`, `inr`, `sensex`, `nifty`, `mumbai`, `delhi`, `chennai`, `pune`, `hyderabad`, and the rupee sign `₹`.

Domain exclusions:

- Host contains `india`, ends in `.in`, or contains `.co.in`
- Or the host contains one of: `indiatimes`, `moneycontrol`, `thehindubusinessline`, `livemint`, `businesstoday`, `financialexpress`, `zeebiz`, `thehindu.com`, `deccanchronicle`, `business-standard`, `hindustantimes`, `freepressjournal`, `navhindtimes`, `heraldgoa`

The exact patterns are `EXCLUDE_HEADLINE_RE` and `EXCLUDE_DOMAINS` in `headline_extraction.py`.

Duplicates stay. Two rows with the same headline on the same calendar day (punctuation and case ignored) share a `dupe_group`. They are not collapsed. A later run skips a URL that is already in the file, and it offsets new `dupe_group` ids so they do not collide with old ones.

## Phrase flag

`REV15_TERMS` in `headline_extraction.py` is 23 bullish phrases and 23 bearish phrases (`dovish fed`, `rate hike`, `safe haven`, and the rest of that list). `match_rev15_terms()` is the only matcher: exact phrase, word boundaries, case-insensitive. `"rate hike"` does not match `"hike the rate"`.

| Column | Meaning |
|--------|---------|
| `kw_flag` | `1` if any phrase matched, else `0`. This is Corpus B. |
| `kw_terms` | The matched phrases, joined with `"; "`. |

## Files

### `corpus_a.csv`

| Column | Meaning |
|--------|---------|
| `date` | Calendar date of `timestamp` (UTC) |
| `timestamp` | GDELT `seendate`, parsed as UTC |
| `headline` | Article title from ArtList |
| `domain` | Source host |
| `url` | Article URL. Used to skip duplicates on later runs. |
| `dupe_group` | Shared id for the same headline on the same day |
| `kw_flag` | Corpus B flag |
| `kw_terms` | Matched phrases |
| `article_text` | Always empty in this pass |

### `coverage_log.csv`

One row per slice that was requested. New rows are appended. An old `failed` row is left in place, so the log is a history. The row that matters is the **latest** row for a given `window_start`.

| Latest status | Meaning |
|---------------|---------|
| `ok` and `raw_count > 0` | That slice returned articles |
| `ok` and `raw_count == 0` | The request succeeded and the slice was empty |
| `failed` | The request failed, usually a 429. This is not an empty day. |

The committed log has 199 rows and 161 distinct windows. Every window’s latest status is `ok`. 21 of those windows returned zero articles.

## How to run

From this folder:

```bash
pip install -r requirements.txt
python headline_extraction.py --start 2026-05-10 --end 2026-05-16
python retry_failed.py --start 2026-05-10 --end 2026-05-16
```

`headline_extraction.py` with no dates uses the constants at the top of the file, currently `2026-08-09` through `2026-08-16`. `--start` and `--end` are inclusive calendar days, UTC. The script fetches through the end of `--end`.

| Flag | Default | Notes |
|------|---------|--------|
| `--start` | `START_DATE` | First day |
| `--end` | `END_DATE` | Last day |
| `--sleep` | `5` | Seconds between GDELT calls |
| `--step-hours` | `24` | Slice length. Use `12` or `6` if a day hits the 250-row cap. |
| `--out` | `corpus_a.csv` | Appended |
| `--coverage-out` | `coverage_log.csv` | Appended |
| `--keep-raw` | off | Also write `headlines_raw.csv` before the filters |

`retry_failed.py` reads the coverage log, keeps the latest row per `window_start` inside the date window, and re-fetches only the ones whose latest status is `failed`. A recovered day is then `ok` and is skipped on the next retry. It uses the same fetch, filters, and append helpers as `headline_extraction.py`.

| Flag | Default | Notes |
|------|---------|--------|
| `--start` | `2026-05-10` | First day to inspect in the log |
| `--end` | `END_DATE` in `headline_extraction.py` | Last day to inspect. This is the extractor constant, not the last date in the CSV. |
| `--sleep` | `5` | Seconds between calls |
| `--out` | `corpus_a.csv` | Appended |
| `--coverage-out` | `coverage_log.csv` | Appended. Each attempt adds an `ok` or `failed` row. |

Typical pass: run the extractor for a window, open `coverage_log.csv`, and run `retry_failed.py` on that same window until the latest status for each slice is `ok` or you stop. Then Corpus A is every row in `corpus_a.csv`, and Corpus B is `kw_flag == 1`.
