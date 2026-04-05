import pandas as pd
import yfinance as yf
import requests
import io
import time
import logging
import os
import smtplib
from collections import deque
from datetime import datetime
from curl_cffi import requests as curl_requests
from email.message import EmailMessage
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Global Session for Browser Impersonation ───────────────────────────────────
# Used ONLY for direct HTTP requests (e.g. JPX download), NOT passed to yfinance.
SESSION = curl_requests.Session(impersonate="chrome")


# ── Request Tracker ────────────────────────────────────────────────────────────
# Keeps a rolling timestamp log of every yfinance call so we can reason about
# which rate-limit window we're likely in when a 429 hits.
class RequestTracker:
    """
    Logs timestamps of every outgoing yfinance request.
    On a 429, we inspect how many requests we've made in the last 60 minutes
    to decide whether to do a short back-off (burst limit) or wait out the
    remainder of the hourly window.

    Hard cap: if the computed wait exceeds MAX_HOURLY_WAIT_SECS we give up
    on the chunk rather than hanging indefinitely.
    """
    MINUTE_WINDOW   = 60          # seconds
    HOURLY_WINDOW   = 3600        # seconds
    MAX_HOURLY_WAIT = 90 * 60     # 90 min — beyond this something is wrong; bail

    # Thresholds: if we've made this many calls in the window we assume that
    # window's limit is exhausted. Yahoo's exact numbers aren't published so
    # these are conservative estimates derived from community observation.
    BURST_THRESHOLD  = 40   # calls per minute that suggest a burst-limit hit
    HOURLY_THRESHOLD = 200  # calls per hour that suggest an hourly-limit hit

    def __init__(self):
        self._log: deque[float] = deque()

    def record(self):
        """Call this immediately before every yfinance request."""
        self._log.append(time.time())

    def _prune(self):
        cutoff = time.time() - self.HOURLY_WINDOW
        while self._log and self._log[0] < cutoff:
            self._log.popleft()

    def calls_in_last(self, seconds: float) -> int:
        cutoff = time.time() - seconds
        return sum(1 for t in self._log if t >= cutoff)

    def compute_wait(self, consecutive_429s: int) -> float | None:
        """
        Return how many seconds to sleep, or None if we should give up.

        Decision tree:
          1. If hourly window looks saturated → sleep until oldest hourly
             request ages out (i.e. the window resets).
          2. Else if burst window looks saturated → short exponential back-off.
          3. Else (first 429, low request count) → assume transient; short back-off.
        """
        self._prune()
        now = time.time()

        hourly_count = self.calls_in_last(self.HOURLY_WINDOW)
        burst_count  = self.calls_in_last(self.MINUTE_WINDOW)

        if hourly_count >= self.HOURLY_THRESHOLD or consecutive_429s >= 4:
            # Almost certainly an hourly limit — wait until the window resets.
            if self._log:
                oldest_in_window = self._log[0]
                wait = (oldest_in_window + self.HOURLY_WINDOW) - now + 5  # +5s buffer
            else:
                wait = self.HOURLY_WINDOW  # fallback: wait the full hour

            if wait > self.MAX_HOURLY_WAIT:
                log.error(
                    "Computed hourly wait of %.0fs exceeds safety cap (%.0fs). "
                    "Giving up on this chunk to avoid hanging indefinitely.",
                    wait, self.MAX_HOURLY_WAIT,
                )
                return None  # caller should skip this chunk

            log.warning(
                "Hourly rate limit detected (%d calls in last 60 min, %d consecutive 429s). "
                "Sleeping %.0fs until window resets...",
                hourly_count, consecutive_429s, wait,
            )
            return wait

        else:
            # Burst / transient limit — exponential back-off starting at 10s.
            wait = min(10 * (2 ** (consecutive_429s - 1)), 300)
            log.warning(
                "Burst rate limit detected (%d calls in last 60s, %d consecutive 429s). "
                "Sleeping %ds before retry...",
                burst_count, consecutive_429s, wait,
            )
            return wait


# One shared tracker for the entire process lifetime.
TRACKER = RequestTracker()


# ── Rate-limit detection ───────────────────────────────────────────────────────
_RL_KEYWORDS = ("429", "too many requests", "rate limit", "throttl")

def _is_rate_limit(exc: Exception) -> bool:
    return any(kw in str(exc).lower() for kw in _RL_KEYWORDS)


# ── yf.download wrapper ────────────────────────────────────────────────────────
def yf_download_with_retry(tickers, max_non_rl_attempts=3, **kwargs):
    """
    Wraps yf.download() with tiered, tracker-aware retry logic.
    Never passes a custom session to yfinance.

    - Rate-limit errors: defer to RequestTracker.compute_wait().
    - Other errors: up to max_non_rl_attempts then give up.
    """
    consecutive_429s = 0
    non_rl_attempts  = 0

    while True:
        TRACKER.record()
        try:
            data = yf.download(tickers, progress=False, **kwargs)
            consecutive_429s = 0   # reset on success
            return data

        except Exception as e:
            if _is_rate_limit(e):
                consecutive_429s += 1
                wait = TRACKER.compute_wait(consecutive_429s)
                if wait is None:
                    return pd.DataFrame()   # give up — tracker said so
                time.sleep(wait)

            else:
                non_rl_attempts += 1
                log.error("yf.download non-rate-limit error (attempt %d): %s", non_rl_attempts, e)
                if non_rl_attempts >= max_non_rl_attempts:
                    return pd.DataFrame()
                time.sleep(5)


# ── yf.Ticker wrapper ─────────────────────────────────────────────────────────
def yf_ticker_with_retry(ticker_symbol):
    """Returns a _RetryTicker that shares the global TRACKER."""
    return _RetryTicker(ticker_symbol)


class _RetryTicker:
    """Transparent wrapper around yf.Ticker with tracker-aware retry."""

    def __init__(self, symbol):
        self._symbol = symbol
        self._ticker = yf.Ticker(symbol)   # no session= argument

    def _fetch(self, attr):
        consecutive_429s = 0
        non_rl_attempts  = 0

        while True:
            TRACKER.record()
            try:
                result = getattr(self._ticker, attr)
                consecutive_429s = 0
                return result

            except Exception as e:
                if _is_rate_limit(e):
                    consecutive_429s += 1
                    wait = TRACKER.compute_wait(consecutive_429s)
                    if wait is None:
                        return pd.DataFrame()
                    time.sleep(wait)

                else:
                    non_rl_attempts += 1
                    log.error(
                        "Error fetching %s.%s (attempt %d): %s",
                        self._symbol, attr, non_rl_attempts, e,
                    )
                    if non_rl_attempts >= 3:
                        return pd.DataFrame()
                    time.sleep(5)

    @property
    def financials(self):
        return self._fetch("financials")

    @property
    def balance_sheet(self):
        return self._fetch("balance_sheet")


# ── Config ─────────────────────────────────────────────────────────────────────
JPX_URL = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_e.xls"
)

START_DATE = "2023-01-01"

# LIMIT: Set to a small number (e.g., 20) for testing; None for full market (~4400)
LIMIT = None

import json

# ── Cache Configuration ───────────────────────────────────────────────────────
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
PRICE_CACHE_FILE = os.path.join(CACHE_DIR, "start_prices.json")
ENRICH_CACHE_FILE = os.path.join(CACHE_DIR, "enrichment.json")
JPX_CACHE_FILE = os.path.join(CACHE_DIR, "jpx_master.csv")

def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Step 1: fetch the JPX master list ─────────────────────────────────────────
def get_jpx_tickers() -> list[tuple[str, str, str]]:
    """Return [(yf_ticker, name, sector), …] for all TSE-listed equities."""
    if os.path.exists(JPX_CACHE_FILE):
        mtime = os.path.getmtime(JPX_CACHE_FILE)
        if (time.time() - mtime) < 86400:
            log.info("Loading JPX stock list from cache...")
            df = pd.read_csv(JPX_CACHE_FILE)
            return list(zip(df['Ticker'], df['Name'], df['Sector']))

    log.info("Downloading JPX stock list from JPX website...")
    try:
        resp = SESSION.get(JPX_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))

        code_col   = next((c for c in df.columns if "Local Code"      in str(c)), "Code")
        name_col   = next((c for c in df.columns if "Name (English)"  in str(c)), "Name")
        sector_col = next((c for c in df.columns if "33 Sector(name)" in str(c)), "Sector")

        tickers = df[code_col].astype(str).str.strip()
        yf_tickers = [f"{t}.T" if len(t) >= 4 else t for t in tickers]

        result_df = pd.DataFrame({
            'Ticker': yf_tickers,
            'Name': df[name_col].tolist(),
            'Sector': df[sector_col].tolist()
        })
        result_df.to_csv(JPX_CACHE_FILE, index=False)

        return list(zip(result_df['Ticker'], result_df['Name'], result_df['Sector']))
    except Exception as exc:
        log.error("Failed to fetch JPX list: %s", exc)
        return []


# ── Step 2: Optimized Analysis ───────────────────────────────────────────────
def analyze_market(ticker_info: list[tuple[str, str, str]]):
    """
    Optimized analysis using caching and high-volume bulk downloads.
    yfinance calls use dynamic retry; no custom session is passed.
    """
    start_cache = load_cache(PRICE_CACHE_FILE)
    all_tickers = [t[0] for t in ticker_info]
    ticker_map = {t[0]: (t[1], t[2]) for t in ticker_info}

    # 1. Identify which tickers need historical start prices
    missing_start = [t for t in all_tickers if t not in start_cache]

    if missing_start:
        log.info("Fetching start prices for %d new stocks...", len(missing_start))
        chunk_size = 250
        total_chunks = (len(missing_start) + chunk_size - 1) // chunk_size
        for i in range(0, len(missing_start), chunk_size):
            chunk = missing_start[i : i + chunk_size]
            log.info(
                "  Processing Start Price chunk %d/%d...",
                (i // chunk_size) + 1, total_chunks,
            )
            # ── FIX: no session= passed; retry handled internally ──────────
            data = yf_download_with_retry(
                chunk,
                start=START_DATE,
                end="2023-01-15",
                group_by="ticker",
            )
            if data.empty:
                continue
            for ticker in chunk:
                try:
                    t_df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
                    hist = t_df.dropna(subset=["Close"])
                    if not hist.empty:
                        start_cache[ticker] = {
                            "date": hist.index[0].strftime('%Y-%m-%d'),
                            "price": float(hist["Close"].iloc[0])
                        }
                except Exception:
                    continue
        save_cache(PRICE_CACHE_FILE, start_cache)

    # 2. Bulk fetch latest prices for ALL stocks
    log.info("Fetching latest prices for %d stocks...", len(all_tickers))
    latest_prices = {}
    chunk_size = 250
    total_chunks = (len(all_tickers) + chunk_size - 1) // chunk_size
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i : i + chunk_size]
        log.info(
            "  Processing Latest Price chunk %d/%d...",
            (i // chunk_size) + 1, total_chunks,
        )
        # ── FIX: no session= passed; retry handled internally ─────────────
        data = yf_download_with_retry(chunk, period="1d", group_by="ticker")
        if data.empty:
            continue
        for ticker in chunk:
            try:
                t_df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
                hist = t_df.dropna(subset=["Close"])
                if not hist.empty:
                    latest_prices[ticker] = {
                        "date": hist.index[-1].strftime('%Y-%m-%d'),
                        "price": float(hist["Close"].iloc[-1])
                    }
            except Exception:
                continue

    # 3. Calculate returns
    hits = []
    for ticker in all_tickers:
        if ticker in start_cache and ticker in latest_prices:
            s_data = start_cache[ticker]
            l_data = latest_prices[ticker]

            p_start = s_data["price"]
            p_end = l_data["price"]

            if p_start > 0:
                return_pct = ((p_end - p_start) / p_start) * 100

                d_start = datetime.strptime(s_data["date"], '%Y-%m-%d')
                d_end = datetime.strptime(l_data["date"], '%Y-%m-%d')
                days = (d_end - d_start).days
                years = days / 365.25
                cagr = (pow(p_end / p_start, 1 / years) - 1) * 100 if years > 0 else return_pct

                name, sector = ticker_map[ticker]
                hits.append({
                    "Ticker": ticker,
                    "Name": name,
                    "Sector": sector,
                    "First Trading Date": s_data["date"],
                    "Start Price (2023)": round(p_start, 2),
                    "Latest Price": round(p_end, 2),
                    "Return %": round(return_pct, 2),
                    "CAGR %": round(cagr, 2)
                })

    return hits


def enrich_data(hits):
    """
    Enriches with 2022 financials, utilizing cache for static data.
    yfinance Ticker calls use dynamic retry; no custom session is passed.
    """
    enrich_cache = load_cache(ENRICH_CACHE_FILE)
    enriched = []
    total = len(hits)

    log.info("Enriching %d top performers with 2022 data...", total)
    for i, hit in enumerate(hits, 1):
        ticker = hit["Ticker"]

        if ticker in enrich_cache:
            hit.update(enrich_cache[ticker])
            enriched.append(hit)
            continue

        log.info("[%d/%d] Fetching NEW 2022 details for %s", i, total, ticker)
        try:
            # ── FIX: use _RetryTicker wrapper instead of yf.Ticker(session=…) ──
            t_obj = yf_ticker_with_retry(ticker)

            data_to_cache = {}

            fin = t_obj.financials
            bs = t_obj.balance_sheet

            def get_year_col(df, year):
                if df is None or df.empty:
                    return None
                for col in df.columns:
                    if str(year) in str(col):
                        return col
                return None

            c_22_f = get_year_col(fin, 2022)
            c_22_b = get_year_col(bs, 2022)

            # 1. ROE (2022)
            data_to_cache["ROE (2022) %"] = "N/A"
            if c_22_f is not None and c_22_b is not None:
                net = fin.loc["Net Income", c_22_f] if "Net Income" in fin.index else None
                eq = None
                if "Stockholders Equity" in bs.index:
                    eq = bs.loc["Stockholders Equity", c_22_b]
                elif "Common Stock Equity" in bs.index:
                    eq = bs.loc["Common Stock Equity", c_22_b]
                if net and eq:
                    data_to_cache["ROE (2022) %"] = round((net / eq) * 100, 2)

            # 2. P/E and Equity Ratio (2022)
            data_to_cache["P/E (2022 Earnings/2023 Price)"] = "N/A"
            data_to_cache["Equity Ratio (2022) %"] = "N/A"

            if c_22_f is not None:
                eps = (
                    fin.loc["Diluted EPS", c_22_f]
                    if "Diluted EPS" in fin.index
                    else fin.get("Basic EPS", {}).get(c_22_f)
                )
                if eps and eps > 0:
                    data_to_cache["P/E (2022 Earnings/2023 Price)"] = round(
                        hit["Start Price (2023)"] / eps, 2
                    )

                if c_22_b is not None:
                    ast = bs.loc["Total Assets", c_22_b] if "Total Assets" in bs.index else None
                    eq = None
                    if "Stockholders Equity" in bs.index:
                        eq = bs.loc["Stockholders Equity", c_22_b]
                    elif "Common Stock Equity" in bs.index:
                        eq = bs.loc["Common Stock Equity", c_22_b]
                    if ast and eq:
                        data_to_cache["Equity Ratio (2022) %"] = round((eq / ast) * 100, 2)

            enrich_cache[ticker] = data_to_cache
            hit.update(data_to_cache)

        except Exception as e:
            log.warning("Could not enrich %s: %s", ticker, e)
            hit.update({
                "ROE (2022) %": "N/A",
                "P/E (2022 Earnings/2023 Price)": "N/A",
                "Equity Ratio (2022) %": "N/A"
            })

        enriched.append(hit)
        time.sleep(0.3)   # light courtesy delay; real back-off happens on 429s

    save_cache(ENRICH_CACHE_FILE, enrich_cache)
    return enriched


def send_email(file_path, total_hits):
    """Sends the result CSV via email."""
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        log.warning("Email credentials missing. Skipping email.")
        return

    msg = EmailMessage()
    msg['Subject'] = "Daily Japan Stock Report: Top 10 Returns per Sector"
    msg['From'] = sender
    msg['To'] = receiver
    msg.set_content(
        f"Found {total_hits} stocks in total for the top 10 performers in each sector "
        f"since {START_DATE}.\n\nPlease find the attached CSV for details."
    )

    with open(file_path, 'rb') as f:
        msg.add_attachment(
            f.read(), maintype='application', subtype='csv', filename=file_path
        )

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    log.info("Email sent successfully to %s", receiver)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    ticker_info = get_jpx_tickers()
    if not ticker_info:
        log.error("No tickers retrieved – exiting.")
    else:
        if LIMIT:
            log.info("LIMIT=%d – testing first %d stocks only.", LIMIT, LIMIT)
            ticker_info = ticker_info[:LIMIT]

        results = analyze_market(ticker_info)

        if results:
            df_initial = pd.DataFrame(results)

            df_top10_initial = (
                df_initial.sort_values(["Sector", "Return %"], ascending=[True, False])
                .groupby("Sector")
                .head(10)
            )

            top_hits = df_top10_initial.to_dict('records')
            enriched_results = enrich_data(top_hits)

            df_out = pd.DataFrame(enriched_results)

            fname = f"highest_returns_since_2023_{datetime.now().strftime('%Y%m%d')}.csv"
            df_out.to_csv(fname, index=False, encoding="utf-8-sig")

            send_email(fname, len(df_out))
            log.info("Results saved and emailed. Total matches: %d", len(df_out))
        else:
            log.info("No data collected.")

    log.info("Total execution time: %.1f seconds", time.time() - t0)