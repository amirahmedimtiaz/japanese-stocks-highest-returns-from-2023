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

# ── Rate Limiting ─────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self):
        # Limits: (count, seconds)
        self.get_limits = [(60, 60), (360, 3600), (8000, 86400)]
        self.put_limits = [(30, 60), (100, 3600), (2000, 86400)]
        self.post_limits = [(30, 60), (75, 3600), (1000, 86400)]
        
        self.history = {
            "GET": deque(),
            "PUT": deque(),
            "POST": deque()
        }

    def _wait_if_needed(self, method: str):
        method = method.upper()
        if method not in self.history:
            return

        limits = {
            "GET": self.get_limits,
            "PUT": self.put_limits,
            "POST": self.post_limits
        }.get(method, [])

        now = time.time()
        for limit_count, limit_seconds in limits:
            # Clean up history for this window
            while self.history[method] and self.history[method][0] < now - limit_seconds:
                self.history[method].popleft()
            
            if len(self.history[method]) >= limit_count:
                # Wait until the oldest request in this window expires
                sleep_time = self.history[method][0] + limit_seconds - now
                if sleep_time > 0:
                    log.info("Rate limit approaching for %s (%d requests in %ds). Sleeping %.2fs...", 
                             method, limit_count, limit_seconds, sleep_time)
                    time.sleep(sleep_time)
                    # Re-check all limits after sleeping
                    return self._wait_if_needed(method)

        self.history[method].append(time.time())

RATE_LIMITER = RateLimiter()
class RateLimitedSession(curl_requests.Session):
    def request(self, method, *args, **kwargs):
        RATE_LIMITER._wait_if_needed(method)
        return super().request(method, *args, **kwargs)

# ── Global Session for Browser Impersonation ───────────────────────────────────
# Using curl_cffi to mimic a real Chrome browser fingerprint.
SESSION = RateLimitedSession(impersonate="chrome")


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
    # Cache JPX list for 24 hours
    if os.path.exists(JPX_CACHE_FILE):
        mtime = os.path.getmtime(JPX_CACHE_FILE)
        if (time.time() - mtime) < 86400: # 24 hours
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
    """
    start_cache = load_cache(PRICE_CACHE_FILE)
    all_tickers = [t[0] for t in ticker_info]
    ticker_map = {t[0]: (t[1], t[2]) for t in ticker_info}
    
    # 1. Identify which tickers need historical start prices
    missing_start = [t for t in all_tickers if t not in start_cache]
    
    if missing_start:
        log.info("Fetching start prices for %d new stocks...", len(missing_start))
        # Larger chunks (250) to minimize GET requests
        chunk_size = 250
        for i in range(0, len(missing_start), chunk_size):
            chunk = missing_start[i : i + chunk_size]
            log.info("  Processing Start Price chunk %d/%d...", (i//chunk_size)+1, (len(missing_start)//chunk_size)+1)
            try:
                data = yf.download(chunk, start=START_DATE, end="2023-01-15", session=SESSION, group_by="ticker", progress=False)
                for ticker in chunk:
                    try:
                        # yfinance behaves differently for single vs multiple tickers
                        t_df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
                        hist = t_df.dropna(subset=["Close"])
                        if not hist.empty:
                            start_cache[ticker] = {
                                "date": hist.index[0].strftime('%Y-%m-%d'),
                                "price": float(hist["Close"].iloc[0])
                            }
                    except Exception: continue
            except Exception as e:
                log.error("Bulk start price error: %s", e)
        save_cache(PRICE_CACHE_FILE, start_cache)

    # 2. Bulk fetch latest prices for ALL stocks
    log.info("Fetching latest prices for %d stocks...", len(all_tickers))
    latest_prices = {}
    chunk_size = 250 # High density chunking
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i : i + chunk_size]
        log.info("  Processing Latest Price chunk %d/%d...", (i//chunk_size)+1, (len(all_tickers)//chunk_size)+1)
        try:
            data = yf.download(chunk, period="1d", session=SESSION, group_by="ticker", progress=False)
            for ticker in chunk:
                try:
                    # yfinance behaves differently for single vs multiple tickers
                    t_df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
                    hist = t_df.dropna(subset=["Close"])
                    if not hist.empty:
                        latest_prices[ticker] = {
                            "date": hist.index[-1].strftime('%Y-%m-%d'),
                            "price": float(hist["Close"].iloc[-1])
                        }
                except Exception: continue
        except Exception as e:
            log.error("Bulk latest price error: %s", e)

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
                
                # CAGR calculation
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
            t_obj = yf.Ticker(ticker, session=SESSION)
            
            data_to_cache = {}
            
            fin = t_obj.financials
            bs = t_obj.balance_sheet

            def get_year_col(df, year):
                if df is None or df.empty: return None
                for col in df.columns:
                    if str(year) in str(col): return col
                return None

            c_22_f = get_year_col(fin, 2022)
            c_22_b = get_year_col(bs, 2022)
            
            # 1. ROE (2022)
            data_to_cache["ROE (2022) %"] = "N/A"
            if c_22_f is not None and c_22_b is not None:
                net = fin.loc["Net Income", c_22_f] if "Net Income" in fin.index else None
                eq = None
                if "Stockholders Equity" in bs.index: eq = bs.loc["Stockholders Equity", c_22_b]
                elif "Common Stock Equity" in bs.index: eq = bs.loc["Common Stock Equity", c_22_b]
                if net and eq: data_to_cache["ROE (2022) %"] = round((net/eq)*100, 2)

            # 2. P/E and Equity Ratio (2022)
            data_to_cache["P/E (2022 Earnings/2023 Price)"] = "N/A"
            data_to_cache["Equity Ratio (2022) %"] = "N/A"
            
            if c_22_f is not None:
                eps = fin.loc["Diluted EPS", c_22_f] if "Diluted EPS" in fin.index else fin.get("Basic EPS", {}).get(c_22_f)
                if eps and eps > 0:
                    data_to_cache["P/E (2022 Earnings/2023 Price)"] = round(hit["Start Price (2023)"] / eps, 2)
                
                if c_22_b is not None:
                    ast = bs.loc["Total Assets", c_22_b] if "Total Assets" in bs.index else None
                    eq = None
                    if "Stockholders Equity" in bs.index: eq = bs.loc["Stockholders Equity", c_22_b]
                    elif "Common Stock Equity" in bs.index: eq = bs.loc["Common Stock Equity", c_22_b]
                    if ast and eq: data_to_cache["Equity Ratio (2022) %"] = round((eq/ast)*100, 2)

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
        time.sleep(0.5)
        
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
    msg['Subject'] = f"Daily Japan Stock Report: Top 10 Returns per Sector"
    msg['From'] = sender
    msg['To'] = receiver
    msg.set_content(f"Found {total_hits} stocks in total for the top 10 performers in each sector since {START_DATE}.\n\nPlease find the attached CSV for details.")

    with open(file_path, 'rb') as f:
        file_data = f.read()
        msg.add_attachment(file_data, maintype='application', subtype='csv', filename=file_path)

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
            
            # Filter top 10 in each sector first to minimize expensive enrichment calls
            df_top10_initial = (
                df_initial.sort_values(["Sector", "Return %"], ascending=[True, False])
                .groupby("Sector")
                .head(10)
            )
            
            # Convert to list of dicts for enrichment
            top_hits = df_top10_initial.to_dict('records')
            
            # Enrich with 2022 financials and summaries
            enriched_results = enrich_data(top_hits)
            
            df_out = pd.DataFrame(enriched_results)
            
            fname = f"highest_returns_since_2023_{datetime.now().strftime('%Y%m%d')}.csv"
            df_out.to_csv(fname, index=False, encoding="utf-8-sig")
            
            # Send Email
            send_email(fname, len(df_out))
            log.info("Results saved and emailed. Total matches: %d", len(df_out))
        else:
            log.info("No data collected.")

    log.info("Total execution time: %.1f seconds", time.time() - t0)
