import pandas as pd
import yfinance as yf
import requests
import io
import time
import logging
import os
import smtplib
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
# Using curl_cffi to mimic a real Chrome browser fingerprint.
SESSION = curl_requests.Session(impersonate="chrome")

# ── Config ─────────────────────────────────────────────────────────────────────
JPX_URL = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_e.xls"
)

START_DATE = "2023-01-01"

# LIMIT: Set to a small number (e.g., 20) for testing; None for full market (~4400)
LIMIT = None 

# ── Step 1: fetch the JPX master list ─────────────────────────────────────────
def get_jpx_tickers() -> list[tuple[str, str, str]]:
    """Return [(yf_ticker, name, sector), …] for all TSE-listed equities."""
    log.info("Downloading JPX stock list …")
    try:
        resp = requests.get(JPX_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))

        code_col   = next((c for c in df.columns if "Local Code"      in str(c)), "Code")
        name_col   = next((c for c in df.columns if "Name (English)"  in str(c)), "Name")
        sector_col = next((c for c in df.columns if "33 Sector(name)" in str(c)), "Sector")

        tickers = df[code_col].astype(str).str.strip()
        names   = df[name_col].tolist()
        sectors = df[sector_col].tolist()

        yf_tickers = [f"{t}.T" if len(t) >= 4 else t for t in tickers]
        return list(zip(yf_tickers, names, sectors))
    except Exception as exc:
        log.error("Failed to fetch JPX list: %s", exc)
        return []

# ── Step 2: Sequential Analysis ───────────────────────────────────────────────
def analyze_market(ticker_info: list[tuple[str, str, str]]):
    """
    Processes each stock sequentially to calculate returns since 2023-01-01.
    """
    hits = []
    total = len(ticker_info)
    
    log.info("Starting sequential analysis of %d stocks...", total)
    
    for i, (ticker, name, sector) in enumerate(ticker_info, 1):
        log.info("Checking %d/%d: %s (%s)", i, total, ticker, name)
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            try:
                t_obj = yf.Ticker(ticker, session=SESSION)
                hist = t_obj.history(start=START_DATE, interval="1d")
                
                if hist.empty:
                    attempts += 1
                    if attempts < max_attempts:
                        wait = attempts * 60
                        log.warning("Empty data/Limit for %s. Waiting %ds (Attempt %d/%d)...", ticker, wait, attempts, max_attempts)
                        time.sleep(wait)
                        continue
                    else:
                        break

                if len(hist) < 2:
                    log.debug("Insufficient data for %s", ticker)
                    break
                
                price_start = float(hist["Close"].iloc[0])
                price_end = float(hist["Close"].iloc[-1])
                
                if price_start > 0:
                    return_pct = ((price_end - price_start) / price_start) * 100
                    
                    # Calculate CAGR
                    # hist.index[0] is the date of the first price in 2023
                    # hist.index[-1] is the date of the latest price
                    days = (hist.index[-1] - hist.index[0]).days
                    years = days / 365.25
                    if years > 0:
                        cagr = (pow(price_end / price_start, 1 / years) - 1) * 100
                    else:
                        cagr = return_pct # Fallback if same day
                        
                    hits.append({
                        "Ticker": ticker, 
                        "Name": name, 
                        "Sector": sector,
                        "First Trading Date": hist.index[0].strftime('%Y-%m-%d'),
                        "Start Price (2023)": round(price_start, 2),
                        "Latest Price": round(price_end, 2),
                        "Return %": round(return_pct, 2),
                        "CAGR %": round(cagr, 2)
                    })
                
                break # Success!
                
            except Exception as e:
                if "Rate Limit" in str(e) or "429" in str(e):
                    attempts += 1
                    wait = attempts * 60
                    log.warning("Rate limit hit for %s. Waiting %ds (Attempt %d/%d)...", ticker, wait, attempts, max_attempts)
                    time.sleep(wait)
                else:
                    log.error("Error processing %s: %s", ticker, e)
                    break
    
    return hits

def enrich_data(hits):
    """
    Fetches financials and business summary for the provided hits.
    """
    enriched = []
    total = len(hits)
    log.info("Enriching %d top performers with financials and summaries...", total)
    
    for i, hit in enumerate(hits, 1):
        ticker = hit["Ticker"]
        log.info("[%d/%d] Fetching details for %s", i, total, ticker)
        
        try:
            t_obj = yf.Ticker(ticker, session=SESSION)
            
            # 1. Business Summary
            info = t_obj.info
            hit["Summary"] = info.get("longBusinessSummary", "N/A")
            
            # 2. Financials
            fin = t_obj.financials
            bs = t_obj.balance_sheet
            
            # Helper to find the column for a specific year
            def get_year_col(df, year):
                if df is None or df.empty: return None
                # Look for a column that has the year in its name
                for col in df.columns:
                    if str(year) in str(col):
                        return col
                return None

            # Calculate ROE for 2020, 2021, 2022
            for year in [2020, 2021, 2022]:
                hit[f"ROE ({year}) %"] = "N/A"
                col_fin = get_year_col(fin, year)
                col_bs = get_year_col(bs, year)
                
                if col_fin is not None and col_bs is not None:
                    net_income = fin.loc["Net Income", col_fin] if "Net Income" in fin.index else None
                    equity = None
                    if "Stockholders Equity" in bs.index:
                        equity = bs.loc["Stockholders Equity", col_bs]
                    elif "Common Stock Equity" in bs.index:
                        equity = bs.loc["Common Stock Equity", col_bs]
                    
                    if net_income and equity and equity != 0:
                        roe = (net_income / equity) * 100
                        hit[f"ROE ({year}) %"] = round(roe, 2)

            # Specific 2022-based metrics
            col_2022_fin = get_year_col(fin, 2022)
            col_2022_bs = get_year_col(bs, 2022)
            
            hit["P/E (2022 Earnings/2023 Price)"] = "N/A"
            hit["Equity Ratio (2022) %"] = "N/A"
            
            if col_2022_fin is not None:
                # EPS for P/E calculation
                eps = None
                if "Diluted EPS" in fin.index:
                    eps = fin.loc["Diluted EPS", col_2022_fin]
                elif "Basic EPS" in fin.index:
                    eps = fin.loc["Basic EPS", col_2022_fin]
                
                if eps and eps > 0:
                    pe = hit["Start Price (2023)"] / eps
                    hit["P/E (2022 Earnings/2023 Price)"] = round(pe, 2)
                
                # Equity Ratio needs Balance Sheet
                if col_2022_bs is not None:
                    total_assets = bs.loc["Total Assets", col_2022_bs] if "Total Assets" in bs.index else None
                    equity = None
                    if "Stockholders Equity" in bs.index:
                        equity = bs.loc["Stockholders Equity", col_2022_bs]
                    elif "Common Stock Equity" in bs.index:
                        equity = bs.loc["Common Stock Equity", col_2022_bs]
                    
                    if total_assets and equity:
                        equity_ratio = (equity / total_assets) * 100
                        hit["Equity Ratio (2022) %"] = round(equity_ratio, 2)

        except Exception as e:
            log.warning("Could not enrich %s: %s", ticker, e)
            # Ensure keys exist even on failure
            hit.setdefault("Summary", "N/A")
            for year in [2020, 2021, 2022]:
                hit.setdefault(f"ROE ({year}) %", "N/A")
            hit.setdefault("P/E (2022 Earnings/2023 Price)", "N/A")
            hit.setdefault("Equity Ratio (2022) %", "N/A")
            
        enriched.append(hit)
        time.sleep(1) # Small delay to be polite
        
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
