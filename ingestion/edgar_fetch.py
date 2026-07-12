"""
Fetches recent 10-K/10-Q filing text for a given ticker from SEC EDGAR's
free full-text search and submissions APIs.

SEC requires a descriptive User-Agent on every request (see EDGAR_USER_AGENT
in .env) — requests without one get blocked.

Note: this module makes real network calls to sec.gov/data.sec.gov. If
you're running this inside a sandboxed environment with restricted network
egress, use ingestion/loaders.py with the sample filing in sample_data/
instead, or run this module from an unrestricted environment.
"""
from __future__ import annotations
import requests
from typing import List, Dict
from config import Config

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def _headers() -> dict:
    return {"User-Agent": Config.EDGAR_USER_AGENT}


def lookup_cik(ticker: str) -> str:
    """Maps a ticker (e.g. 'AAPL') to its zero-padded 10-digit CIK number."""
    resp = requests.get(EDGAR_TICKER_LOOKUP_URL, headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    ticker = ticker.upper()
    for entry in data.values():
        if entry.get("ticker") == ticker:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC company_tickers.json")


def list_recent_filings(ticker: str, forms: List[str] = None, limit: int = 5) -> List[Dict]:
    """
    Returns recent filings for a ticker, filtered to the given forms
    (defaults to 10-K and 10-Q).
    """
    forms = forms or ["10-K", "10-Q"]
    cik = lookup_cik(ticker)
    resp = requests.get(EDGAR_SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    results = []
    for i, form in enumerate(recent["form"]):
        if form in forms:
            results.append({
                "ticker": ticker.upper(),
                "cik": cik,
                "form": form,
                "filing_date": recent["filingDate"][i],
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
            })
        if len(results) >= limit:
            break
    return results


def fetch_filing_text(filing: Dict) -> str:
    """Downloads the primary document for a filing dict from list_recent_filings()."""
    accession_nodash = filing["accession_number"].replace("-", "")
    url = f"{EDGAR_ARCHIVES_BASE}/{int(filing['cik'])}/{accession_nodash}/{filing['primary_document']}"
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.text
