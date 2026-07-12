"""
GA3 Q3 — Fixed Schema Invoice Extraction API

POST /extract  {"invoice_text": "..."}  ->  6 fixed fields, null if not found.

Parsing strategy: deterministic regex/heuristics (no LLM call) so the endpoint
is fast, free, and has no external dependency/rate-limit/token-expiry risk
while grading. Invoice formats vary (different label words, date formats,
number formats, currency notations), so each field is extracted by trying a
list of patterns in priority order and taking the first match.

Run locally:
    python -m uvicorn main:app --host 0.0.0.0 --port 8003
On Render:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import re
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Invoice Extraction API")

# CORS: grader calls this from a Cloudflare Worker (a browser-like caller),
# so cross-origin must be allowed. No specific-origin requirement here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    invoice_text: str


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _first_match(patterns: list[str], text: str, flags=re.IGNORECASE) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return None


def extract_invoice_no(text: str) -> str | None:
    return _first_match(
        [
            r"(?:invoice\s*(?:no\.?|number|#)|ref(?:erence)?)\s*[:#]\s*([A-Za-z0-9/\-]+)",
        ],
        text,
    )


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _parse_date_fragment(frag: str) -> str | None:
    frag = frag.strip().rstrip(".,")

    # Already ISO: YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", frag)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # "15 March 2026" or "15 Mar 2026"
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", frag)
    if m:
        d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if mon in _MONTHS:
            return f"{y:04d}-{_MONTHS[mon]:02d}-{d:02d}"

    # "April 3, 2026" or "April 3 2026"
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", frag)
    if m:
        mon, d, y = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        if mon in _MONTHS:
            return f"{y:04d}-{_MONTHS[mon]:02d}-{d:02d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$", frag)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    return None


def extract_date(text: str) -> str | None:
    raw = _first_match(
        [
            r"(?:date|issued|invoice\s*date)\s*[:]\s*([A-Za-z0-9,\s/\-]+?)(?:\n|$)",
        ],
        text,
    )
    if not raw:
        return None
    return _parse_date_fragment(raw)


def extract_vendor(text: str) -> str | None:
    # Explicit labels first.
    labelled = _first_match(
        [
            r"(?:vendor|seller|from|billed\s*by)\s*[:]\s*([^\n]+)",
        ],
        text,
    )
    if labelled:
        return labelled.strip()

    # Fallback: header line pattern "Name — Tax Invoice" / "Name - Invoice"
    m = re.match(r"^\s*([A-Za-z0-9&.,'\s]+?)\s*[—\-]\s*(?:Tax\s+)?Invoice", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


_CUR_TOKEN = r"Rs\.?|INR|USD|EUR|GBP|\$|₹|£|€"


def _parse_amount(num_str: str) -> float | None:
    cleaned = num_str.strip()
    if "," in cleaned and "." in cleaned:
        # Both present: comma is a thousands separator (e.g. "1,40,000.00").
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # Only a comma: could be thousands ("1,600") or a decimal comma ("780,00").
        # A comma followed by exactly 2 digits at the end is treated as decimal.
        if re.search(r",\d{2}$", cleaned):
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_money(labels: list[str], text: str) -> tuple[float | None, str | None]:
    """Search for a labelled money line, return (amount, currency_hint).
    Allows extra words between the label and the colon (e.g. "Amount Due:"),
    and accepts the currency symbol/code either before or after the number
    (e.g. "Rs. 780.00" or "780.00 INR")."""
    for label in labels:
        # currency BEFORE the number: "Label ...: Rs. 780.00" / "Label: $780"
        m = re.search(
            rf"{label}[^:\n]*[:]\s*(?:({_CUR_TOKEN})\s*)?([\d,]+[.,]?\d*)"
            rf"(?:\s*({_CUR_TOKEN}))?",
            text,
            re.IGNORECASE,
        )
        if m:
            currency_hint = m.group(1) or m.group(3)
            amount = _parse_amount(m.group(2))
            if amount is not None:
                return amount, currency_hint
    return None, None


def extract_amount_tax(text: str) -> tuple[float | None, float | None, str | None]:
    amount, cur1 = _extract_money(
        [
            r"sub\s*-?\s*total",
            r"(?:taxable|net)\s*(?:value|amount)",
            r"(?:base|basic)\s*amount",
            r"amount\s*\(?before\s*tax\)?",
            r"\bamount\b(?!\s*due)",
            r"\bprice\b",
        ],
        text,
    )
    tax, cur2 = _extract_money(
        [
            r"(?:gst|igst|cgst|sgst|vat|tax|service\s*tax|sales\s*tax|duty)\s*\([^)]*\)",
            r"(?:gst|igst|cgst|sgst|vat|tax|service\s*tax|sales\s*tax|duty)",
        ],
        text,
    )
    total, cur3 = _extract_money(
        [r"grand\s*total", r"total\s*due", r"amount\s*due", r"\btotal\b"],
        text,
    )

    # Fallback: no explicit subtotal/amount label found, but we do have a total
    # and a tax figure -> derive amount = total - tax (subtotal before tax).
    if amount is None and total is not None and tax is not None:
        amount = round(total - tax, 2)

    currency_hint = cur1 or cur2 or cur3
    return amount, tax, currency_hint


_CURRENCY_MAP = {
    "rs": "INR", "rs.": "INR", "inr": "INR", "₹": "INR",
    "usd": "USD", "$": "USD",
    "eur": "EUR", "€": "EUR",
    "gbp": "GBP", "£": "GBP",
}


def extract_currency(text: str, hint: str | None) -> str | None:
    # 1. Explicit "Currency: XXX" line wins.
    explicit = _first_match([r"currency\s*[:]\s*([A-Za-z]{3})"], text)
    if explicit:
        return explicit.upper()

    # 2. Currency symbol/code seen next to the amount/tax figures.
    if hint:
        mapped = _CURRENCY_MAP.get(hint.strip().lower())
        if mapped:
            return mapped

    # 3. Any known currency token anywhere in the text.
    m = re.search(r"\b(INR|USD|EUR|GBP)\b|Rs\.?|₹|\$|£|€", text, re.IGNORECASE)
    if m:
        token = m.group(0).lower()
        return _CURRENCY_MAP.get(token, token.upper() if token.isalpha() else None)

    return None


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@app.post("/extract")
async def extract(req: ExtractRequest):
    text = req.invoice_text or ""

    invoice_no = extract_invoice_no(text)
    date = extract_date(text)
    vendor = extract_vendor(text)
    amount, tax, currency_hint = extract_amount_tax(text)
    currency = extract_currency(text, currency_hint)

    return {
        "invoice_no": invoice_no,
        "date": date,
        "vendor": vendor,
        "amount": amount,
        "tax": tax,
        "currency": currency,
    }


@app.get("/")
async def root():
    return {"status": "ok", "endpoint": "POST /extract {invoice_text}"}
