# Invoice Extraction API (GA3 Q3)

FastAPI service that parses raw invoice text into a fixed 6-field JSON schema.

- `POST /extract` with `{"invoice_text": "..."}`
- Returns `invoice_no, date, vendor, amount, tax, currency` (null if not found)
- CORS enabled for all origins (grader calls from a Cloudflare Worker)

## Deploy on Render
Auto-detects `render.yaml`, or manually:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
