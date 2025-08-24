# CERAaiERPsetupCOPILOT

Minimal FastAPI service for ERP setup demo with agenic AI

## Endpoints
- `/health` → returns `{"ok": true}`
- `/secure-check` → requires Basic Auth (user: demo, pass: demo)

## Run locally
```bash
pip install -r requirements.txt
uvicorn api:app --reload
