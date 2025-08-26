from fastapi import FastAPI, Depends, HTTPException, Request, File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from io import BytesIO
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill
from datetime import datetime
from typing import Any
import os, sqlite3, json, pathlib

from ceraai.tools import RulesTool, ERPConnector, AuditTool
from ceraai.agents import InterviewAgent, ValidatorAgent, MapperAgent, ExecutorAgent, AuditorAgent
from ceraai.rag import RAGTool

# ---------- Config ----------
DB_PATH = os.getenv("DB_PATH", str(pathlib.Path("/tmp/ceraai.db")))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
UPLOAD_ROOT = pathlib.Path("/tmp/ceraai_uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CERAai ERP Setup Copilot - MVP")
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Agents/Tools ----------
rag = RAGTool()
rules_tool = RulesTool(DB_PATH)
interview_agent = InterviewAgent()
validator_agent = ValidatorAgent()
executor_agent  = ExecutorAgent()
erp_connector   = ERPConnector()
auditor_agent   = AuditorAgent()
audit_tool      = AuditTool()
mapper_agent    = MapperAgent()

# ---------- DB bootstrap (JSON state) ----------
def init_db():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    # Base table (keep name 'runs' but ensure a 'state' JSON column exists)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
      run_id TEXT PRIMARY KEY
    )
    """)

    # country_packs: stores normalized “pack” JSON per country
    cur.execute("""
    CREATE TABLE IF NOT EXISTS country_packs (
      country TEXT PRIMARY KEY,
      json TEXT NOT NULL,
      version TEXT DEFAULT 'demo',
      updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )""")
    
    # Add columns if missing
    def _add(col, ddl):
        try: cur.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError: pass

    _add("state", "TEXT")  # single source of truth
    _add("created_at", "TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))")
    _add("updated_at", "TEXT")
    con.commit(); con.close()

def default_state(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "inputs": {},
        "missing": [],
        "mapped": None,
        "result": None,
        "gating": {},
        "phase": "gating",
        "required_fields": None,
        "current_ask_for": None,
        "complete": False,
    }

def jdump(x): return json.dumps(x, ensure_ascii=False)
def jload(s, default):
    try:
        return json.loads(s) if s not in (None, "") else default
    except Exception:
        return default

def load_state(run_id: str) -> dict:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT state FROM runs WHERE run_id=?", (run_id,))
    row = cur.fetchone(); con.close()
    if not row:
        return default_state(run_id)
    st = jload(row[0], {}) if row[0] else {}
    # Ensure required keys exist
    base = default_state(run_id)
    base.update(st or {})
    return base

def save_state(run_id: str, st: dict):
    # Never pass **st; always pass dict
    sjson = jdump({k: v for k, v in st.items() if k != "run_id"})
    now = datetime.utcnow().isoformat() + "Z"
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO runs (run_id) VALUES (?)", (run_id,))
    cur.execute("UPDATE runs SET state=?, updated_at=? WHERE run_id=?", (sjson, now, run_id))
    con.commit(); con.close()

# ---------- Middleware ----------
@app.middleware("http")
async def ensure_run_id(request: Request, call_next):
    rid = request.headers.get("x-run-id") or str(uuid4())
    request.state.run_id = rid
    resp = await call_next(request)
    resp.headers["X-Run-ID"] = rid
    return resp

# ---------- Helpers ----------
def _run_dir(run_id: str) -> pathlib.Path:
    d = UPLOAD_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def basic_ok(creds: HTTPBasicCredentials):
    return (
        creds.username == os.getenv("DEMO_USER", "demo") and
        creds.password == os.getenv("DEMO_PASS", "demo")
    )

# Gating logic
GATING_ORDER = ["country","employ_in_country","sells_in_country","has_tax_id","regulatory_category"]

def get_country_pack(ctry: str | None):
    c = (ctry or "US").upper()
    # 1) try DB
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT json FROM country_packs WHERE country=?", (c,))
    row = cur.fetchone()
    con.close()
    if row:
        return json.loads(row[0])

    # 2) generate via RAG, persist, return
    pack = rag.generate_pack(c)
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""INSERT INTO country_packs(country,json,version,updated_at)
                   VALUES(?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                   ON CONFLICT(country) DO UPDATE SET json=excluded.json,
                       version=excluded.version, updated_at=excluded.updated_at""",
                (c, json.dumps(pack), "rag-derivative"))
    con.commit(); con.close()
    return pack


def normalize_country(text: str) -> str | None:
    t = text.strip().lower()
    if t in {"us","usa","united states","united states of america"}: return "US"
    return t.upper() if len(t) == 2 else None

def parse_bool(text: str) -> bool | None:
    t = text.strip().lower()
    if t in {"true","yes","y"}: return True
    if t in {"false","no","n"}: return False
    return None

def _reg_opts(pack: dict) -> set:
    for g in pack.get("gating", []):
        if g.get("key") == "regulatory_category":
            return set(g.get("options", []))
    return {"none","npo","gov","fsi"}

def validate_gating_answer(key: str, text: str, pack: dict):
    if key == "country":
        c = normalize_country(text)
        return (True, {"country": c}) if c else (False, "Reply with a valid 2-letter ISO code (e.g., US).")
    if key in {"employ_in_country","sells_in_country","has_tax_id"}:
        b = parse_bool(text)
        return (True, {key: b}) if b is not None else (False, "Reply exactly: true or false.")
    if key == "regulatory_category":
        opts = _reg_opts(pack)
        v = text.strip().lower()
        return (True, {key: v}) if v in opts else (False, f"Choose one of: {', '.join(sorted(opts))}.")
    return (False, "Unsupported gating key.")

def next_gating_key(st: dict, pack: dict) -> str | None:
    g = st.get("gating", {})
    for k in GATING_ORDER:
        if g.get(k) in (None, ""):
            return k
    return None

def gating_question(key: str, pack: dict) -> str:
    fallback = {
      "country": "Which country is this legal entity in? Reply with the 2-letter ISO code (e.g., US).",
      "employ_in_country": "Will this entity employ staff in this country at go-live? Reply true or false.",
      "sells_in_country": "Will this entity sell goods/services in this country? Reply true or false.",
      "has_tax_id": "Do you already have a national tax ID for this entity? Reply true or false.",
      "regulatory_category": "Regulatory category? Reply one of: none, npo, gov, fsi."
    }
    return fallback.get(key, f"Provide value for {key}.")

def derive_required_fields(pack: dict, gating: dict) -> list[str]:
    req = list(pack["base_fields"])
    for g in pack["gating"]:
        k = g["key"]; v = gating.get(k)
        if v is True and "on_true_add" in g:
            req += g["on_true_add"]
        if isinstance(v, str) and g.get("type") == "enum":
            req += (g.get("map") or {}).get(v, [])
    return list(dict.fromkeys(req))

# ---------- Basic routes ----------
@app.get("/")
def root(): return {"service": "CERAaiERPsetupCOPILOT", "status": "ok"}

@app.get("/health")
def health(): return {"ok": True}

@app.get("/secure-check")
def secure_check(credentials: HTTPBasicCredentials = Depends(security)):
    if basic_ok(credentials): return {"access": "granted"}
    raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- State ----------
@app.get("/state")
def get_state(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    st = load_state(request.state.run_id)
    return st

# ---------- Validator/Mapper/Execute (unchanged except save_state signature safe) ----------
@app.post("/validate")
def validate(payload: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    res = validator_agent.validate(payload, rules_tool)
    return {"run_id": request.state.run_id, **res}

@app.post("/map")
def map_to_fusion(payload: dict | None = None, request: Request = None,
                  credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    st = load_state(request.state.run_id)
    if payload:
        st["inputs"].update(payload)
    res = validator_agent.validate(st["inputs"], rules_tool)
    if res["status"] != "ok":
        st["missing"] = res["missing"]
        save_state(request.state.run_id, st)
        raise HTTPException(status_code=422, detail={"missing": res["missing"]})
    mapped = mapper_agent.map_to_fusion(st["inputs"])
    st["mapped"] = mapped
    save_state(request.state.run_id, st)
    return {"run_id": request.state.run_id, "status": "ok", "mapped": mapped}

def idem_get(scope, key): return None
def idem_put(scope, key, value): pass

@app.post("/execute")
def execute(payload: dict | None = None, request: Request = None,
           credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    st = load_state(request.state.run_id)
    work = payload or st.get("mapped")
    if not work:
        raise HTTPException(status_code=400, detail="No mapped payload. Call /map first or pass payload.")
    idem_key = request.headers.get("idempotency-key")
    if idem_key:
        cached = idem_get("execute", idem_key)
        if cached: return {"run_id": request.state.run_id, **cached}
    res = executor_agent.execute(work, erp_connector)
    st["result"] = res
    save_state(request.state.run_id, st)
    if idem_key: idem_put("execute", idem_key, res)
    return {"run_id": request.state.run_id, **res}

# ---------- Audit ----------
@app.post("/audit")
def audit(event: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    event = {**event, "run_id": request.state.run_id}
    return auditor_agent.record(event, audit_tool)

@app.get("/audit/logs")
def audit_logs(run_id: str | None = None, limit: int = 200,
               credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    data = auditor_agent.fetch(audit_tool, limit)["logs"]
    return {"logs": [e for e in data if not run_id or e.get("run_id") == run_id]}

# ---------- Interview (gating Q&A) ----------
@app.get("/interview/next")
def interview_next(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    run_id = request.state.run_id
    st = load_state(run_id)
    st.setdefault("gating", {}); st.setdefault("inputs", {})
    ctry = st["gating"].get("country") or st["inputs"].get("country")
    pack = get_country_pack(ctry)
    ask_for = st.get("current_ask_for") or next_gating_key(st, pack)
    if not ask_for:
        st["complete"] = True
        save_state(run_id, st)
        return {"run_id": run_id, "phase": "gating", "complete": True,
                "message": "All gating questions answered."}
    st["current_ask_for"] = ask_for
    save_state(run_id, st)
    return {"run_id": run_id, "phase": "gating", "ask_for": ask_for,
            "complete": False, "question": gating_question(ask_for, pack)}

@app.post("/interview/answer")
def interview_answer(payload: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    run_id = request.state.run_id
    text = str(payload.get("text", "")).strip()
    st = load_state(run_id)
    st.setdefault("gating", {}); st.setdefault("inputs", {})
    ctry = st["gating"].get("country") or st["inputs"].get("country")
    pack = get_country_pack(ctry)

    ask_for = st.get("current_ask_for")
    if not ask_for:
        ask_for = next_gating_key(st, pack)
        if not ask_for:
            st["complete"] = True
            save_state(run_id, st)
            return {"run_id": run_id, "phase": "gating", "accepted": True,
                    "complete": True, "message": "Gating already complete."}
        st["current_ask_for"] = ask_for
        save_state(run_id, st)

    ok, res = validate_gating_answer(ask_for, text, pack)
    if not ok:
        return {"run_id": run_id, "phase": "gating", "accepted": False,
                "ask_for": ask_for, "message": res}

    # persist answer
    st["gating"].update(res)
    st["inputs"].update(res)

    # compute next
    next_key = next_gating_key(st, pack)
    if next_key:
        st["current_ask_for"] = next_key
        save_state(run_id, st)
        return {"run_id": run_id, "phase": "gating", "accepted": True,
                "next": {"run_id": run_id, "phase": "gating",
                         "ask_for": next_key, "complete": False,
                         "question": gating_question(next_key, pack)}}

    # done
    st["current_ask_for"] = None
    st["complete"] = True
    st["required_fields"] = derive_required_fields(pack, st["gating"])
    save_state(run_id, st)
    return {"run_id": run_id, "phase": "gating", "accepted": True,
            "complete": True, "message": "Gating questions complete. You may now download the template."}

# ---------- Template (XLSX) ----------
def build_template_xlsx(country: str = "US", required_fields: list[str] | None = None) -> BytesIO:
    wb = Workbook(); ws = wb.active; ws.title = "LegalEntity"
    ctry = (country or "US").upper()
    if required_fields and isinstance(required_fields, list) and required_fields:
        cols = required_fields
    else:
        base_cols = ["legal_name","country","address_line1","city","state_region","postal_code"]
        cols = base_cols + (["ein"] if ctry == "US" else ["tax_id"])
    ws.append(cols)
    demo_map = {
        "legal_name": "ABC, Inc.",
        "country": ctry,
        "address_line1": "123 Main St",
        "city": "San Jose",
        "state_region": "CA",
        "postal_code": "95110",
        "ein": "12-3456789",
        "tax_id": "TAX-XXX",
        "employer_registration_id": "",
        "indirect_tax_id": "",
        "regulator_code": "",
    }
    ws.append([demo_map.get(c, "") for c in cols])
    buf = BytesIO(); wb.save(buf); buf.seek(0); return buf

@app.get("/template/draft")
def template_draft(request: Request, country: str | None = None,
                   credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    st = load_state(request.state.run_id)
    ctry = (country or st.get("gating", {}).get("country") or st.get("inputs", {}).get("country") or "US").upper()
    fields = st.get("required_fields")
    xlsx = build_template_xlsx(ctry, fields)
    return StreamingResponse(
        xlsx,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="legal_entity_template_{ctry}.xlsx"'}
    )

# ---------- File upload/validate ----------
@app.post("/files/upload")
async def files_upload(request: Request, file: UploadFile = File(...),
                       credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx accepted")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    p = _run_dir(request.state.run_id) / "uploaded.xlsx"
    p.write_bytes(contents)
    return {"run_id": request.state.run_id, "stored": True, "path": str(p)}

@app.post("/files/validate")
def files_validate(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    p = _run_dir(request.state.run_id) / "uploaded.xlsx"
    if not p.exists():
        raise HTTPException(status_code=400, detail="No uploaded file for this run. POST /files/upload first.")
    wb = load_workbook(p); ws = wb.active
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    issues = []

    st = load_state(request.state.run_id)
    ctry = (st["inputs"].get("country") or "US").upper()
    base_req = ["legal_name","country","address_line1","city","state_region","postal_code"]
    req = base_req + (["ein"] if ctry=="US" else ["tax_id"])

    for col in req:
        if col not in headers:
            issues.append({"row": 1, "field": col, "error": "Missing required column"})

    hdr_idx = {h:i for i,h in enumerate(headers)}
    if not issues and ws.max_row >= 2:
        r = 2
        for col in req:
            cell = ws.cell(row=r, column=hdr_idx[col]+1)
            if cell.value in (None, ""):
                issues.append({"row": r, "field": col, "error": "Value required"})

    if "Errors" not in headers:
        ws.cell(row=1, column=len(headers)+1, value="Errors")
        headers.append("Errors")
    red = PatternFill(start_color="FFFECACA", end_color="FFFECACA", fill_type="solid")
    for item in issues:
        r = item["row"]; err_col = len(headers)
        curr = ws.cell(row=r, column=err_col).value or ""
        msg = f"{item['field']}: {item['error']}"
        ws.cell(row=r, column=err_col, value=(curr + ("; " if curr else "") + msg))
        if item["field"] in hdr_idx:
            ws.cell(row=r, column=hdr_idx[item["field"]]+1).fill = red

    review_path = _run_dir(request.state.run_id) / "review.xlsx"
    wb.save(review_path)
    return {"run_id": request.state.run_id, "issues": issues, "review_download": "/files/review"}

@app.get("/files/review")
def files_review(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    review_path = _run_dir(request.state.run_id) / "review.xlsx"
    if not review_path.exists():
        raise HTTPException(status_code=404, detail="No review file for this run.")
    return FileResponse(review_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename="review.xlsx")


# --- Packs CRUD via DB + RAG ---------------------------------

@app.post("/rag/upsert")
def rag_upsert(payload: Dict[str, Any], credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    # payload: {docs:[{title, text, country?, tags?}]}
    docs = payload.get("docs") or []
    n = rag.upsert_docs(docs)
    return {"upserted": n}

@app.post("/rag/search")
def rag_search(payload: Dict[str, Any], credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    q = payload.get("query","")
    k = int(payload.get("top_k",5))
    flt = payload.get("filter")
    res = rag.search(q, top_k=k, filter_=flt)
    return {"results": res}

@app.post("/packs/upsert")
def packs_upsert(payload: Dict[str, Any], credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    c = (payload.get("country") or "").upper()
    if not c or not payload.get("json"):
        raise HTTPException(status_code=400, detail="country and json required")
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""INSERT INTO country_packs(country,json,version,updated_at)
                   VALUES(?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                   ON CONFLICT(country) DO UPDATE SET json=excluded.json,
                       version=excluded.version, updated_at=excluded.updated_at""",
                (c, json.dumps(payload["json"]), payload.get("version","demo")))
    con.commit(); con.close()
    return {"status":"ok","country":c}

@app.get("/packs/get/{country}")
def packs_get(country: str, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT json,version,updated_at FROM country_packs WHERE country=?", (country.upper(),))
    row = cur.fetchone(); con.close()
    if not row: raise HTTPException(status_code=404, detail="Not found")
    js, ver, ts = row
    return {"country": country.upper(), "json": json.loads(js), "version": ver, "updated_at": ts}


# ---------- Startup ----------
@app.on_event("startup")
def boot():
    init_db()
