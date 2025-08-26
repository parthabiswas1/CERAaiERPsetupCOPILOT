from fastapi import FastAPI, Depends, HTTPException, Request, File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from uuid import uuid4
import sqlite3, os, hashlib, time, json
from typing import Dict
from ceraai.tools import RAGTool, RulesTool, ERPConnector, AuditTool
from ceraai.agents import InterviewAgent, ValidatorAgent, MapperAgent, ExecutorAgent, AuditorAgent
import random
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill
import pathlib, json


DB_PATH = os.getenv("DB_PATH", "rules.sqlite")  # default file name if DB_PATH is not set
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # set to your UI origin later

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


@app.on_event("startup")
def boot():
    init_db()  # make sure rules/runs/idem tables exist at process start


@app.middleware("http")
async def ensure_run_id(request: Request, call_next):
    rid = request.headers.get("x-run-id") or str(uuid4())
    request.state.run_id = rid
    resp = await call_next(request)
    resp.headers["X-Run-ID"] = rid
    return resp


rag = RAGTool()
rules_tool = RulesTool(DB_PATH)
interview_agent = InterviewAgent()
validator_agent = ValidatorAgent()
executor_agent  = ExecutorAgent()
erp_connector   = ERPConnector()
auditor_agent   = AuditorAgent()
audit_tool      = AuditTool()
mapper_agent    = MapperAgent()


def _run_dir(run_id: str) -> pathlib.Path:
    d = UPLOAD_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # pragmas for stability on lightweight hosts
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA foreign_keys=ON;")

    # --- rules (validator) ---
    cur.execute("""
      CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY,
        country TEXT,
        condition_key TEXT,
        condition_value TEXT,
        field TEXT,
        mandatory INTEGER,
        message TEXT
      )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rules_country ON rules(country)")
    cur.execute("SELECT COUNT(*) FROM rules")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO rules(country,condition_key,condition_value,field,mandatory,message) VALUES(?,?,?,?,?,?)",
            [
              ("US", None, None, "ein", 1, "EIN is required for US entities."),
              ("US", "employees_in_CA", "true", "ca_edd_id", 1, "CA EDD ID required if employees in CA.")
            ]
        )

    # --- runs (shared state per run_id) ---
    cur.execute("""
      CREATE TABLE IF NOT EXISTS runs (
        run_id    TEXT PRIMARY KEY,
        inputs    TEXT,   -- JSON
        missing   TEXT,   -- JSON
        mapped    TEXT,   -- JSON
        result    TEXT,   -- JSON
        updated_at INTEGER
      )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at)")

    # --- idem (idempotency cache) ---
    cur.execute("""
      CREATE TABLE IF NOT EXISTS idem (
        action TEXT,
        key    TEXT,
        resp   TEXT,  -- JSON
        PRIMARY KEY(action, key)
      )
    """)

    con.commit()
    con.close()

def _conn(): return sqlite3.connect(DB_PATH)

def load_state(run_id: str) -> dict:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    try:
        cur.execute("SELECT inputs,missing,mapped,result FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
    except sqlite3.OperationalError as e:
        # table missing -> initialize once and return empty state
        if "no such table" in str(e):
            con.close()
            init_db()
            return {"inputs": {}, "missing": [], "mapped": None, "result": None}
        con.close()
        raise
    con.close()
    if not row:
        return {"inputs": {}, "missing": [], "mapped": None, "result": None}
    inputs, missing, mapped, result = row
    return {
        "inputs": json.loads(inputs or "{}"),
        "missing": json.loads(missing or "[]"),
        "mapped": json.loads(mapped or "null"),
        "result": json.loads(result or "null"),
    }


def save_state(run_id: str, **patch):
    st = load_state(run_id)
    st.update(patch)
    con=_conn(); cur=con.cursor()
    cur.execute("""INSERT INTO runs(run_id,inputs,missing,mapped,result,updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     inputs=excluded.inputs, missing=excluded.missing,
                     mapped=excluded.mapped, result=excluded.result,
                     updated_at=excluded.updated_at""",
        (run_id,
         json.dumps(st["inputs"]), json.dumps(st["missing"]),
         json.dumps(st["mapped"]), json.dumps(st["result"]), int(time.time())))
    con.commit(); con.close()

def idem_get(action: str, key: str):
    con=_conn(); cur=con.cursor()
    cur.execute("SELECT resp FROM idem WHERE action=? AND key=?", (action,key))
    row=cur.fetchone(); con.close()
    return json.loads(row[0]) if row else None

def idem_put(action: str, key: str, resp: dict):
    con=_conn(); cur=con.cursor()
    cur.execute("INSERT OR REPLACE INTO idem(action,key,resp) VALUES(?,?,?)",
                (action,key,json.dumps(resp)))
    con.commit(); con.close()



def basic_ok(creds: HTTPBasicCredentials):
    return creds.username == os.getenv("DEMO_USER", "demo") and creds.password == os.getenv("DEMO_PASS", "demo")

def satisfied(inputs: dict) -> bool:
    ctry = (inputs.get("country") or "").upper()
    if ctry == "US":
        base = all(inputs.get(k) for k in
                   ["legal_name","country","address_line1","city","state_region","postal_code","ein"])
        if not base: return False
        if str(inputs.get("employees_in_CA")).lower() == "true":
            return bool(inputs.get("ca_edd_id"))
        return True
    # Non-US: minimal example
    return all(inputs.get(k) for k in
               ["legal_name","country","address_line1","city","state_region","postal_code","tax_id"])




@app.get("/")
def root():
    return {"service": "CERAaiERPsetupCOPILOT", "status": "ok"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/secure-check")
def secure_check(credentials: HTTPBasicCredentials = Depends(security)):
    if basic_ok(credentials): return {"access": "granted"}
    raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/validate")
def validate(payload: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    res = validator_agent.validate(payload, rules_tool)
    return {"run_id": request.state.run_id, **res}

@app.get("/state")
def get_state(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    return {"run_id": request.state.run_id, **load_state(request.state.run_id)}

@app.post("/execute")
def execute(payload: dict | None = None, request: Request = None,
           credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    st = load_state(request.state.run_id)

    # prefer mapped from state if payload not supplied
    work = payload or st.get("mapped")
    if not work:
        raise HTTPException(status_code=400, detail="No mapped payload. Call /map first or pass payload.")

    # idempotency
    idem_key = request.headers.get("idempotency-key")
    if idem_key:
        cached = idem_get("execute", idem_key)
        if cached: return {"run_id": request.state.run_id, **cached}

    res = executor_agent.execute(work, erp_connector)
    st["result"] = res; save_state(request.state.run_id, **st)

    if idem_key: idem_put("execute", idem_key, res)
    return {"run_id": request.state.run_id, **res}

LOG_FILE = "audit_log.jsonl"

#def write_audit(entry: dict):
#    entry["timestamp"] = int(time.time())
#    with open(LOG_FILE, "a") as f:
#        f.write(json.dumps(entry) + "\n")

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


@app.get("/interview/next")
def interview_next(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    st = load_state(request.state.run_id)
    res = validator_agent.validate(st["inputs"], rules_tool)
    st["missing"] = res["missing"]; save_state(request.state.run_id, **st)

    if satisfied(st["inputs"]):
        return {"run_id": request.state.run_id, "complete": True,
                "question": "I have enough to generate your Legal Entity template. Download it when ready.",
                "context": rag.retrieve("Oracle Fusion Legal Entity sequence")[:2]}

    ask_for = st["missing"][0]["field"] if st["missing"] else "confirmation"
    hits = rag.retrieve(f"Legal Entity field {ask_for} US")
    try:
        from ceraai import llm
        q = llm.next_best_question([h["snippet"] for h in hits], {"needed": ask_for})
    except Exception:
        q = f"Please provide {ask_for}."
    return {"run_id": request.state.run_id, "complete": False, "ask_for": ask_for, "question": q, "context": hits[:2]}



@app.post("/interview/answer")
def interview_answer(payload: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    text = (payload or {}).get("text", "").strip()
    if not text: raise HTTPException(status_code=400, detail="Provide 'text' with your answer.")

    st = load_state(request.state.run_id)
    hits = rag.retrieve("Oracle Fusion Legal Entity requirements US")
    from ceraai import llm
    extracted = llm.extract_fields(text, st["inputs"].get("country"), [h["snippet"] for h in hits])

    # merge & validate
    st["inputs"].update(extracted or {})
    res = validator_agent.validate(st["inputs"], rules_tool)
    st["missing"] = res["missing"]
    is_ready = satisfied(st["inputs"])

    save_state(request.state.run_id, **st)
    return {
        "run_id": request.state.run_id,
        "status": "ready" if is_ready else ("ok" if res["status"]=="ok" else "incomplete"),
        "extracted": extracted,
        "missing": st["missing"],
        "next_hint": (st["missing"][0]["field"] if st["missing"] else None),
        "complete": is_ready
    }



@app.post("/map")
def map_to_fusion(payload: dict | None = None, request: Request = None,
                  credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    st = load_state(request.state.run_id)
    if payload:  # allow overriding/adding
        st["inputs"].update(payload)

    res = validator_agent.validate(st["inputs"], rules_tool)
    if res["status"] != "ok":
        save_state(request.state.run_id, inputs=st["inputs"], missing=res["missing"])
        raise HTTPException(status_code=422, detail={"missing": res["missing"]})

    mapped = mapper_agent.map_to_fusion(st["inputs"])
    st["mapped"] = mapped; save_state(request.state.run_id, **st)
    return {"run_id": request.state.run_id, "status":"ok", "mapped": mapped}

def build_template_xlsx(country: str = "US") -> BytesIO:
    wb = Workbook()
    ws = wb.active
    ws.title = "LegalEntity"

    base_cols = ["legal_name","country","address_line1","city","state_region","postal_code"]
    if (country or "US").upper() == "US":
        cols = base_cols + ["ein","employees_in_CA","ca_edd_id"]
    else:
        cols = base_cols + ["tax_id"]

    ws.append(cols)
    # example row for guidance
    demo = ["ABC, Inc.", country, "123 Main St", "San Jose", "CA", "95110"]
    if country.upper()=="US":
        demo += ["12-3456789", "false", ""]
    else:
        demo += ["TAX-XXX"]
    ws.append(demo)

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


@app.get("/template/draft")
def template_draft(request: Request, country: str | None = None,
                   credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    st = load_state(request.state.run_id)
    ctry = (country or st["inputs"].get("country") or "US").upper()
    xlsx = build_template_xlsx(ctry)
    return StreamingResponse(xlsx, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": f'attachment; filename="legal_entity_template_{ctry}.xlsx"'})


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

    wb = load_workbook(p)
    ws = wb.active
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    issues = []

    # required columns (US as default); country from state if present
    st = load_state(request.state.run_id)
    ctry = (st["inputs"].get("country") or "US").upper()
    base_req = ["legal_name","country","address_line1","city","state_region","postal_code"]
    req = base_req + (["ein"] if ctry=="US" else ["tax_id"])

    for col in req:
        if col not in headers:
            issues.append({"row": 1, "field": col, "error": "Missing required column"})

    # per-row empties for first data row (row 2)
    hdr_idx = {h:i for i,h in enumerate(headers)}
    if not issues and ws.max_row >= 2:
        r = 2
        for col in req:
            cell = ws.cell(row=r, column=hdr_idx[col]+1)
            if cell.value in (None, ""):
                issues.append({"row": r, "field": col, "error": "Value required"})

    # produce review.xlsx with Errors column + highlight
    if "Errors" not in headers:
        ws.cell(row=1, column=len(headers)+1, value="Errors")
        headers.append("Errors")
    red = PatternFill(start_color="FFFECACA", end_color="FFFECACA", fill_type="solid")
    for item in issues:
        r = item["row"]
        # append message into Errors column
        err_col = len(headers)
        curr = ws.cell(row=r, column=err_col).value or ""
        msg = f"{item['field']}: {item['error']}"
        ws.cell(row=r, column=err_col, value=(curr + ("; " if curr else "") + msg))
        # highlight offending cell if we can
        if item["field"] in hdr_idx:
            ws.cell(row=r, column=hdr_idx[item["field"]]+1).fill = red

    review_path = _run_dir(request.state.run_id) / "review.xlsx"
    wb.save(review_path)

    return {"run_id": request.state.run_id, "issues": issues,
            "review_download": "/files/review"}


@app.get("/files/review")
def files_review(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    review_path = _run_dir(request.state.run_id) / "review.xlsx"
    if not review_path.exists():
        raise HTTPException(status_code=404, detail="No review file for this run.")
    return FileResponse(review_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename="review.xlsx")



