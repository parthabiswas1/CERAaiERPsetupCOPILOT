from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from fastapi import Request
from uuid import uuid4
import sqlite3, os, hashlib, time, json
from typing import Dict
from ceraai.tools import RAGTool, RulesTool, ERPConnector, AuditTool
from ceraai.agents import InterviewAgent, ValidatorAgent, MapperAgent, ExecutorAgent, AuditorAgent
import random



app = FastAPI(title="CERAai ERP Setup Copilot - MVP")
security = HTTPBasic()
DB_PATH = os.getenv("DB_PATH", "rules.sqlite")  # default file name if DB_PATH is not set

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
    # compute current missing to steer the question
    res = validator_agent.validate(st["inputs"], rules_tool)
    st["missing"] = res["missing"]; save_state(request.state.run_id, **st)

    # pick first missing field name (if any) to focus the LLM
    want = st["missing"][0]["field"] if st["missing"] else None
    hits = rag.retrieve("US legal entity setup")
    try:
        from ceraai import llm
        prompt = f"Ask one concise question to collect field: {want}" if want else "Ask one final confirmation question before mapping/executing."
        q = llm.next_best_question([h["snippet"] for h in hits], {"needed": want} if want else {})
    except Exception:
        q = f"Please provide {want}." if want else "Ready to proceed to mapping and execution?"
    return {"run_id": request.state.run_id, "question": q, "context": hits[:2], "missing": st["missing"]}


@app.post("/interview/answer")
def interview_answer(payload: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401)
    st = load_state(request.state.run_id)

    # accept either {"answers": {...}} or {"field": "...", "value": "..."}
    answers = payload.get("answers") or {}
    if "field" in payload and "value" in payload:
        answers[payload["field"]] = payload["value"]

    st["inputs"].update(answers)

    # re-validate after update
    res = validator_agent.validate(st["inputs"], rules_tool)
    st["missing"] = res["missing"]
    save_state(request.state.run_id, **st)

    return {"run_id": request.state.run_id, "status": res["status"], "missing": res["missing"], "inputs": st["inputs"]}


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
