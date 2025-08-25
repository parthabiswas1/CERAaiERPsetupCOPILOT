from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from fastapi import Request
from uuid import uuid4
import sqlite3, os, hashlib, time
from typing import Dict
import json
from ceraai.tools import RAGTool, RulesTool, ERPConnector, AuditTool
from ceraai.agents import InterviewAgent, ValidatorAgent, MapperAgent, ExecutorAgent, AuditorAgent
import random


app = FastAPI(title="CERAai ERP Setup Copilot - MVP")
security = HTTPBasic()
DB_PATH = "rules.sqlite"

@app.middleware("http")
async def ensure_run_id(request: Request, call_next):
    rid = request.headers.get("x-run-id") or str(uuid4())
    request.state.run_id = rid
    resp = await call_next(request)
    resp.headers["X-Run-ID"] = rid
    return resp


rag = RAGTool()
rules_tool = RulesTool()
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
    # seed minimal rules if empty
    cur.execute("SELECT COUNT(*) FROM rules")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO rules(country,condition_key,condition_value,field,mandatory,message) VALUES(?,?,?,?,?,?)",
            [
              ("US", None, None, "ein", 1, "EIN is required for US entities."),
              ("US", "employees_in_CA", "true", "ca_edd_id", 1, "CA EDD ID required if employees in CA.")
            ]
        )
    con.commit(); con.close()

init_db()

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

@app.post("/execute")
def execute(payload: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    res = executor_agent.execute(payload, erp_connector)
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
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    out = interview_agent.next_question(state={}, rag=rag)
    return {"run_id": request.state.run_id, **out}

@app.post("/map")
def map_to_fusion(payload: dict, request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials): raise HTTPException(status_code=401, detail="Unauthorized")
    mapped = mapper_agent.map_to_fusion(payload)
    return {"run_id": request.state.run_id, "status": "ok", "mapped": mapped}
