from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import sqlite3, os, hashlib, time
from typing import Dict
import json
from ceraai.tools import RAGTool, RulesTool, ERPConnector
from ceraai.agents import InterviewAgent, ValidatorAgent, MapperAgent, ExecutorAgent
import random


app = FastAPI(title="CERAai ERP Setup Copilot - MVP")
security = HTTPBasic()

DB_PATH = "rules.sqlite"

rag = RAGTool()
rules_tool = RulesTool()
interview_agent = InterviewAgent()
validator_agent = ValidatorAgent()
executor_agent  = ExecutorAgent()
erp_connector   = ERPConnector()

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
def validate(payload: dict, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return validator_agent.validate(payload, rules_tool)




@app.post("/execute")
def execute(payload: dict, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return executor_agent.execute(payload, erp_connector)


LOG_FILE = "audit_log.jsonl"

def write_audit(entry: dict):
    entry["timestamp"] = int(time.time())
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

@app.post("/audit")
def audit(event: dict, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")
    write_audit(event)
    return {"logged": True}

@app.get("/audit/logs")
def get_audit(credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                logs.append(json.loads(line))
    return {"logs": logs}


rag = RAGTool()
interview_agent = InterviewAgent()

# replace (or add) the interview endpoint
@app.get("/interview/next")
def interview_next(credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")
    out = interview_agent.next_question(state={}, rag=rag)
    return out



mapper_agent = MapperAgent()

@app.post("/map")
def map_to_fusion(payload: dict, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")
    mapped = mapper_agent.map_to_fusion(payload)
    return {"status": "ok", "mapped": mapped}

