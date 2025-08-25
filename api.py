from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import sqlite3, os, hashlib, time
from typing import Dict
import json

app = FastAPI(title="CERAai ERP Setup Copilot - MVP")
security = HTTPBasic()

DB_PATH = "rules.sqlite"

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

    country = str(payload.get("country", "")).upper()
    flags = {k:str(v).lower() for k,v in payload.items()}

    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT country,condition_key,condition_value,field,mandatory,message FROM rules WHERE country=? OR country IS NULL", (country,))
    rows = cur.fetchall(); con.close()

    missing = []
    for (_country, ckey, cval, field, mandatory, msg) in rows:
        if ckey and str(flags.get(ckey, "false")) != str(cval):
            continue  # condition not met
        if mandatory and not payload.get(field):
            missing.append({"field": field, "message": msg})

    return {"status": "ok" if not missing else "incomplete", "missing": missing}


def build_fusion_payload(inputs: Dict) -> Dict:
    legal_name = inputs.get("legal_name", "DemoCo")
    country = (inputs.get("country") or "US").upper()
    address = inputs.get("address") or {"line1":"123 Main St","city":"San Jose","state":"CA","postalCode":"95110","country":country}

    registrations = []
    if inputs.get("ein"):
        registrations.append({"type": "EIN", "value": inputs["ein"]})
    if inputs.get("ca_edd_id"):
        registrations.append({"type": "CA_EDD", "value": inputs["ca_edd_id"]})

    return {
        "apiVersion": "v1",
        "endpoint": "/legalEntities",
        "body": {
            "legalName": legal_name,
            "legalAddress": address,
            "country": country,
            "registrations": registrations
        },
        "sequence": [
            {"endpoint": "/legalAddresses", "method": "POST"},
            {"endpoint": "/legalEntities", "method": "POST"},
            {"endpoint": "/legalEntityRegistrations", "method": "POST"}
        ],
        "schemaVersion": "fusion-2024.1"
    }

@app.post("/map")
def map_to_fusion(payload: dict, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")
    mapped = build_fusion_payload(payload)
    return {"status": "ok", "mapped": mapped}

import random

@app.post("/execute")
def execute(payload: dict, credentials: HTTPBasicCredentials = Depends(security)):
    if not basic_ok(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")

    endpoint = payload.get("endpoint", "/legalEntities")
    body = payload.get("body", {})

    # generate deterministic fake IDs
    le_id = f"LE-{random.randint(1000, 9999)}"
    reg_ids = []
    for i, reg in enumerate(body.get("registrations", []), start=1):
        reg_ids.append({
            "type": reg.get("type"),
            "id": f"REG-{i}-{random.randint(100,999)}"
        })

    result = {
        "executed_endpoint": endpoint,
        "legalEntityId": le_id,
        "registrations": reg_ids,
        "status": "success"
    }
    return result


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

