from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import sqlite3, os, hashlib, time

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
