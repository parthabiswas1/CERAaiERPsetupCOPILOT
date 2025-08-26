# ceraai/tools.py
from typing import Dict, List
import json, os, re, time
import sqlite3


class RulesTool:
    def __init__(self, db_path: str = "rules.sqlite"):
        self.db_path = db_path

    def compute_missing(self, payload: Dict) -> List[Dict]:
        country = str(payload.get("country", "")).upper()
        flags = {k: str(v).lower() for k, v in payload.items()}
        con = sqlite3.connect(self.db_path); cur = con.cursor()
        cur.execute("""
            SELECT country, condition_key, condition_value, field, mandatory, message
            FROM rules
            WHERE country=? OR country IS NULL
        """, (country,))
        rows = cur.fetchall(); con.close()

        missing = []
        for (_country, ckey, cval, field, mandatory, msg) in rows:
            if ckey and str(flags.get(ckey, "false")) != str(cval):
                continue
            if mandatory and not payload.get(field):
                missing.append({"field": field, "message": msg})
        return missing


class ERPConnector:
    def create_legal_entity(self, payload: Dict) -> Dict:
        return {"id": "LE-1001"}


class AuditTool:
    def __init__(self, path: str = os.environ.get("AUDIT_FILE", "audit_log.jsonl")):
        self.path = path

    def log(self, entry: Dict) -> None:
        entry = dict(entry)
        entry["timestamp"] = int(time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read(self, limit: int = 200) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            lines = f.readlines()[-limit:]
        return [json.loads(x) for x in lines]
