# ceraai/tools.py
from typing import Dict, List
import json, os, re, time
import sqlite3


PACK_PATH = os.path.join("knowledge", "pack_us_tech.json")

class RAGTool:
    def __init__(self, pack_path: str = PACK_PATH):
        self.docs = []
        if os.path.exists(pack_path):
            with open(pack_path) as f:
                self.docs = json.load(f)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.docs: return []
        q = query.lower()
        def score(doc):
            text = (doc.get("text","") + " " + " ".join(doc.get("tags",[]))).lower()
            # very small keyword score
            hits = 0
            for kw in re.findall(r"\w+", q):
                if kw in text: hits += 1
            return hits
        ranked = sorted(self.docs, key=score, reverse=True)
        return [{"id": d["id"], "snippet": d["text"], "tags": d.get("tags", [])} for d in ranked[:top_k]]


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
