# ceraai/tools.py
from typing import Dict, List
import json, os, re

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

class ERPConnector:
    def create_legal_entity(self, payload: Dict) -> Dict:
        return {"id": "LE-1001"}

class AuditTool:
    def log(self, entry: Dict) -> None:
        pass
