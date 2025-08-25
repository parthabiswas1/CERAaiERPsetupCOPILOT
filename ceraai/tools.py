# ceraai/tools.py
from typing import Dict, List

class RAGTool:
    def retrieve(self, query: str) -> List[Dict]:
        # placeholder: will use JSON pack in 3b
        return [{"source":"pack_us_tech","snippet":"EIN required for US entities."}]

class RulesTool:
    def __init__(self, db_path: str = "rules.sqlite"):
        self.db_path = db_path

class ERPConnector:
    def create_legal_entity(self, payload: Dict) -> Dict:
        # mock only; real connector later
        return {"id": "LE-1001"}

class AuditTool:
    def log(self, entry: Dict) -> None:
        # will wire to JSONL in 2d (already added)
        pass
