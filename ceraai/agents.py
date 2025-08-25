# ceraai/agents.py
from typing import Dict, List
from .tools import RAGTool, RulesTool, ERPConnector, AuditTool

class InterviewAgent:
    def next_question(self, state: Dict, rag: RAGTool) -> Dict:
        hits = rag.retrieve("legal entity setup US")
        return {"question": "What is the company’s country of registration?", "context": hits[:1]}


class ValidatorAgent:
    def validate(self, payload: Dict, rules: RulesTool) -> Dict:
        missing = rules.compute_missing(payload)
        return {"status": "ok" if not missing else "incomplete", "missing": missing}


class MapperAgent:
    def map_to_fusion(self, inputs: Dict) -> Dict:
        # will delegate to build_fusion_payload in next step
        return {"status": "stub"}

class ExecutorAgent:
    def execute(self, mapped: Dict, erp: ERPConnector) -> Dict:
        # will delegate to /execute logic in next step
        return {"status": "stub"}

class AuditorAgent:
    def record(self, event: Dict, audit: AuditTool) -> None:
        audit.log(event)
