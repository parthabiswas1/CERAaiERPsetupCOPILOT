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

# ceraai/agents.py

class MapperAgent:
    def map_to_fusion(self, inputs: Dict) -> Dict:
        legal_name = inputs.get("legal_name", "DemoCo")
        country = (inputs.get("country") or "US").upper()
        address = inputs.get("address") or {
            "line1": "123 Main St",
            "city": "San Jose",
            "state": "CA",
            "postalCode": "95110",
            "country": country,
        }

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
                "registrations": registrations,
            },
            "sequence": [
                {"endpoint": "/legalAddresses", "method": "POST"},
                {"endpoint": "/legalEntities", "method": "POST"},
                {"endpoint": "/legalEntityRegistrations", "method": "POST"},
            ],
            "schemaVersion": "fusion-2024.1",
        }

