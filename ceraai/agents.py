# ceraai/agents.py
from typing import Dict, List
from .tools import RAGTool, RulesTool, ERPConnector, AuditTool
import hashlib, time

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
    def execute(self, payload: Dict, erp: ERPConnector) -> Dict:
        # expect { "endpoint": "/legalEntities", "body": {...} }
        endpoint = payload.get("endpoint", "/legalEntities")
        body = payload.get("body", {}) or {}

        # deterministic fake IDs based on legalName hash
        legal_name = (body.get("legalName") or "DemoCo").encode()
        seed = int(hashlib.sha256(legal_name).hexdigest()[:6], 16)
        le_id = f"LE-{seed % 9000 + 1000}"

        reg_ids = []
        for idx, reg in enumerate(body.get("registrations", []), start=1):
            rseed = int(hashlib.sha256(f"{legal_name}-{idx}".encode()).hexdigest()[:4], 16)
            reg_ids.append({"type": reg.get("type"), "id": f"REG-{idx}-{rseed % 900 + 100}"})

        return {
            "executed_endpoint": endpoint,
            "legalEntityId": le_id,
            "registrations": reg_ids,
            "status": "success",
            "ts": int(time.time()),
        }


class AuditorAgent:
    def record(self, event: Dict, audit: AuditTool) -> Dict:
        audit.log(event); return {"logged": True}

    def fetch(self, audit: AuditTool, limit: int = 200) -> Dict:
        return {"logs": audit.read(limit)}


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

