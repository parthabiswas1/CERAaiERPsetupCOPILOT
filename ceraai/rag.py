import os, json, time, hashlib
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pinecone import Pinecone

_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
_EMBED_DIM   = 1536
_PX_INDEX    = os.getenv("PINECONE_INDEX", "")

class RAGTool:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if not _PX_INDEX:
            raise RuntimeError("PINECONE_INDEX not set")
        self.index = self.pc.Index(_PX_INDEX)

    def _embed(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(model=_EMBED_MODEL, input=text)
        return emb.data[0].embedding

    def upsert_docs(self, docs: List[Dict[str, Any]], namespace: str = "oracle-legal") -> int:
        """
        docs: [{id?, title, text, country?, tags?}]
        """
        items = []
        for d in docs:
            _id = d.get("id") or hashlib.sha1((d.get("title","")+d.get("text","")).encode()).hexdigest()
            vec = self._embed(d["text"])
            meta = {k:v for k,v in d.items() if k not in ("id","text")}
            items.append({"id": _id, "values": vec, "metadata": meta})
        self.index.upsert(vectors=items, namespace=namespace)
        return len(items)

    def search(self, query: str, top_k: int = 5, namespace: str = "oracle-legal",
               filter_: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        qv = self._embed(query)
        res = self.index.query(vector=qv, top_k=top_k, include_metadata=True,
                               namespace=namespace, filter=filter_)
        out = []
        for m in res.matches:
            out.append({"id": m.id, "score": m.score, **(m.metadata or {})})
        return out

    def propose_pack_from_hits(self, country: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Very lean heuristic pack composer for demo.
        You can swap this later to an LLM prompt that synthesizes fields from hit snippets.
        """
        c = country.upper()
        # Minimal base; expand via keywords detected in hits
        base_fields = ["legal_name","country","address_line1","city","state_region","postal_code"]
        gating = [
            {"key":"employ_in_country","type":"bool"},
            {"key":"sells_in_country","type":"bool"},
            {"key":"has_tax_id","type":"bool"},
            {"key":"regulatory_category","type":"enum","options":["none","npo","gov","fsi"]}
        ]
        hints = {}
        # Simple signals from doc titles/tags
        blob = " ".join([(h.get("title","")+" ") + " ".join(h.get("tags",[]) ) for h in hits]).lower()

        if c == "US" or "ein" in blob:
            base_fields = base_fields + ["ein"]
            hints["ein"] = "Format NN-NNNNNNN."
            # common conditional US signals
            if "employment" in blob or "payroll" in blob or "edd" in blob:
                # add on_true fields for employing in country
                gating[0] = {"key":"employ_in_country","type":"bool","on_true_add":["employer_registration_id"]}
        elif c == "UK" or "hmrc" in blob or "paye" in blob or "companies house" in blob:
            base_fields = base_fields + ["crn"]
            hints["crn"] = "Companies House number (8 chars)."
            gating[0] = {"key":"employ_in_country","type":"bool","on_true_add":["paye_employer_ref","paye_accounts_office_ref"]}
            gating[1] = {"key":"sells_in_country","type":"bool","on_true_add":["vat_number"]}
        elif c == "IN" or "gstin" in blob or "cin" in blob or "pan" in blob:
            base_fields = base_fields + ["cin","pan"]
            hints["pan"] = "ABCDE1234F format."
            hints["cin"] = "21-character CIN."
            gating[0] = {"key":"employ_in_country","type":"bool","on_true_add":["epfo_reg_no","esic_no"]}
            gating[1] = {"key":"sells_in_country","type":"bool","on_true_add":["gstin"]}
            gating[2] = {"key":"has_tax_id","type":"bool","on_true_add":["tan"]}

        pack = {
            "country": c,
            "base_fields": list(dict.fromkeys(base_fields)),
            "gating": gating,
            "field_hints": hints
        }
        return pack

    def generate_pack(self, country: str) -> Dict[str, Any]:
        # retrieve top docs with country signal
        hits = self.search(query=f"legal entity setup requirements {country}", top_k=8,
                           filter_={"country": {"$in": [country.upper(), "GLOBAL"]}})
        return self.propose_pack_from_hits(country, hits or [])
