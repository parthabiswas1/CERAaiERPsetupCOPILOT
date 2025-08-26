# ceraai/llm.py
import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM = (
    "You are CERAai Interview Agent. Goal: collect minimal fields to create a US Legal Entity. "
    "Ask ONE next-best question, concise, no chit-chat. If US and employees_in_CA=true, ensure CA EDD ID gets asked."
)

def next_best_question(context_snippets: list, answers: dict) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context_snippets}\nCurrent answers:\n{answers}\nReturn ONLY the next question."}
    ]
    r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2, max_tokens=64)
    return r.choices[0].message.content.strip()


def extract_fields(text: str, country: str | None, snippets: list[str]) -> dict:
    allowed = ["legal_name","country","address_line1","city","state_region","postal_code",
               "ein","employees_in_CA","ca_edd_id","tax_id"]
    prompt = (
        "Extract structured fields for Legal Entity creation. "
        "Return ONLY JSON with keys from this set: " + ", ".join(allowed) + ". "
        "Booleans as true/false. If unknown, omit the key.\n\n"
        f"Country hint: {country or ''}\n"
        f"Context: {snippets}\n"
        f"User text: {text}"
    )
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":"You extract precise JSON fields."},
                      {"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=200
        )
        import json as _json
        out = r.choices[0].message.content.strip()
        data = _json.loads(out) if out.startswith("{") else {}
    except Exception:
        data = {}

    # Fallback regex
    import re
    if "ein" not in data:
        m = re.search(r"\b(\d{2}-\d{7})\b", text)
        if m: data["ein"] = m.group(1)
    if "employees_in_CA" not in data:
        if re.search(r"\b(employees? in (CA|California))\b.*\b(yes|true)\b", text, re.I):
            data["employees_in_CA"] = True
    if "ca_edd_id" not in data and data.get("employees_in_CA") is True:
        m = re.search(r"\b(\d{3}-\d{4})\b", text)  # simple demo pattern
        if m: data["ca_edd_id"] = m.group(1)
    return data

