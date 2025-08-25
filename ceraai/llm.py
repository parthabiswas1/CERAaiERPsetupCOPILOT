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
