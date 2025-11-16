# app/api/ask.py
from fastapi import APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
import os, requests, re
from typing import Optional

from app.connectors import (
    fetch_messages,
    build_index,
    search_similar,
    format_numbered_context,
)

router = APIRouter()


def configure_cors(app):
    """Optional: allow local/browser testing from any origin."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@router.on_event("startup")
async def startup_event():
    """
    Fetch messages and build the FAISS index at startup.
    """
    items = fetch_messages()
    if not items:
        # We still allow the app to start; /ask will reply "I don't know."
        print("No items fetched at startup.")
        return
    build_index(items)


def _extract_person_hint(question: str) -> Optional[str]:
    """
    Very light heuristic: a quoted name or a simple proper-noun span.
    """
    # "Name" inside quotes
    m = re.search(r'["“”](.+?)["“”]', question)
    if m and len(m.group(1).split()) <= 4:
        return m.group(1).strip()

    # Proper noun (First [Middle] Last)
    m2 = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", question)
    return m2.group(1).strip() if m2 else None


def _structured_fallback_answer(picks, person_hint: Optional[str]) -> str:
    """
    Deterministic answer when no LLM key is configured.
    Produces a numbered list with message ordinals and dates.
    """
    who = person_hint or (picks[0]["item"].get("user_name") if picks else "The member")
    bullets = []
    for p in picks:
        it = p["item"]
        bullets.append(
            f"{len(bullets)+1}. {it.get('message')} ({p['ordinal']}, Date: {p['pretty_date']})."
        )
    return f"{who} had several concerns, including:\n\n" + "\n".join(bullets)


def answer_with_openrouter(context: str, question: str) -> str:
    """
    Use LLM generate an answer from the provided context.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "__NO_LLM__"

    endpoint = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
    model = os.getenv("OPENROUTER_MODEL", "openrouter/sherlock-dash-alpha")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://aurora-test-nw4g.onrender.com",
        "X-Title": "Aurora QA",
    }

    system_prompt = (
        "You are an extractive assistant. Answer ONLY using the provided context.\n"
        "If the answer is not present, reply exactly: I don't know.\n\n"
        "Required format:\n"
        "answer: \"<Name> had several concerns, including:\\n\\n"
        "1. <Paraphrased concern> (Message <N>, Date: <Month DD, YYYY>).\\n"
        "2. ...\"\n"
        "• Use ONLY the message text and dates in the context.\n"
        "• Preserve the message numbering (Message N) shown in the context.\n"
        "• Keep 2–6 bullet points depending on relevance.\n"
        "• Do not invent facts, IDs, or dates.\n"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Return ONLY the answer string in the exact format above (JSON-ready, escaped as needed)."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


@router.post("/ask")
async def ask(request: Request):
    """
    Body: { "question": "..." }
    Returns: { "answer": "..." }
    """
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON body."}

    question = (body.get("question") or "").strip()
    if not question:
        return {"error": "Please provide a question."}

    try:
        person_hint = _extract_person_hint(question)
        picks = search_similar(question, person_hint=person_hint)
    except Exception as e:
        # Index not built or other retrieval error
        print(f"❌ Retrieval error: {e}")
        return {"answer": "I don't know."}

    if not picks:
        return {"answer": "I don't know."}

    context = format_numbered_context(picks)

    llm_answer = answer_with_openrouter(context, question)
    if llm_answer != "__NO_LLM__":
        return {"answer": llm_answer}

    # Fallback when no LLM key configured
    return {"answer": _structured_fallback_answer(picks, person_hint)}
