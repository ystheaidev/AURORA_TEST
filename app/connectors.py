# app/connectors.py
import os
import numpy as np
import requests
import faiss
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

AURORA_MESSAGES_API = os.getenv("AURORA_MESSAGES_API")
if not AURORA_MESSAGES_API:
    raise ValueError("AURORA_MESSAGES_API is not set in .env")

# Normalized sentence embeddings (cosine via inner product)
_HF_EMBED = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

INDEX = None          # FAISS index (flat IP)
ITEMS = []            # raw items from API (dicts)
EMBED_TEXTS = []      # structured strings used for embedding


def _iso_to_pretty(date_str: str) -> str:
    """Convert ISO 8601 to 'Month DD, YYYY'. Safe fallback on parsing errors."""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return date_str


def fetch_messages():
    """
    Fetch items from the Aurora/November7 API and keep the fields we need.
    Each item => {id, user_id, user_name, timestamp, message}
    """
    try:
        resp = requests.get(AURORA_MESSAGES_API, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        valid = []
        for m in items:
            if not isinstance(m, dict):
                continue
            text = (m.get("message") or "").strip()
            if not text:
                continue
            valid.append({
                "id": m.get("id"),
                "user_id": m.get("user_id"),
                "user_name": m.get("user_name") or "",
                "timestamp": m.get("timestamp") or "",
                "message": text,
            })
        print(f"✅ Fetched {len(valid)} messages from API.")
        return valid
    except Exception as e:
        print(f"❌ Error fetching messages: {e}")
        return []


def _to_embed_string(item: dict) -> str:
    """
    Build a richer text for retrieval that includes speaker and date.
    """
    name = item.get("user_name", "")
    when = _iso_to_pretty(item.get("timestamp", ""))
    text = item.get("message", "")
    return f"User: {name}\nDate: {when}\nMessage: {text}"


def embed_texts(texts):
    """Return float32 numpy array of embeddings."""
    if isinstance(texts, str):
        vec = _HF_EMBED.embed_query(texts)
        return np.array([vec], dtype=np.float32)
    vecs = _HF_EMBED.embed_documents(list(texts))
    return np.array(vecs, dtype=np.float32)


def build_index(items):
    """
    Build FAISS over structured strings (name + date + message).
    """
    global INDEX, ITEMS, EMBED_TEXTS
    if not items:
        raise ValueError("No messages to embed.")
    embed_ready = [_to_embed_string(it) for it in items]
    vectors = embed_texts(embed_ready)
    if vectors.size == 0:
        raise ValueError("Embedding returned no vectors.")
    dim = vectors.shape[1]
    INDEX = faiss.IndexFlatIP(dim)  # normalized embeddings → cosine via inner product
    INDEX.add(vectors.astype(np.float32))
    ITEMS = items
    EMBED_TEXTS = embed_ready
    print(f"✅ Built FAISS index with {len(items)} vectors (dim={dim})")


def _name_tokens(name: str):
    return [t for t in (name or "").split() if t]


def _matches_user(item: dict, target_name: str | None) -> bool:
    if not target_name:
        return True
    toks = _name_tokens(target_name.lower())
    uname = (item.get("user_name") or "").lower()
    return all(tok in uname for tok in toks)


def search_similar(question: str, person_hint: str | None = None):
    """
    Retrieve **all** messages ranked by similarity, optionally filtered by user.
    Returns a list of dicts:
      { item, score, rank, pretty_date, index, ordinal }
    """
    if INDEX is None:
        raise RuntimeError("Index not built yet.")

    qvec = embed_texts(question)
    # Search the entire index (NO top_k cap)
    D, I = INDEX.search(qvec, len(ITEMS))

    filtered = []
    for rank, idx in enumerate(I[0].tolist()):
        if idx < 0 or idx >= len(ITEMS):
            continue
        it = ITEMS[idx]
        if person_hint and not _matches_user(it, person_hint):
            continue
        filtered.append({
            "item": it,
            "score": float(D[0][rank]),
            "rank": rank,
            "pretty_date": _iso_to_pretty(it.get("timestamp", "")),
            "index": idx,  # stable index based on API order (or insert order)
        })

    # If person-filter yields nothing, degrade to unfiltered list
    picks = filtered if filtered else [
        {
            "item": ITEMS[idx],
            "score": float(D[0][rank]),
            "rank": rank,
            "pretty_date": _iso_to_pretty(ITEMS[idx].get("timestamp", "")),
            "index": idx,
        }
        for rank, idx in enumerate(I[0].tolist())
        if 0 <= idx < len(ITEMS)
    ]

    # Provide a deterministic "Message N" label based on original index
    for p in picks:
        p["ordinal"] = f"Message {p['index'] + 1}"

    return picks


def format_numbered_context(picks):
    """
    Produce a numbered, LLM-friendly context block:
    """
    lines = []
    for p in picks:
        it = p["item"]
        lines.append(
            f"{p['ordinal']} (User: {it.get('user_name','').strip() or 'Unknown'}, "
            f"Date: {p['pretty_date']})\n"
            f"Text: {it.get('message','').strip()}"
        )
    return "\n\n".join(lines)
