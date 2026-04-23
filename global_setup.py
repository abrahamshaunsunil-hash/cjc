import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# MODEL INIT
# -----------------------------
model_embed = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# FLATTEN JSON → STRUCTURED CHUNKS
# -----------------------------
def flatten_json(data):
    chunks = []

    def recurse(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                recurse(v, new_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                recurse(item, new_path)

        else:
            chunks.append({
                "text": f"{path}: {obj}",
                "path": path
            })

    recurse(data)
    return chunks


# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
def deduplicate_chunks(chunks):
    seen = set()
    unique = []

    for c in chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)

    return unique


# -----------------------------
# LOAD MEMORY JSON
# -----------------------------
def load_json(memory_path="veronica_memory.json"):
    mem_file = Path(memory_path)

    if not mem_file.exists():
        raise FileNotFoundError(f"{memory_path} not found")

    return json.loads(mem_file.read_text(encoding="utf-8"))


# -----------------------------
# EMBEDDINGS
# -----------------------------
def compute_embeddings(chunks, emb_path="chunk_embs.npy", force_recompute=False):
    emb_file = Path(emb_path)

    if emb_file.exists() and not force_recompute:
        return np.load(emb_path)

    texts = [c["text"] for c in chunks]

    embs = model_embed.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(emb_path, embs)
    return embs


# -----------------------------
# SEARCH (SEMANTIC)
# -----------------------------
def search_memory(query, chunks, chunk_embs, top_k=5):
    query_emb = model_embed.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(chunk_embs, query_emb)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [
        {
            "text": chunks[i]["text"],
            "path": chunks[i]["path"],
            "score": float(scores[i])
        }
        for i in top_indices
    ]


# -----------------------------
# 🔥 MAPPING LOGIC (KEY FIX)
# -----------------------------
def resolve_stream_mapping(query, mappings):
    query = query.lower()

    for key, streams in mappings.items():
        if key in query:
            return streams

    return None


# -----------------------------
# 💰 FEES RESPONSE (DETERMINISTIC)
# -----------------------------
def get_stream_fees(query, data):
    mappings = data.get("mappings", {})
    fees = data.get("fees", {})

    mapped_streams = resolve_stream_mapping(query, mappings)

    if not mapped_streams:
        return None

    results = []
    for stream in mapped_streams:
        if stream in fees:
            results.append(f"{stream}: ₹{fees[stream]}")

    return results if results else None


# -----------------------------
# 🧠 MAIN RESPONSE ROUTER
# -----------------------------
def get_response(query, data, chunks, chunk_embs):
    query_lower = query.lower()

    # 1. HANDLE FEES FIRST (priority logic)
    if "fee" in query_lower or "fees" in query_lower:
        fee_result = get_stream_fees(query, data)
        if fee_result:
            return "\n".join(fee_result)

    # 2. FALLBACK → SEMANTIC SEARCH
    results = search_memory(query, chunks, chunk_embs)

    return "\n".join([r["text"] for r in results])


# -----------------------------
# STARTUP
# -----------------------------
DATA = load_json()

CHUNKS = deduplicate_chunks(flatten_json(DATA))
CHUNK_EMBS = compute_embeddings(CHUNKS)
