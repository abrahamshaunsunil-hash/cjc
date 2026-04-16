# global_setup.py (run once at app start)
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# 👇 ADD THIS HERE
def flatten_json(data):
    lines = []

    def recurse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, f"{prefix} {k}".strip())
        elif isinstance(obj, list):
            for item in obj:
                recurse(item, prefix)
        else:
            lines.append(f"{prefix}: {obj}")

    recurse(data)
    return lines

def load_memory_and_precompute(memory_path="veronica_memory.json", emb_path="chunk_embs.npy"):
    mem_file = Path(memory_path)
    data = json.loads(mem_file.read_text(encoding="utf-8"))
    chunks = flatten_json(data)
    # compute & save embeddings only if not present
    emb_file = Path(emb_path)
    if emb_file.exists():
        chunk_embs = np.load(emb_path)
    else:
        chunk_embs = model_embed.encode(chunks, convert_to_tensor=False)  # returns numpy array
        np.save(emb_path, chunk_embs)
    return chunks, chunk_embs

# At app startup:
CHUNKS, CHUNK_EMBS = load_memory_and_precompute()
