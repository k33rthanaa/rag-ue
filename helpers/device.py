
python -m scripts.index_wiki18
cat > scripts/index_wiki18.py << 'PY'
#!/usr/bin/env python3
"""
Build a FAISS index over the Wikipedia-2018 JSONL corpus using Contriever.

- Works on Mac (Apple Silicon MPS), CUDA, or CPU.
- Mean-pools Contriever hidden states, L2 normalizes, and indexes with FAISS (IP).
- Saves:
    index/contriever/contriever.faiss
    index/contriever/meta.parquet  (title + passage text per vector)

You can control how much to index via CLI flags, e.g.:
  python -m scripts.index_wiki18 --max_files 2 --bs 8 --max_toks 256 --stride 40
"""

import sys, os, glob, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# ---------- device helpers ----------
def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def default_dtype_for(device: torch.device):
    return torch.float16 if device.type in ("mps", "cuda") else torch.float32

# ---------- chunking ----------
def passages_from_text(tok, text: str, max_toks: int = 256, stride: int = 40):
    """
    Slice a long document into overlapping token windows (decode back to text).
    """
    if not text:
        return []
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return []
    chunks, i = [], 0
    step = max(1, max_toks - stride)
    while i < len(ids):
        chunk_ids = ids[i:i + max_toks]
        if not chunk_ids:
            break
        chunks.append(tok.decode(chunk_ids))
        i += step
    return chunks

# ---------- embedding ----------
@torch.no_grad()
def encode_texts(tok, enc, device, texts, bs: int = 16):
    """
    Mean-pool last_hidden_state, then L2 normalize.
    Returns float32 CPU numpy array [N, D].
    """
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        t = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        last = enc(**t).last_hidden_state              # [B, T, H]
        emb  = last.mean(dim=1)                        # [B, H]
        emb  = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.to("cpu", dtype=torch.float32).numpy())
    return np.vstack(out) if out else np.zeros((0, enc.config.hidden_size), dtype="float32")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", default="data/wiki18", help="Root of wiki-18 JSONL dataset")
    ap.add_argument("--index_dir",  default="index/contriever", help="Where to write FAISS + meta")
    ap.add_argument("--model_dir",  default="models/contriever", help="Path to contriever weights")
    ap.add_argument("--max_files",  type=int, default=2, help="How many JSONL files to index (for smoke tests)")
    ap.add_argument("--bs",         type=int, default=8, help="Encode batch size")
    ap.add_argument("--max_toks",   type=int, default=256, help="Tokens per chunk")
    ap.add_argument("--stride",     type=int, default=40, help="Overlap between chunks")
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    device = pick_device()
    dtype  = default_dtype_for(device)
    print(f"[info] device={device} dtype={dtype}")

    # tokenizer/encoder
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    enc = AutoModel.from_pretrained(args.model_dir, torch_dtype=dtype).to(device).eval()

    # collect files
    files = sorted(glob.glob(os.path.join(args.corpus_dir, "**/*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"[error] no JSONL files found under: {args.corpus_dir}")
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f"[info] found {len(files)} jsonl file(s) to index")

    meta_rows = []
    all_vecs  = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"chunk+embed {os.path.basename(fp)}"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text  = obj.get("text", "") or ""
                title = obj.get("title", "") or ""
                chunks = passages_from_text(tok, text, max_toks=args.max_toks, stride=args.stride)
                if not chunks:
                    continue
                meta_rows.extend([{"title": title, "passage": c} for c in chunks])
                vecs = encode_texts(tok, enc, device, chunks, bs=args.bs)
                all_vecs.append(vecs)

    if not all_vecs:
        raise SystemExit("[error] no vectors produced — check corpus paths and tokenizer.")

    vecs = np.vstack(all_vecs).astype("float32")
    print(f"[info] total passages: {vecs.shape[0]}  dim: {vecs.shape[1]}")

    # FAISS index (inner product; embeddings are L2-normalized)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss_path = os.path.join(args.index_dir, "contriever.faiss")
    faiss.write_index(index, faiss_path)

    # metadata (title + passage per vector)
    meta_df = pd.DataFrame(meta_rows)
    meta_path = os.path.join(args.index_dir, "meta.parquet")
    meta_df.to_parquet(meta_path)

    print(f"[ok] wrote index: {faiss_path}")
    print(f"[ok] wrote meta:  {meta_path}")
    print(f"[done] ntotal={index.ntotal}")

if __name__ == "__main__":
    # Helpful on Mac to avoid hard failures for missing MPS ops
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
