#!/usr/bin/env python3
"""
Index Wikipedia-2018 using Contriever with BATCH processing (no chunking).

- Reads all *.jsonl under --corpus_dir
- Tokenizes with truncation to --max_length
- Encodes in batches (--bs) with HF Transformers
- Mean-pools with NumPy and L2-normalizes with NumPy
- Adds to FAISS incrementally (low RAM)

Outputs:
  index/contriever/contriever.faiss
  index/contriever/meta.parquet  (title + raw text used)
"""

import os, sys, glob, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

# ---------------- device helpers ----------------
def pick_device() -> torch.device:
    forced = os.environ.get("FORCE_DEVICE", "").lower()
    if forced in {"cpu","cuda","mps"}:
        return torch.device(forced)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def default_dtype_for(device: torch.device):
    return torch.float16 if device.type in ("mps","cuda") else torch.float32

# ---------------- encoding (batch, no chunking) ----------------
@torch.no_grad()
def encode_batch(tok, model, device, texts, max_length: int) -> np.ndarray:
    """
    Encode a list of raw texts as one batch.
    Returns float32 numpy array [B, H], mean-pooled & L2-normalized (NumPy).
    """
    if not texts:
        return np.zeros((0, model.config.hidden_size), dtype="float32")

    # tokenize on device (HF)
    t = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # forward pass (HF)
    out = model(**t, return_dict=True).last_hidden_state  # [B, T, H]

    # to NumPy, then pool + normalize with NumPy
    last_np = out.detach().to("cpu").numpy()              # float32/16 -> CPU numpy
    pooled  = last_np.mean(axis=1)                        # [B, H] mean over tokens
    norms   = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
    pooled  = (pooled / norms).astype("float32")
    return pooled

# ---------------- data reader ----------------
def iter_jsonl_files(corpus_dir: str):
    files = sorted(glob.glob(os.path.join(corpus_dir, "**/*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"[error] no JSONL found under {corpus_dir}")
    return files

def iter_rows(files, max_files: int):
    count = 0
    for fp in files:
        if max_files > 0 and count >= max_files:
            break
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text  = (obj.get("text")  or "").strip()
                title = (obj.get("title") or "").strip()
                if not text:
                    continue
                yield title, text
        count += 1

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", default="data/wiki18")
    ap.add_argument("--index_dir",  default="index/contriever")
    ap.add_argument("--model_dir",  default="models/contriever")
    ap.add_argument("--max_files",  type=int, default=2, help="0 = all files")
    ap.add_argument("--bs",         type=int, default=8, help="batch size")
    ap.add_argument("--max_length", type=int, default=256, help="token limit per doc (truncation)")
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = pick_device()
    dtype  = default_dtype_for(device)
    print(f"[info] device={device} dtype={dtype}")

    # HF models (use dtype= to avoid deprecation)
    tok   = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir, dtype=dtype).to(device).eval()

    # prepare FAISS
    dim   = model.config.hidden_size
    index = faiss.IndexFlatIP(dim)
    meta  = []  # will store title/text rows incrementally

    files = iter_jsonl_files(args.corpus_dir)

    # batch up titles/texts
    batch_titles, batch_texts = [], []
    total = 0

    for title, text in tqdm(iter_rows(files, args.max_files), desc="encode (batch, no-chunk)"):
        batch_titles.append(title)
        batch_texts.append(text)
        if len(batch_texts) >= args.bs:
            vecs = encode_batch(tok, model, device, batch_texts, args.max_length)
            if vecs.shape[0]:
                index.add(vecs)  # incremental FAISS add
                meta.extend([{"title": t, "passage": s} for t, s in zip(batch_titles, batch_texts)])
                total += vecs.shape[0]
            batch_titles, batch_texts = [], []

    # flush last partial batch
    if batch_texts:
        vecs = encode_batch(tok, model, device, batch_texts, args.max_length)
        if vecs.shape[0]:
            index.add(vecs)
            meta.extend([{"title": t, "passage": s} for t, s in zip(batch_titles, batch_texts)])
            total += vecs.shape[0]

    if total == 0:
        raise SystemExit("[error] no vectors produced")

    # write outputs
    faiss_path = os.path.join(args.index_dir, "contriever.faiss")
    meta_path  = os.path.join(args.index_dir, "meta.parquet")
    faiss.write_index(index, faiss_path)
    pd.DataFrame(meta).to_parquet(meta_path)

    print(f"[ok] wrote index -> {faiss_path}")
    print(f"[ok] wrote meta  -> {meta_path}")
    print(f"[done] ntotal={index.ntotal} dim={dim}")

if __name__ == "__main__":
    main()
