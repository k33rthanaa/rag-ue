#!/usr/bin/env python3
"""
Index Wikipedia-2018 using Contriever with BATCH processing (no chunking).
Supports *.jsonl and *.jsonl.gz.

Outputs:
  <index_dir>/contriever.faiss
  <index_dir>/meta.parquet  (title + text used)
"""

import os
import sys
import glob
import json
import argparse
import gzip

# Make repo root importable if you later add helpers/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel


# ---------------- device helpers ----------------
def pick_device() -> torch.device:
    """Pick device: FORCE_DEVICE env > MPS > CUDA > CPU."""
    forced = os.environ.get("FORCE_DEVICE", "").lower()
    if forced in {"cpu", "cuda", "mps"}:
        return torch.device(forced)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_dtype_for(device: torch.device):
    """Use half precision on GPU/MPS, full on CPU."""
    return torch.float16 if device.type in ("mps", "cuda") else torch.float32


# ---------------- encoding (batch, no chunking) ----------------
@torch.no_grad()
def encode_batch(tok, model, device, texts, max_length: int) -> np.ndarray:
    """
    Encode a list of raw texts in one batch.
    Returns [B, H] float32 L2-normalized.
    """
    if not texts:
        return np.zeros((0, model.config.hidden_size), dtype="float32")

    # Tokenize with truncation on device
    t = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # HF forward pass
    out = model(**t, return_dict=True).last_hidden_state  # [B, T, H]

    # To NumPy for pooling + normalization
    arr = out.detach().to("cpu").numpy()                  # [B, T, H]
    pooled = arr.mean(axis=1)                             # [B, H]
    norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
    pooled = (pooled / norms).astype("float32")           # [B, H]
    return pooled


# ---------------- data readers ----------------
def iter_jsonl_files(corpus_dir: str):
    """Return sorted list of all *.jsonl and *.jsonl.gz under corpus_dir."""
    files = sorted(
        glob.glob(os.path.join(corpus_dir, "**", "*.jsonl"), recursive=True)
        + glob.glob(os.path.join(corpus_dir, "**", "*.jsonl.gz"), recursive=True)
    )
    if not files:
        raise SystemExit(f"[error] no JSONL/JSONL.GZ found under {corpus_dir}")
    return files


def iter_rows(files, max_files: int):
    """
    Yield (title, text) per line across up to max_files files (0 = all).

    Uses errors='ignore' so that weird bytes / bad encodings don't crash
    the indexer (avoids UnicodeDecodeError).
    """
    count = 0
    for fp in files:
        if max_files > 0 and count >= max_files:
            break

        opener = gzip.open if fp.endswith(".gz") else open
        mode = "rt" if fp.endswith(".gz") else "r"

        # errors='ignore' avoids UTF-8 issues like byte 0x80 crashes
        with opener(fp, mode, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = (obj.get("text") or "").strip()
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
    ap.add_argument("--max_files",  type=int, default=2,
                    help="how many files to read (0 = all)")
    ap.add_argument("--bs",         type=int, default=64,
                    help="batch size (docs per forward)")
    ap.add_argument("--max_length", type=int, default=256,
                    help="token cap per doc (truncation)")
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = pick_device()
    dtype = default_dtype_for(device)
    print(f"[info] device={device} dtype={dtype}")

    # HF models
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir, dtype=dtype).to(device).eval()

    dim = model.config.hidden_size
    index = faiss.IndexFlatIP(dim)
    meta = []

    files = iter_jsonl_files(args.corpus_dir)

    batch_titles, batch_texts = [], []
    total = 0

    for title, text in tqdm(iter_rows(files, args.max_files),
                            desc="encode (batch, no-chunk)"):
        batch_titles.append(title)
        batch_texts.append(text)

        if len(batch_texts) >= args.bs:
            vecs = encode_batch(tok, model, device, batch_texts, args.max_length)
            if vecs.size:
                index.add(vecs)
                meta.extend(
                    {"title": t, "passage": s}
                    for t, s in zip(batch_titles, batch_texts)
                )
                total += vecs.shape[0]
            batch_titles, batch_texts = [], []

    # Flush tail batch
    if batch_texts:
        vecs = encode_batch(tok, model, device, batch_texts, args.max_length)
        if vecs.size:
            index.add(vecs)
            meta.extend(
                {"title": t, "passage": s}
                for t, s in zip(batch_titles, batch_texts)
            )
            total += vecs.shape[0]

    if total == 0:
        raise SystemExit("[error] no vectors produced")

    faiss_path = os.path.join(args.index_dir, "contriever.faiss")
    meta_path = os.path.join(args.index_dir, "meta.parquet")

    faiss.write_index(index, faiss_path)
    pd.DataFrame(meta).to_parquet(meta_path)

    print(f"[ok] wrote index -> {faiss_path}")
    print(f"[ok] wrote meta  -> {meta_path}")
    print(f"[done] ntotal={index.ntotal} dim={dim}")


if __name__ == "__main__":
    main()
