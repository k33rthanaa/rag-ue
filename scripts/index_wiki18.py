#!/usr/bin/env python3
import os, sys, argparse, faiss, numpy as np, pandas as pd, torch
from tqdm import tqdm
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, AutoModel

def pick_device():
    forced = os.environ.get("FORCE_DEVICE","").lower()
    if forced in {"cpu","cuda","mps"}: return torch.device(forced)
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def default_dtype_for(device): return torch.float16 if device.type in ("mps","cuda") else torch.float32

@torch.no_grad()
def encode_batch(tok, model, device, texts, max_length: int) -> np.ndarray:
    if not texts: return np.zeros((0, model.config.hidden_size), dtype="float32")
    t = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    last = model(**t, return_dict=True).last_hidden_state            # [B,T,H]
    arr  = last.detach().to("cpu").numpy()                           # -> numpy
    pooled = arr.mean(axis=1)                                        # [B,H]
    norms  = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
    return (pooled / norms).astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", default="data/wiki18")
    ap.add_argument("--index_dir",  default="index/contriever")
    ap.add_argument("--model_dir",  default="models/contriever")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--limit_docs", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")

    device = pick_device()
    dtype  = default_dtype_for(device)
    print(f"[info] device={device} dtype={dtype}")

    tok   = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir, dtype=dtype).to(device).eval()
    dim   = model.config.hidden_size
    index = faiss.IndexFlatIP(dim)
    meta_rows = []

    # ---- Hugging Face Datasets (streaming over JSONL) ----
    # Collect all jsonl files under corpus_dir
    data_files = {"train": [str(p) for p in sorted(__import__("glob").glob(os.path.join(args.corpus_dir, "**/*.jsonl"), recursive=True))]}
    if not data_files["train"]:
        raise SystemExit(f"[error] no JSONL found under {args.corpus_dir}")

    ds: IterableDataset = load_dataset("json", data_files=data_files, split="train", streaming=True)
    # Optionally limit docs (for smoke tests)
    if args.limit_docs > 0:
        ds = ds.take(args.limit_docs)

    # Batch iteration
    batch_titles, batch_texts = [], []
    total = 0
    with tqdm(unit="docs", desc="encode (HF Datasets, batch, no-chunk)") as pbar:
        for row in ds:
            text  = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if not text: 
                continue
            batch_titles.append(title); batch_texts.append(text)
            if len(batch_texts) >= args.bs:
                vecs = encode_batch(tok, model, device, batch_texts, args.max_length)
                if vecs.size:
                    index.add(vecs)
                    meta_rows.extend({"title": t, "passage": s} for t, s in zip(batch_titles, batch_texts))
                    total += vecs.shape[0]
                batch_titles, batch_texts = [], []
                pbar.update(args.bs)

        # flush tail
        if batch_texts:
            vecs = encode_batch(tok, model, device, batch_texts, args.max_length)
            if vecs.size:
                index.add(vecs)
                meta_rows.extend({"title": t, "passage": s} for t, s in zip(batch_titles, batch_texts))
                total += vecs.shape[0]
            pbar.update(len(batch_texts))

    if total == 0:
        raise SystemExit("[error] no vectors produced")

    faiss_path = os.path.join(args.index_dir, "contriever.faiss")
    meta_path  = os.path.join(args.index_dir, "meta.parquet")
    faiss.write_index(index, faiss_path)
    pd.DataFrame(meta_rows).to_parquet(meta_path)
    print(f"[ok] wrote index -> {faiss_path}")
    print(f"[ok] wrote meta  -> {meta_path}")
    print(f"[done] ntotal={index.ntotal} dim={dim}")

if __name__ == "__main__":
    main()

