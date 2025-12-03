import argparse
import gzip
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from utils import load_config, load_model_and_tokenizer, encode_batch, get_device, setup_logging


def get_shard_range(shard_id: int, shard_size: int):
    """
    Return the global document range for this shard.
    """
    start = shard_id * shard_size
    end = (shard_id + 1) * shard_size
    return start, end


def main(args):
    # -------------------------------
    # Load config + prepare paths
    # -------------------------------
    cfg = load_config(args.config)
    
    # Setup logging
    root = Path(__file__).resolve().parents[1]
    log_dir = root / cfg.get("paths", {}).get("logs_dir", "outputs/logs")
    logger = setup_logging(str(log_dir), cfg.get("runtime", {}).get("log_level", "INFO"))

    shard_size = args.shard_size or cfg["sharding"]["shard_size"]
    save_texts = cfg["sharding"]["save_texts"]

    dataset_path = cfg["dataset"]["local_path"]
    dataset_path = root / dataset_path

    # Resolve output directories
    output_root = root / cfg["paths"]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    # Folder for this shard
    shard_dir = output_root / f"shard_{args.shard_id:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    index_path = shard_dir / f"shard_{args.shard_id:04d}.index"
    meta_path = shard_dir / f"shard_{args.shard_id:04d}.meta.jsonl.gz"

    logger.info(f"\n📌 Building shard: {args.shard_id}")
    logger.info(f"   Docs per shard: {shard_size}")
    logger.info(f"   Dataset:        {dataset_path}")
    logger.info(f"   Output index:   {index_path}")
    logger.info(f"   Output meta:    {meta_path}")

    # -------------------------------
    # Load model + tokenizer + device
    # -------------------------------
    tokenizer, model, device = load_model_and_tokenizer(cfg)
    batch_size = cfg["batch_size"]
    max_length = cfg.get("max_length", 512)

    # -------------------------------
    # Determine shard doc range
    # -------------------------------
    shard_start, shard_end = get_shard_range(args.shard_id, shard_size)
    logger.info(f"➡️ Shard covers global docs [{shard_start} .. {shard_end})\n")

    # -------------------------------
    # Build FAISS index (lazy init)
    # -------------------------------
    index = None
    next_local_id = 0

    # -------------------------------
    # Prepare metadata writer
    # -------------------------------
    meta_f = gzip.open(meta_path, "wt", encoding="utf-8")

    # -------------------------------
    # Read dataset streaming
    # -------------------------------
    global_id = 0                       # count valid docs in dataset
    batch_texts = []
    batch_meta = []

    with gzip.open(dataset_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(tqdm(f, desc="Reading dataset")):

            if global_id >= shard_end:
                break

            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = obj.get("contents")
            title = obj.get("title", "")

            if not text:
                continue

            # Only keep docs inside this shard's range
            if global_id < shard_start:
                global_id += 1
                continue

            # Build metadata entry
            meta = {
                "local_id": next_local_id,
                "global_id": global_id,
                "dataset_id": obj.get("id", None),
                "line_no": line_no,
                "title": title,
                "contents": text,        # <-- full text for RAG
            }

            batch_texts.append(text)
            batch_meta.append(meta)
            next_local_id += 1
            global_id += 1

            # Encode + add batch to index
            if len(batch_texts) >= batch_size or global_id == shard_end:
                embs = encode_batch(
                    batch_texts,
                    tokenizer,
                    model,
                    device,
                    max_length=max_length,
                )  # shape: [B, dim]

                # Normalize for cosine similarity
                if cfg["index"]["normalize"]:
                    faiss.normalize_L2(embs)

                # Initialize index lazily
                if index is None:
                    dim = embs.shape[1]
                    logger.info(f"🔧 Initializing FAISS index (dim={dim})")
                    index = faiss.IndexFlatIP(dim)

                index.add(embs)

                # Write metadata (one JSON object per line)
                for m in batch_meta:
                    meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")

                batch_texts.clear()
                batch_meta.clear()

    meta_f.close()

    # -------------------------------
    # Save FAISS index
    # -------------------------------
    if index is None:
        raise RuntimeError("No documents indexed in this shard.")

    faiss.write_index(index, str(index_path))
    logger.info(f"\n✅ Shard {args.shard_id} completed.")
    logger.info(f"📦 Saved FAISS index: {index_path}")
    logger.info(f"📝 Saved metadata:    {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index shard.")
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--shard_size", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    main(args)

