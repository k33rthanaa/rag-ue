import argparse
import faiss
import gzip
import json
from pathlib import Path
from utils import load_config, load_model_and_tokenizer, encode_batch, get_device, setup_logging

def load_metadata(meta_path: Path):
    """Load metadata JSONL.GZ into a list."""
    records = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    return records

def query_shard(shard_id, query, index, metadata, tokenizer, model, device, max_length=512, top_k=2):
    """Helper function to query a single shard and retrieve top K results"""
    # Encode the query
    q_emb = encode_batch([query], tokenizer, model, device, max_length=max_length)

    # Normalize if used during indexing
    faiss.normalize_L2(q_emb)

    # Perform FAISS search
    D, I = index.search(q_emb, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        title = meta.get("title", "")
        text = meta.get("contents", "")
        doc_id = meta.get("dataset_id") or meta.get("id") or meta.get("global_id")

        snippet = text[:200].replace("\n", " ")
        if len(text) > 200:
            snippet += "…"

        results.append({
            "rank": rank,
            "score": float(score),
            "doc_id": doc_id,
            "title": title,
            "snippet": snippet,
        })

    return results

def main(args):
    # -------------------------------
    # Load config
    # -------------------------------
    cfg = load_config(args.config)

    # Setup logging
    root = Path(__file__).resolve().parents[1]
    log_dir = root / cfg.get("paths", {}).get("logs_dir", "outputs/logs")
    logger = setup_logging(str(log_dir), cfg.get("runtime", {}).get("log_level", "INFO"))

    # Model + Tokenizer
    tokenizer, model, device = load_model_and_tokenizer(cfg)
    max_length = cfg.get("max_length", 512)

    # -------------------------------
    # Iterate through all shards
    # -------------------------------
    output_root = root / cfg["paths"]["output_root"]
    all_results = []
    for shard_id in range(args.total_shards):
        shard_dir = output_root / f"shard_{shard_id:04d}"
        index_path = shard_dir / f"shard_{shard_id:04d}.index"
        meta_path = shard_dir / f"shard_{shard_id:04d}.meta.jsonl.gz"

        if not index_path.exists():
            logger.warning(f"Index not found for shard {shard_id}, skipping...")
            continue
        if not meta_path.exists():
            logger.warning(f"Meta file not found for shard {shard_id}, skipping...")
            continue

        logger.info(f"📁 Using shard {shard_id} directory: {shard_dir}")
        logger.info(f"   Index: {index_path}")
        logger.info(f"   Meta: {meta_path}\n")

        # Load FAISS index and metadata
        logger.info(f"🔧 Loading FAISS index for shard {shard_id}...")
        index = faiss.read_index(str(index_path))
        logger.info(f"✅ Loaded index with {index.ntotal} vectors for shard {shard_id}.\n")

        logger.info(f"🔧 Loading metadata for shard {shard_id}...")
        metadata = load_metadata(meta_path)
        logger.info(f"✅ Loaded {len(metadata)} metadata records for shard {shard_id}.\n")

        # Query the shard and retrieve top 2 results
        logger.info(f"🔎 Querying shard {shard_id}...")
        results = query_shard(shard_id, args.query, index, metadata, tokenizer, model, device, max_length)

        all_results.extend(results)
        logger.info(f"Found {len(results)} results from shard {shard_id}.\n")

    # -------------------------------
    # Display results
    # -------------------------------
    if not all_results:
        logger.info("No results found across all shards.")
    else:
        logger.info(f"Found {len(all_results)} results across all shards.")
        for r in all_results:
            logger.info(f"Rank {r['rank']}: {r['score']} | {r['title']} | {r['snippet']}")

        # Print the aggregated results in a table
        from tabulate import tabulate
        table = [[r['rank'], f"{r['score']:.4f}", r['doc_id'], r['title'], r['snippet']] for r in all_results]
        print(tabulate(table, headers=["Rank", "Score", "Doc ID", "Title", "Snippet"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query all shards.")
    parser.add_argument("--query", type=str, required=True, help="The query string.")
    parser.add_argument("--total_shards", type=int, required=True, help="Total number of shards.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of results to return per shard.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (default: configs/default.yaml)")
    args = parser.parse_args()
    main(args)
