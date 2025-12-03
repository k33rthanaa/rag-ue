import faiss
import json
import torch
from transformers import AutoModel, AutoTokenizer
from utils import load_config, load_model_and_tokenizer, encode_batch, get_device, setup_logging
import argparse
from pathlib import Path
import gzip

def load_metadata(meta_path):
    """Load metadata from JSONL.GZ file"""
    records = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:  # Open the .gz file in text mode
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def query_shard(shard_id, query, index, metadata, tokenizer, model, device, max_length=512, top_k=2):
    """Query a single shard and retrieve the top K results"""
    # Encode the query
    q_emb = encode_batch([query], tokenizer, model, device, max_length=max_length)
    faiss.normalize_L2(q_emb)  # Normalize if used during indexing

    # Perform FAISS search
    D, I = index.search(q_emb, top_k)  # D: scores, I: indices
    results = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx >= len(metadata): continue
        meta = metadata[idx]
        title = meta.get("title", "")
        text = meta.get("contents", "")
        doc_id = meta.get("dataset_id", meta.get("id", meta.get("global_id")))

        snippet = text[:200] + ("…" if len(text) > 200 else "")
        results.append({
            "rank": rank,
            "score": float(score),
            "doc_id": doc_id,
            "title": title,
            "snippet": snippet,
        })

    return results

def main(args):
    # Load config
    cfg = load_config(args.config)

    # Setup logging
    logger = setup_logging(str(cfg.get("paths", {}).get("logs_dir", "outputs/logs")), "INFO")

    # Load model and tokenizer for retrieval (Contriever model)
    device = get_device()  # Can be "cuda" or "cpu"
    retriever_tokenizer, retriever_model, device = load_model_and_tokenizer(cfg, task_type="retriever")

    # Iterate through all shards and retrieve results
    output_root = Path(cfg["paths"]["output_root"])
    all_results = []

    for shard_id in range(args.total_shards):
        shard_dir = output_root / f"shard_{shard_id:04d}"
        index_path = shard_dir / f"shard_{shard_id:04d}.index"
        meta_path = shard_dir / f"shard_{shard_id:04d}.meta.jsonl.gz"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(f"Index or metadata not found for shard {shard_id}, skipping...")
            continue

        logger.info(f"📁 Using shard {shard_id} directory: {shard_dir}")
        index = faiss.read_index(str(index_path))
        metadata = load_metadata(meta_path)

        logger.info(f"🔎 Querying shard {shard_id}...")
        results = query_shard(shard_id, args.query, index, metadata, retriever_tokenizer, retriever_model, device, top_k=args.top_k)
        all_results.extend(results)

    # Rank all results across shards
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Display ranked results
    if not all_results:
        logger.info("No results found across all shards.")
    else:
        logger.info(f"Found {len(all_results)} results across all shards.")
        for r in all_results:
            logger.info(f"Rank {r['rank']}: {r['score']} | {r['title']} | {r['snippet']}")
        from tabulate import tabulate
        table = [[r['rank'], f"{r['score']:.4f}", r['doc_id'], r['title'], r['snippet']] for r in all_results]
        print(tabulate(table, headers=["Rank", "Score", "Doc ID", "Title", "Snippet"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query all shards and rank results.")
    parser.add_argument("--query", type=str, required=True, help="The query string.")
    parser.add_argument("--total_shards", type=int, required=True, help="Total number of shards.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of results to return per shard.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)

