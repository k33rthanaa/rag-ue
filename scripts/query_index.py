import argparse
import gzip
import json
from pathlib import Path

import faiss
from tabulate import tabulate  # optional, for nicer printing (pip install tabulate)

from utils import load_config, load_model_and_tokenizer, encode_batch, get_device


def load_metadata(meta_path: Path):
    """
    Load metadata JSONL.GZ into a list.

    We assume that documents were written in the same order as
    embeddings were added to FAISS, so FAISS id == index in this list.
    """
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


def main(args):
    # -------------------------------
    # Load config
    # -------------------------------
    cfg = load_config(args.config)

    # Paths
    root = Path(__file__).resolve().parents[1]
    output_root = root / cfg["paths"]["output_root"]

    shard_dir = output_root / f"shard_{args.shard_id:04d}"
    index_path = shard_dir / f"shard_{args.shard_id:04d}.index"
    meta_path = shard_dir / f"shard_{args.shard_id:04d}.meta.jsonl.gz"

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    print(f"📁 Using shard directory: {shard_dir}")
    print(f"   Index: {index_path}")
    print(f"   Meta : {meta_path}\n")

    # -------------------------------
    # Load FAISS index + metadata
    # -------------------------------
    print("🔧 Loading FAISS index...")
    index = faiss.read_index(str(index_path))
    print(f"✅ Index loaded with {index.ntotal} vectors.\n")

    print("🔧 Loading metadata...")
    metadata = load_metadata(meta_path)
    print(f"✅ Loaded {len(metadata)} metadata records.\n")

    if len(metadata) != index.ntotal:
        print("⚠️ Warning: metadata count and index size differ.")
        print("   Retrieval may still work, but ids might not align perfectly.\n")

    # -------------------------------
    # Load model + tokenizer
    # -------------------------------
    print("🔧 Loading model + tokenizer...")
    tokenizer, model, device = load_model_and_tokenizer(cfg)
    max_length = cfg.get("max_length", 512)
    print(f"✅ Model loaded on device: {device}\n")

    print("🔎 Ready to query.")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # Encode query
        q_emb = encode_batch(
            [query],
            tokenizer,
            model,
            device,
            max_length=max_length,
        )

        # Normalize if used during indexing
        if cfg["index"]["normalize"]:
            faiss.normalize_L2(q_emb)

        # Search
        k = args.top_k
        D, I = index.search(q_emb, k)  # D: scores, I: indices

        scores = D[0]
        indices = I[0]

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0 or idx >= len(metadata):
                continue
            meta = metadata[idx]
            title = meta.get("title") or ""
            text = meta.get("contents") or ""
            doc_id = meta.get("dataset_id") or meta.get("id") or meta.get("global_id")

            snippet = text[:200].replace("\n", " ")
            if len(text) > 200:
                snippet += "…"

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "doc_id": doc_id,
                    "title": title,
                    "snippet": snippet,
                }
            )

        if not results:
            print("No results found.\n")
            continue

        # Pretty print results
        table = [
            [r["rank"], f"{r['score']:.4f}", r["doc_id"], r["title"], r["snippet"]]
            for r in results
        ]
        print()
        print(
            tabulate(
                table,
                headers=["Rank", "Score", "Doc ID", "Title", "Snippet"],
                tablefmt="fancy_grid",
                showindex=False,
            )
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a single FAISS shard index.")
    parser.add_argument("--shard_id", type=int, required=True, help="Shard ID to query.")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return per query.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()
    main(args)

