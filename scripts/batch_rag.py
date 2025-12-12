"""
Batch RAG with local Qwen2.5-7B-Instruct
Generates answers for 50 queries using a local Qwen model instead of external APIs.

This version:
- Uses sharded FAISS indices via rag_query1.retrieve_documents (memory friendly)
- Uses local Qwen2.5-7B-Instruct for answer generation via rag_query1.generate_answer
"""

import json
from pathlib import Path

from tqdm import tqdm

from rag_query_local import retrieve_documents, generate_answer as generate_qwen_answer
from utils import load_config, load_model_and_tokenizer, load_model_and_tokenizer2


def main():
    """Process all queries using local Qwen model for answer generation."""

    # Configuration
    # Answering model name comes from configs/default.yaml -> answering_model_name
    MODEL_NAME = "Qwen2.5-7B-Instruct-local"
    INPUT_FILE = "data/factscore_bio_50.jsonl"
    OUTPUT_DIR = Path("outputs")
    OUTPUT_FILE = OUTPUT_DIR / f"rag_answers_api_{MODEL_NAME.split('/')[-1]}.jsonl"

    # Retrieval settings
    NUM_DOCS_RETRIEVE = 25  # Retrieve top 25 documents
    NUM_DOCS_FOR_ANSWER = 15  # Use top 15 for answer generation

    print("\n" + "=" * 80)
    print("Batch RAG with local Qwen2.5-7B-Instruct")
    print("=" * 80)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Retrieving: {NUM_DOCS_RETRIEVE} documents")
    print(f"Using: {NUM_DOCS_FOR_ANSWER} documents for answer generation")
    print("=" * 80 + "\n")

    # Load queries
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    print(f"Loaded {len(queries)} queries")

    # Load retriever model (only retriever, no answering model!)
    cfg = load_config("configs/default.yaml")
    retriever_tokenizer, retriever_model, device_cpu = load_model_and_tokenizer(
        cfg, "retriever"
    )
    print("Retriever model loaded")

    # Load answering model (Qwen) once
    print("Loading Qwen answering model...")
    answering_tokenizer, answering_model, device_qwen = load_model_and_tokenizer2(
        cfg, task_type="answering"
    )
    print("Qwen answering model loaded\n")

    # Discover shard indices (memory friendly)
    print("Discovering shard indices...")
    output_root = Path(cfg["paths"]["output_root"])
    shard_dirs = [
        d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("shard_")
    ]
    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard directories found under {output_root}. "
            "Make sure you have built shard indices (shard_0000, shard_0001, ...)."
        )
    total_shards = len(shard_dirs)
    print(f"Found {total_shards} shard directories\n")

    # Generate answers
    results = []
    errors = []

    for query_obj in tqdm(queries, desc="Processing queries"):
        query = query_obj["prompt"]
        qid = query_obj["qid"]

        try:
            # Retrieve documents across all shards and take top N
            all_results = retrieve_documents(
                query,
                cfg,
                retriever_tokenizer,
                retriever_model,
                device_cpu,
                total_shards=total_shards,
                top_k_per_shard=3,
            )

            docs = all_results[:NUM_DOCS_RETRIEVE]

            # Generate answer using local Qwen with top 15 documents
            answer = generate_qwen_answer(
                query,
                docs[:NUM_DOCS_FOR_ANSWER],
                answering_tokenizer,
                answering_model,
                device_qwen,
            )

            results.append(
                {
                    "qid": qid,
                    "query": query,
                    "answer": answer,
                    "model": MODEL_NAME,
                    "num_docs_retrieved": len(docs),
                    "num_docs_used": min(len(docs), NUM_DOCS_FOR_ANSWER),
                }
            )

        except Exception as e:
            print(f"\nError processing {qid}: {e}")
            errors.append({"qid": qid, "error": str(e)})
            results.append(
                {
                    "qid": qid,
                    "query": query,
                    "answer": f"ERROR: {str(e)}",
                    "model": MODEL_NAME,
                    "error": True,
                }
            )

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "=" * 80)
    print("Completed batch RAG run")
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {len(results) - len(errors)}")
    print(f"Errors: {len(errors)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 80 + "\n")

    # Save error log if any
    if errors:
        error_file = OUTPUT_DIR / f"errors_{MODEL_NAME.split('/')[-1]}.jsonl"
        with open(error_file, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Error log saved to: {error_file}\n")

if __name__ == "__main__":
    main()