# rag_query.py
import faiss
import json
import torch
import argparse
import gzip
from pathlib import Path
from transformers import AutoModelForCausalLM
from utils import load_config, load_model_and_tokenizer, encode_batch, get_device, setup_logging
from tabulate import tabulate

def load_metadata(meta_path):
    """Load metadata from JSONL.GZ file"""
    records = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def retrieve_documents_merged(query, cfg, retriever_tokenizer, retriever_model, device, top_k=10):
    """Retrieve documents from merged index (FAST!)"""
    logger = setup_logging(str(cfg.get("paths", {}).get("logs_dir", "outputs/logs")), "INFO")
    output_root = Path(cfg["paths"]["output_root"])
    
    # Paths to merged files
    merged_index_path = output_root / "merged" / "merged_index.index"
    merged_meta_path = output_root / "merged" / "merged_metadata.jsonl.gz"
    
    if not merged_index_path.exists() or not merged_meta_path.exists():
        raise FileNotFoundError(f"Merged index not found! Please run merge_indices.py first.")
    
    print(f"📂 Loading merged index from: {merged_index_path}")
    
    # Load merged index (one-time load)
    index = faiss.read_index(str(merged_index_path))
    print(f"✅ Loaded merged index with {index.ntotal:,} vectors")
    
    # Load merged metadata
    print(f"📂 Loading merged metadata from: {merged_meta_path}")
    metadata = load_metadata(merged_meta_path)
    print(f"✅ Loaded {len(metadata):,} metadata records\n")
    
    # Encode query
    q_emb = encode_batch([query], retriever_tokenizer, retriever_model, device, max_length=512)
    faiss.normalize_L2(q_emb)
    
    # Single FAISS search (instead of 11!)
    print(f"🔍 Searching for top {top_k} documents...")
    D, I = index.search(q_emb, top_k)
    
    # Build results
    results = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx >= len(metadata):
            continue
        
        meta = metadata[idx]
        title = meta.get("title", "")
        text = meta.get("contents", "")
        doc_id = meta.get("dataset_id", meta.get("id", meta.get("global_id")))
        
        results.append({
            "rank": rank,
            "score": float(score),
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "snippet": text[:200] + ("…" if len(text) > 200 else "")
        })
    
    return results

def query_shard(shard_id, query, index, metadata, tokenizer, model, device, max_length=512, top_k=2):
    """Query a single shard and retrieve the top K results"""
    # Encode the query
    q_emb = encode_batch([query], tokenizer, model, device, max_length=max_length)
    faiss.normalize_L2(q_emb)

    # Perform FAISS search
    D, I = index.search(q_emb, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx >= len(metadata):
            continue
        meta = metadata[idx]
        title = meta.get("title", "")
        text = meta.get("contents", "")
        doc_id = meta.get("dataset_id", meta.get("id", meta.get("global_id")))

        results.append({
            "rank": rank,
            "score": float(score),
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "snippet": text[:200] + ("…" if len(text) > 200 else "")
        })
    return results

def retrieve_documents(query, cfg, retriever_tokenizer, retriever_model, device, total_shards, top_k_per_shard=2):
    """Retrieve documents from all shards and rank them (OLD METHOD - SLOW)"""
    logger = setup_logging(str(cfg.get("paths", {}).get("logs_dir", "outputs/logs")), "INFO")
    output_root = Path(cfg["paths"]["output_root"])
    all_results = []

    # Query all shards
    for shard_id in range(total_shards):
        shard_dir = output_root / f"shard_{shard_id:04d}"
        index_path = shard_dir / f"shard_{shard_id:04d}.index"
        meta_path = shard_dir / f"shard_{shard_id:04d}.meta.jsonl.gz"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(f"Skipping shard {shard_id} (missing files)")
            continue

        index = faiss.read_index(str(index_path))
        metadata = load_metadata(meta_path)

        results = query_shard(
            shard_id, query, index, metadata,
            retriever_tokenizer, retriever_model, device,
            top_k=top_k_per_shard
        )
        all_results.extend(results)

    # Sort by relevance score (descending)
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results

def generate_answer(query, top_docs, answering_tokenizer, answering_model, device, max_context_length=2048):
    """Generate answer using top retrieved documents"""

    # Build context from top documents
    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        context_parts.append(f"Document {i}: {doc['title']}\n{doc['text'][:500]}")

    context = "\n\n".join(context_parts)

    # Create prompt
    prompt = f"""Based on the following documents, answer the question.

Question: {query}

Documents:
{context}

Answer:"""

    # Tokenize with proper settings
    inputs = answering_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_context_length,
        truncation=True,
        padding=True,
        return_attention_mask=True
    ).to(device)

    # Generate answer
    outputs = answering_model.generate(
        **inputs,
        max_new_tokens=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=answering_tokenizer.pad_token_id,
        eos_token_id=answering_tokenizer.eos_token_id,
        temperature=0.7,
        do_sample=True
    )

    # Decode answer
    full_response = answering_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer part (after "Answer:")
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
    else:
        answer = full_response

    return answer

def main(args):
    # Load config
    cfg = load_config(args.config)
    logger = setup_logging(str(cfg.get("paths", {}).get("logs_dir", "outputs/logs")), "INFO")

    print("\n" + "="*80)
    print("🔍 RAG System - Question Answering")
    print("="*80 + "\n")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Step 1: Load retriever model
    print("📚 Loading retriever model...")
    retriever_tokenizer, retriever_model, device = load_model_and_tokenizer(cfg, task_type="retriever")
    logger.info("✅ Retriever model loaded")

    # Step 2: Retrieve documents
    print(f"\n🔎 Searching for relevant documents...")
    print(f"Query: {args.query}\n")

    # Choose retrieval method
    if args.use_merged:
        print("⚡ Using MERGED index (fast!)\n")
        all_results = retrieve_documents_merged(
            args.query, cfg, retriever_tokenizer, retriever_model,
            device, top_k=args.top_k
        )
    else:
        print("🐌 Using SHARDED indices (slower)\n")
        all_results = retrieve_documents(
            args.query, cfg, retriever_tokenizer, retriever_model,
            device, args.total_shards, top_k_per_shard=args.top_k_per_shard
        )

    if not all_results:
        print("❌ No documents found!")
        return

    # Take top N for answer generation
    top_docs = all_results[:args.top_n_for_answer]

    print(f"✅ Found {len(all_results)} documents")
    print(f"📄 Using top {len(top_docs)} documents for answer generation\n")

    # Display retrieved documents
    print("📋 Top Retrieved Documents:")
    print("-" * 80)
    table_data = [
        [i+1, f"{doc['score']:.4f}", doc['doc_id'], doc['title'][:50] + "..." if len(doc['title']) > 50 else doc['title']]
        for i, doc in enumerate(top_docs)
    ]
    print(tabulate(table_data, headers=["Rank", "Score", "Doc ID", "Title"], tablefmt="grid"))

    # Step 3: Load answering model
    print(f"\n🤖 Loading answering model ({cfg.get('answering_model_name', 'Qwen/Qwen2.5-7B-Instruct')})...")
    answering_tokenizer, answering_model, device = load_model_and_tokenizer(cfg, task_type="answering")

    # Set pad token properly
    if answering_tokenizer.pad_token is None:
        answering_tokenizer.pad_token = answering_tokenizer.eos_token
        answering_tokenizer.pad_token_id = answering_tokenizer.eos_token_id
        answering_model.config.pad_token_id = answering_tokenizer.pad_token_id

    logger.info("✅ Answering model loaded")

    # Step 4: Generate answer
    print("\n💭 Generating answer...\n")
    answer = generate_answer(args.query, top_docs, answering_tokenizer, answering_model, device)

    # Display answer
    print("="*80)
    print("📝 ANSWER:")
    print("="*80)
    print(answer)
    print("="*80 + "\n")

    # Save results if needed
    if args.save_output:
        output_file = Path("outputs") / f"rag_answer_{args.query[:30].replace(' ', '_')}.txt"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {args.query}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("Top Documents:\n")
            for i, doc in enumerate(top_docs, 1):
                f.write(f"\n{i}. {doc['title']} (Score: {doc['score']:.4f})\n")
                f.write(f"   {doc['snippet']}\n")
        print(f"💾 Results saved to: {output_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System: Retrieve documents and generate answers")
    parser.add_argument("--query", type=str, required=True, help="The question to answer")
    parser.add_argument("--use_merged", action="store_true", help="Use merged index (FAST!)")
    parser.add_argument("--top_k", type=int, default=10, help="Top K docs from merged index")
    parser.add_argument("--total_shards", type=int, default=11, help="Total number of shards (for non-merged)")
    parser.add_argument("--top_k_per_shard", type=int, default=2, help="Docs per shard (for non-merged)")
    parser.add_argument("--top_n_for_answer", type=int, default=5, help="Number of top docs for answer generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--save_output", action="store_true", help="Save results to file")

    args = parser.parse_args()
    main(args)
