"""
RAG Query with Google AI Studio (OpenAI-compatible) API
Single query version using API for answer generation.

This uses Google's OpenAI-compatible `/chat/completions` endpoint.
You must set `GOOGLE_API_KEY` in your environment.
"""
import faiss
import json
import gzip
import os
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv
from utils import load_config, load_model_and_tokenizer, encode_batch, get_device, setup_logging
from tabulate import tabulate

load_dotenv()

def load_metadata(meta_path):
    """Load metadata from JSONL.GZ file"""
    records = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def generate_answer_api(query, top_docs, api_key, model_name="gpt-4.1-mini", max_tokens=300):
    """
    Generate biographical answer using Google AI Studio OpenAI-compatible API.
    
    Args:
        query: The question to answer
        top_docs: List of retrieved documents
        api_key: Google AI Studio API key (GOOGLE_API_KEY)
        model_name: Model identifier (e.g., 'gpt-4.1-mini')
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated answer string
    """
    
    # Build context from top documents
    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        context_parts.append(f"Document {i}: {doc['title']}\n{doc['text'][:500]}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a factual and precise assistant that writes grounded biographical summaries.

Question: {query}

Reference Documents:
{context}

Task:
Write a comprehensive biographical summary (150–200 words) using ONLY information
explicitly found in the Reference Documents.

If any required detail is missing (such as birth date, nationality, achievements,
career history, key life events), simply omit that detail rather than guessing.

Requirements:
- Full name (if present in the documents)
- Nationality and profession (only if present)
- Major achievements and career highlights (from documents)
- Notable works or contributions (from documents)
- Key life events (from documents)
- Write 1–2 coherent paragraphs in complete, objective sentences.
- DO NOT add external knowledge, speculation, or assumptions.

Important:
You MUST NOT add facts that do not appear in the Reference Documents.
Each sentence must be verifiable from the documents.

Biography:"""

    # Google AI Studio OpenAI-compatible API call
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional metadata headers; can be customized
        "HTTP-Referer": "https://github.com/your-username/rag-project",
        "X-Title": "RAG Biographical QA",
    }
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip()
        
        return answer
        
    except requests.exceptions.RequestException as e:
        print(f"Google API error: {e}")
        return f"Error generating answer: {str(e)}"
    except (KeyError, IndexError) as e:
        print(f"Response parsing error: {e}")
        return "Error parsing API response"

def retrieve_documents_merged(query, cfg, retriever_tokenizer, retriever_model, device, top_k=10):
    """Retrieve documents from merged index"""
    output_root = Path(cfg["paths"]["output_root"])
    merged_index_path = output_root / "merged" / "merged_index.index"
    merged_meta_path = output_root / "merged" / "merged_metadata.jsonl.gz"
    
    if not merged_index_path.exists() or not merged_meta_path.exists():
        raise FileNotFoundError(f"Merged index not found! Please run merge_indices.py first.")
    
    print(f"Loading merged index from: {merged_index_path}")
    index = faiss.read_index(str(merged_index_path))
    print(f"Loaded merged index with {index.ntotal:,} vectors")
    
    print(f"Loading merged metadata from: {merged_meta_path}")
    metadata = load_metadata(merged_meta_path)
    print(f"Loaded {len(metadata):,} metadata records\n")
    
    # Encode query
    q_emb = encode_batch([query], retriever_tokenizer, retriever_model, device, max_length=512)
    faiss.normalize_L2(q_emb)
    
    # Search
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

def main(args):
    cfg = load_config(args.config)
    
    print("\n" + "=" * 80)
    print("RAG System - Question Answering (Google AI Studio API)")
    print("=" * 80 + "\n")
    
    # Load retriever model
    print("Loading retriever model...")
    device = get_device()
    retriever_tokenizer, retriever_model, device = load_model_and_tokenizer(cfg, task_type="retriever")
    print("Retriever model loaded")
    
    # Retrieve documents
    print(f"\nSearching for relevant documents...")
    print(f"Query: {args.query}\n")
    
    all_results = retrieve_documents_merged(
        args.query, cfg, retriever_tokenizer, retriever_model,
        device, top_k=args.top_k
    )
    
    if not all_results:
        print("No documents found.")
        return
    
    top_docs = all_results[:args.top_n_for_answer]
    
    print(f"Found {len(all_results)} documents")
    print(f"Using top {len(top_docs)} documents for answer generation\n")
    
    # Display retrieved documents
    print("Top retrieved documents:")
    print("-" * 80)
    table_data = [
        [i+1, f"{doc['score']:.4f}", doc['doc_id'], doc['title'][:50] + "..." if len(doc['title']) > 50 else doc['title']]
        for i, doc in enumerate(top_docs)
    ]
    print(tabulate(table_data, headers=["Rank", "Score", "Doc ID", "Title"], tablefmt="grid"))
    
    # Generate answer using API
    print(f"\nGenerating answer using OpenRouter API...")
    print(f"Model: {args.model}\n")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment.")
        return
    
    answer = generate_answer_api(args.query, top_docs, api_key, args.model)
    
    # Display answer
    print("="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System with OpenRouter API")
    parser.add_argument("--query", type=str, required=True, help="The question to answer")
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-72b-instruct", help="OpenRouter model name")
    parser.add_argument("--top_k", type=int, default=10, help="Top K docs to retrieve")
    parser.add_argument("--top_n_for_answer", type=int, default=5, help="Number of docs for answer generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    
    args = parser.parse_args()
    main(args)
