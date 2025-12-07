import json
from pathlib import Path
from tqdm import tqdm
import faiss
import gzip
from rag_query1 import generate_answer, load_metadata
from utils import load_config, load_model_and_tokenizer, encode_batch

# Load 50 queries
queries = []
with open('data/factscore_bio_50.jsonl', 'r') as f:
    queries = [json.loads(line) for line in f]

# Load models once
cfg = load_config('configs/default.yaml')
retriever_tokenizer, retriever_model, device_cpu = load_model_and_tokenizer(cfg, "retriever")
answering_tokenizer, answering_model, device_answering = load_model_and_tokenizer(cfg, "answering")

print("✅ Models loaded")

# Load merged index ONCE!
print("\n📂 Loading merged index (one-time, ~15 minutes)...")
output_root = Path(cfg["paths"]["output_root"])
merged_index_path = output_root / "merged" / "merged_index.index"
merged_meta_path = output_root / "merged" / "merged_metadata.jsonl.gz"

index = faiss.read_index(str(merged_index_path))
metadata = load_metadata(merged_meta_path)
print(f"✅ Loaded merged index with {index.ntotal:,} vectors\n")

# Generate answers
results = []
for query_obj in tqdm(queries, desc="Processing queries"):
    query = query_obj['prompt']
    
    # Retrieve using ALREADY LOADED index
    q_emb = encode_batch([query], retriever_tokenizer, retriever_model, device_cpu, max_length=512)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, 15)
    
    # Build results
    docs = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < len(metadata):
            meta = metadata[idx]
            docs.append({
                "rank": rank,
                "score": float(score),
                "doc_id": meta.get("dataset_id", meta.get("id", meta.get("global_id"))),
                "title": meta.get("title", ""),
                "text": meta.get("contents", "")
            })
    
    # Generate
    answer = generate_answer(query, docs[:5], answering_tokenizer, 
                            answering_model, device_answering)
    
    results.append({
        'qid': query_obj['qid'],
        'query': query,
        'answer': answer
    })

# Save
with open('outputs/rag_answers.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

print(f"\n✅ Done! Saved {len(results)} answers to outputs/rag_answers.jsonl")
