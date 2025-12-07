import json
from pathlib import Path
from tqdm import tqdm
# Import your RAG functions
from rag_query1 import retrieve_documents_merged, generate_answer
from utils import load_config, load_model_and_tokenizer

# Load 50 queries
queries = []
with open('data/factscore_bio_50.jsonl', 'r') as f:
    queries = [json.loads(line) for line in f]

# Load models once
cfg = load_config('configs/default.yaml')
retriever_tokenizer, retriever_model, device_cpu = load_model_and_tokenizer(cfg, "retriever")
answering_tokenizer, answering_model, device_gpu = load_model_and_tokenizer(cfg, "answering")

# Generate answers
results = []
for query_obj in tqdm(queries):
    query = query_obj['prompt']
    
    # Retrieve
    docs = retrieve_documents_merged(query, cfg, retriever_tokenizer, 
                                     retriever_model, device_cpu, top_k=15)
    
    # Generate
    answer = generate_answer(query, docs[:5], answering_tokenizer, 
                            answering_model, device_gpu)
    
    results.append({
        'qid': query_obj['qid'],
        'query': query,
        'answer': answer
    })

# Save
with open('outputs/rag_answers.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
