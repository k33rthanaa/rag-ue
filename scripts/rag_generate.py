#cat > scripts/rag_generate.py << 'PY'
import faiss, json, pyarrow.parquet as pq
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from helpers.device import pick_device, default_dtype_for

INDEX_DIR = "index/contriever"
LLM_DIR   = "models/qwen2_5_7b_instruct"
RET_DIR   = "models/contriever"

device = pick_device()
dtype  = default_dtype_for(device)

# LLM
tok_llm = AutoTokenizer.from_pretrained(LLM_DIR)
llm = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype=dtype).to(device).eval()

# Retriever encoder
ret_tok = AutoTokenizer.from_pretrained(RET_DIR)
ret_enc = AutoModel.from_pretrained(RET_DIR, torch_dtype=dtype).to(device).eval()

@torch.no_grad()
def embed_query(q: str):
    t = ret_tok(q, return_tensors="pt", truncation=True).to(device)
    last = ret_enc(**t).last_hidden_state
    emb  = last.mean(dim=1)
    emb  = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.to("cpu", dtype=torch.float32).numpy()

def retrieve(q, k=5):
    index = faiss.read_index(f"{INDEX_DIR}/contriever.faiss")
    meta  = pq.read_table(f"{INDEX_DIR}/meta.parquet").to_pandas()
    v = embed_query(q)
    D, I = index.search(v, k)
    ctx = "\n\n".join([meta.iloc[i]["passage"] for i in I[0]])
    return ctx

@torch.no_grad()
def generate_answer(q):
    ctx = retrieve(q, k=5)
    prompt = ("You are a careful biographer. Use ONLY the context to write a factual paragraph.\n\n"
              f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")
    inputs = tok_llm(prompt, return_tensors="pt").to(device)
    out = llm.generate(**inputs, max_new_tokens=256, do_sample=False,
                       output_scores=True, return_dict_in_generate=True, use_cache=True)
    text = tok_llm.decode(out.sequences[0], skip_special_tokens=True)
    scores = [s.detach().to("cpu").float().numpy().tolist() for s in out.scores]
    return {"question": q, "answer": text, "scores": scores, "context": ctx}

pilot = [
    "Write a short biography of Ada Lovelace.",
    "Write a short biography of Nikola Tesla.",
    "Write a short biography of Marie Curie.",
]

rows = [generate_answer(q) for q in pilot]
with open("runs/pilot_rag_qwen.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("Saved -> runs/pilot_rag_qwen.jsonl  |  Device:", device, " Dtype:", dtype)

