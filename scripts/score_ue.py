#cat > scripts/score_ue.py << 'PY'
import json
from truthtorchlm import TruthTorchLM
from truthtorchlm.methods import mars, eccentricity

rows = [json.loads(l) for l in open("runs/pilot_rag_qwen.jsonl")]

tt = TruthTorchLM()

scored = []
for r in rows:
    mars_score = mars.score(text=r["answer"], token_scores=r.get("scores"))
    ecc_score  = eccentricity.score(text=r["answer"])
    scored.append({"question": r["question"], "mars": float(mars_score), "eccentricity": float(ecc_score)})

with open("runs/pilot_ue_scores.jsonl", "w") as f:
    for x in scored: f.write(json.dumps(x)+"\n")
print("Saved -> runs/pilot_ue_scores.jsonl")
