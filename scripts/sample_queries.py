import json
import random

# Read all queries
with open('data/factscore_bio.jsonl', 'r') as f:
    queries = [json.loads(line) for line in f]

# Random sample of 50
random.seed(42)
sample = random.sample(queries, 50)

# Save
with open('data/factscore_bio_50.jsonl', 'w') as f:
    for q in sample:
        f.write(json.dumps(q) + '\n')

print(f"✅ Saved 50 random queries to data/factscore_bio_50.jsonl")
