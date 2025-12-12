"""
Merge SAFE labels with RAG answers (with context/UE) by qid.

Inputs:
- outputs/rag_answers_with_context.jsonl
- outputs/safe_labels.jsonl  (expected fields: qid, safe_score or safe_label)

Output:
- outputs/rag_answers_with_safe.jsonl
"""

import json
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    root = Path(__file__).resolve().parents[1]
    rag_path = root / "outputs" / "rag_answers_with_context.jsonl"
    safe_path = root / "outputs" / "safe_labels.jsonl"
    out_path = root / "outputs" / "rag_answers_with_safe.jsonl"

    rag_records = load_jsonl(rag_path)
    safe_records = load_jsonl(safe_path)

    safe_by_qid = {r["qid"]: r for r in safe_records if "qid" in r}

    merged = []
    for r in rag_records:
        qid = r.get("qid")
        safe = safe_by_qid.get(qid, {})
        merged.append({**r, "safe": safe})

    with open(out_path, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    print(f"Merged {len(merged)} records into {out_path}")


if __name__ == "__main__":
    main()

