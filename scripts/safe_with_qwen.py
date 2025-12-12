import os
import json
from pathlib import Path

from tqdm import tqdm

import TruthTorchLM.long_form_generation as LFG

from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def extract_claims(decomposition_output):
    """Convert decomposition output to a list of claim strings."""
    if isinstance(decomposition_output, dict):
        raw_claims = decomposition_output.get("claims", [])
    elif isinstance(decomposition_output, list):
        raw_claims = decomposition_output
    else:
        raw_claims = []

    claims = []
    for c in raw_claims:
        if isinstance(c, str):
            text = c
        elif isinstance(c, dict):
            text = c.get("text") or c.get("claim") or ""
        else:
            text = str(c)
        text = text.strip()
        if text:
            claims.append(text)
    return claims


def main():
    """
    Batch SAFE scoring for RAG answers using Google Search API (via Serper).

    Input:
      - outputs/rag_answers_api_qwen-2.5-72b-instruct.jsonl

    Output:
      - outputs/safe_labels_api_qwen-2.5-72b-instruct.jsonl
        Each line: {
          "qid": ...,
          "num_claims": int,
          "supported": int,
          "not_supported": int,
          "total_used": int,
          "safe_score": float | null
        }
    """
    root = Path(__file__).resolve().parents[1]
    input_path = root / "outputs" / "rag_answers_api_qwen-2.5-72b-instruct.jsonl"
    output_path = root / "outputs" / "safe_labels_api_qwen-2.5-72b-instruct.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. SAFE evaluator expects OpenRouter access for Qwen."
        )
    if not SERPER_API_KEY:
        raise RuntimeError(
            "SERPER_API_KEY is not set. Google Search (Serper) is required for SAFE evidence."
        )

    # Load all records
    with input_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} RAG answers from {input_path}")
    print("Using Google Search via Serper for SAFE evidence\n")

    # Decomposer and SAFE evaluator (Qwen via OpenRouter)
    decomposition_method = LFG.decomposition_methods.StructuredDecompositionAPI(
        model="openrouter/qwen/qwen-2.5-72b-instruct",
        decomposition_depth=1,
    )
    safe_evaluator = LFG.ClaimEvaluator(
        rater="openrouter/qwen/qwen-2.5-72b-instruct",
        tokenizer=None,
        max_steps=5,
        max_retries=10,
        num_searches=3,
    )

    output_path.parent.mkdir(exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for rec in tqdm(records, desc="SAFE scoring answers"):
            qid = rec.get("qid")
            answer = rec.get("answer", "") or ""

            if not answer.strip():
                result = {
                    "qid": qid,
                    "num_claims": 0,
                    "supported": 0,
                    "not_supported": 0,
                    "total_used": 0,
                    "safe_score": None,
                }
                out_f.write(json.dumps(result) + "\n")
                continue

            try:
                decomposition_output = decomposition_method(answer)
                claims = extract_claims(decomposition_output)
            except Exception as e:
                print(f"\nDecomposition error for qid={qid}: {e}")
                result = {
                    "qid": qid,
                    "num_claims": 0,
                    "supported": 0,
                    "not_supported": 0,
                    "total_used": 0,
                    "safe_score": None,
                }
                out_f.write(json.dumps(result) + "\n")
                continue

            supported = 0
            not_supported = 0

            for claim in claims:
                try:
                    resp = safe_evaluator(claim)
                    label = (resp or {}).get("answer")
                except Exception as e:
                    print(f"\nSAFE eval error for qid={qid} claim='{claim[:50]}...': {e}")
                    continue

                if label == "Supported":
                    supported += 1
                elif label == "Not Supported":
                    not_supported += 1

            total_used = supported + not_supported
            safe_score = None
            if total_used > 0:
                safe_score = supported / total_used

            result = {
                "qid": qid,
                "num_claims": len(claims),
                "supported": supported,
                "not_supported": not_supported,
                "total_used": total_used,
                "safe_score": safe_score,
            }
            out_f.write(json.dumps(result) + "\n")

    print(f"\nSAFE scores written to {output_path}")

if __name__ == "__main__":
    main()