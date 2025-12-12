"""
Phase 2: Uncertainty Estimation

Computes two UE signals for a given RAG answers file and evaluates
their correlation with SAFE correctness scores:

- White-box UE: average LM NLL of the answer under Qwen (conditioned on the question)
- Black-box UE: 1 - MARS-style faithfulness score (entailment of answer vs. retrieved docs)

Outputs:
- A JSONL file with per-query UE scores
- A small JSON summary with Pearson/Spearman correlations vs. SAFE scores
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from utils import load_config, load_model_and_tokenizer, load_model_and_tokenizer2
from rag_query_local import retrieve_documents
from ue_utils import MarsScorer, compute_answer_nll, mars_uncertainty


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def index_by_qid(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["qid"]): r for r in records if "qid" in r}


def pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = np.sqrt((xv ** 2).sum()) * np.sqrt((yv ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((xv * yv).sum() / denom)


def spearman_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)

    # Simple rank transform (no tie-correction; fine for small N)
    xr = xv.argsort().argsort().astype(np.float64)
    yr = yv.argsort().argsort().astype(np.float64)

    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = np.sqrt((xr ** 2).sum()) * np.sqrt((yr ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((xr * yr).sum() / denom)


def discover_total_shards(cfg: Dict[str, Any]) -> int:
    output_root = Path(cfg["paths"]["output_root"])
    if not output_root.exists():
        raise FileNotFoundError(
            f"Output root directory not found at {output_root}. "
            "Ensure you have run the sharding/indexing scripts from the project root."
        )
    shard_dirs = [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("shard_")]
    return len(shard_dirs)


def compute_ue_for_answers(
    answers_path: Path,
    safe_path: Path,
    config_path: Path | None = None,
    output_scores_path: Path | None = None,
    output_summary_path: Path | None = None,
) -> Tuple[Path, Path]:
    cfg = load_config(str(config_path) if config_path is not None else None)

    # Resolve output_root to an absolute path based on the project root so that
    # this script works no matter what the current working directory is.
    project_root = Path(__file__).resolve().parents[1]
    paths_cfg = cfg.get("paths", {})
    rel_output_root = paths_cfg.get("output_root", "outputs")
    abs_output_root = project_root / rel_output_root
    paths_cfg["output_root"] = str(abs_output_root)
    cfg["paths"] = paths_cfg

    print(f"Loading answers from: {answers_path}")
    answers = load_jsonl(answers_path)
    ans_by_qid = index_by_qid(answers)
    print(f"Loaded {len(ans_by_qid)} answers")

    print(f"Loading SAFE scores from: {safe_path}")
    safe_records = load_jsonl(safe_path)
    safe_by_qid = index_by_qid(safe_records)
    print(f"Loaded {len(safe_by_qid)} SAFE records")

    # Intersect QIDs
    shared_qids = sorted(set(ans_by_qid.keys()) & set(safe_by_qid.keys()))
    print(f"Found {len(shared_qids)} shared qids between answers and SAFE")
    if not shared_qids:
        raise ValueError("No overlapping qids between answers and SAFE scores.")

    # Load models
    print("Loading retriever model (for re-retrieval used in black-box UE)...")
    retriever_tokenizer, retriever_model, retriever_device = load_model_and_tokenizer(
        cfg, task_type="retriever"
    )
    print("Retriever model loaded")

    print("Loading answering model (Qwen) for white-box UE (NLL)...")
    try:
        # Preferred path: local Qwen with INT4 on GPU (as in Phase 1)
        answering_tokenizer, answering_model, answering_device = load_model_and_tokenizer2(
            cfg, task_type="answering"
        )
        answering_model.eval()
        print("Answering model loaded (Qwen INT4 on GPU)\n")
        white_box_model_kind = "qwen_int4_gpu"
    except RuntimeError as e:
        # Fallback path: no GPU → use a small CPU LM just for NLL scoring
        print(f"Warning while loading Qwen answering model: {e}")
        print("Falling back to a small CPU LM 'gpt2' for white-box NLL UE...")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        answering_device = torch.device("cpu")
        answering_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        answering_model = AutoModelForCausalLM.from_pretrained("gpt2")
        answering_model.to(answering_device)
        answering_model.eval()
        print("Answering model loaded (gpt2 on CPU)\n")
        white_box_model_kind = "gpt2_cpu"

    # MARS on CPU by default (safer than competing with Qwen on GPU)
    print("Initializing MARS scorer for black-box UE (entailment)...")
    mars_scorer = MarsScorer(model_name="roberta-large-mnli", device=-1)
    print("MARS scorer ready\n")

    total_shards = discover_total_shards(cfg)
    if total_shards == 0:
        raise FileNotFoundError(
            "No shard directories found; Phase 1 retrieval indices must be built first."
        )
    print(f"Discovered {total_shards} shard directories for retrieval\n")

    ue_records: List[Dict[str, Any]] = []
    safe_scores: List[float] = []
    ue_white_list: List[float] = []
    ue_black_list: List[float] = []

    for qid in shared_qids:
        ans_rec = ans_by_qid[qid]
        safe_rec = safe_by_qid[qid]

        # Skip explicit errors if present
        if ans_rec.get("error"):
            continue

        query = ans_rec.get("query", "")
        answer = ans_rec.get("answer", "")
        safe_score = safe_rec.get("safe_score", None)
        if safe_score is None:
            continue

        # --- White-box UE: LM NLL of the answer under Qwen ---
        nll_info = compute_answer_nll(
            answer=answer,
            query=query,
            tokenizer=answering_tokenizer,
            model=answering_model,
            device=answering_device,
        )
        ue_white = nll_info["avg_nll"]

        # --- Black-box UE: 1 - MARS faithfulness ---
        # Re-retrieve docs for this query using sharded indices
        all_results = retrieve_documents(
            query,
            cfg,
            retriever_tokenizer,
            retriever_model,
            retriever_device,
            total_shards=total_shards,
            top_k_per_shard=2,
        )
        top_docs = all_results[:10]
        faith, ue_black = mars_uncertainty(answer, top_docs, mars_scorer, top_docs=len(top_docs))

        ue_record = {
            "qid": qid,
            "query": query,
            "answer": answer,
            "safe_score": float(safe_score),
            "ue_white_nll": float(ue_white) if ue_white is not None else None,
            "ue_white_num_tokens": int(nll_info.get("num_tokens", 0)),
            "mars_faithfulness": float(faith),
            "ue_black_mars_uncertainty": float(ue_black),
        }
        ue_records.append(ue_record)

        safe_scores.append(float(safe_score))
        ue_white_list.append(float(ue_white))
        ue_black_list.append(float(ue_black))

    if not ue_records:
        raise RuntimeError("No UE records computed; check that answers and SAFE scores are valid.")

    # Compute correlations
    pearson_white = pearson_corr(safe_scores, ue_white_list)
    spearman_white = spearman_corr(safe_scores, ue_white_list)
    pearson_black = pearson_corr(safe_scores, ue_black_list)
    spearman_black = spearman_corr(safe_scores, ue_black_list)

    print("\nCorrelations vs. SAFE score (higher |corr| is better):")
    print(f"- White-box UE (LM NLL):     Pearson = {pearson_white:.4f}, Spearman = {spearman_white:.4f}")
    print(f"- Black-box UE (1 - MARS):   Pearson = {pearson_black:.4f}, Spearman = {spearman_black:.4f}\n")

    # Paths
    if output_scores_path is None:
        output_scores_path = answers_path.parent / f"ue_scores_{answers_path.name}"
    if output_summary_path is None:
        output_summary_path = answers_path.parent / "ue_correlation_summary.json"

    # Save per-query scores
    with output_scores_path.open("w", encoding="utf-8") as f:
        for r in ue_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"UE scores written to: {output_scores_path}")

    # Save summary
    summary = {
        "num_examples": len(ue_records),
        "pearson_white_nll": pearson_white,
        "spearman_white_nll": spearman_white,
        "pearson_black_mars_uncertainty": pearson_black,
        "spearman_black_mars_uncertainty": spearman_black,
        "answers_path": str(answers_path),
        "safe_path": str(safe_path),
        "white_box_model_kind": locals().get("white_box_model_kind", "unknown"),
    }
    with output_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"UE correlation summary written to: {output_summary_path}\n")

    return output_scores_path, output_summary_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Uncertainty Estimation for RAG answers")
    parser.add_argument(
        "--answers_path",
        type=str,
        required=True,
        help="Path to RAG answers JSONL (with qid/query/answer fields).",
    )
    parser.add_argument(
        "--safe_path",
        type=str,
        required=True,
        help="Path to SAFE scores JSONL (with qid/safe_score fields).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config YAML (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--scores_out",
        type=str,
        default=None,
        help="Optional output JSONL path for per-query UE scores.",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default=None,
        help="Optional output JSON path for correlation summary.",
    )

    args = parser.parse_args()

    answers_path = Path(args.answers_path)
    safe_path = Path(args.safe_path)
    config_path = Path(args.config) if args.config is not None else None
    scores_out = Path(args.scores_out) if args.scores_out is not None else None
    summary_out = Path(args.summary_out) if args.summary_out is not None else None

    compute_ue_for_answers(
        answers_path=answers_path,
        safe_path=safe_path,
        config_path=config_path,
        output_scores_path=scores_out,
        output_summary_path=summary_out,
    )


