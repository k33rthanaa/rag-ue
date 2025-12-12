"""
Utilities for Uncertainty Estimation (UE) and faithfulness scoring.

Provides:
- token entropy aggregation helpers
- a lightweight MARS-style faithfulness scorer using an NLI model
"""

from typing import List, Dict, Any, Tuple

import torch
from transformers import pipeline


def compute_mean_entropy_from_scores(scores) -> Dict[str, Any]:
    """
    Given generate() scores (list of logits tensors), return mean entropy and per-token entropies.
    """
    import torch.nn.functional as F

    if not scores:
        return {"mean_token_entropy": None, "token_entropies": []}

    entropies = []
    for step_logits in scores:
        probs = F.softmax(step_logits, dim=-1)
        entropy = (-probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
        entropies.append(entropy.mean().item())

    mean_entropy = float(sum(entropies) / len(entropies)) if entropies else None
    return {"mean_token_entropy": mean_entropy, "token_entropies": entropies}


class MarsScorer:
    """
    A lightweight MARS-style scorer using an entailment model.
    For each answer sentence, we compute entailment against each retrieved doc
    and aggregate the max entailment probability across docs, then average over sentences.
    """

    def __init__(self, model_name: str = "roberta-large-mnli", device: int = 0):
        # device: -1 for CPU; int GPU index otherwise
        self.device = device
        self.clf = pipeline("text-classification", model=model_name, device=device)

    def _split_sentences(self, text: str) -> List[str]:
        parts = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return parts if parts else [text.strip()]

    def score(self, answer: str, docs: List[Dict[str, Any]], top_docs: int = 5) -> float:
        if not answer or not docs:
            return 0.0
        sentences = self._split_sentences(answer)
        entailment_scores = []
        for sent in sentences:
            sent_scores = []
            for doc in docs[:top_docs]:
                premise = doc.get("text", "") or ""
                if not premise:
                    continue
                out = self.clf({"text": premise, "text_pair": sent}, truncation=True)

                # HuggingFace pipelines may return either:
                # - a single dict: {"label": "...", "score": ...}
                # - a list of dicts: [{"label": "...", "score": ...}, ...]
                if isinstance(out, dict):
                    entries = [out]
                else:
                    entries = out

                # Expect labels like 'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'
                entail_prob = 0.0
                for entry in entries:
                    label = entry["label"].upper()
                    if "ENTAIL" in label:
                        entail_prob = max(entail_prob, float(entry["score"]))
                sent_scores.append(entail_prob)
            entailment_scores.append(max(sent_scores) if sent_scores else 0.0)

        if not entailment_scores:
            return 0.0
        return float(sum(entailment_scores) / len(entailment_scores))


@torch.no_grad()
def compute_answer_nll(
    answer: str,
    query: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    White-box UE: average token negative log-likelihood of the answer under the LM.

    We condition on a simple "Question ... Answer ..." prefix so that the model
    scores the answer *given* the question, and mask out the question tokens
    when computing the loss.
    """
    if not answer:
        return {"avg_nll": None, "num_tokens": 0}

    prompt = f"Question: {query}\n\nAnswer:\n"

    # Token IDs for the prompt only (used to find the split point)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # Full sequence: prompt + answer
    full_text = prompt + answer
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"]

    # Number of prompt tokens actually present in the truncated sequence
    prompt_len = min(len(prompt_ids), input_ids.shape[1])

    # Mask out prompt tokens so loss is computed only on answer tokens
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # ignore index for CrossEntropyLoss

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss  # mean NLL over non-masked tokens

    num_tokens = int((labels != -100).sum().item())
    return {
        "avg_nll": float(loss.item()),
        "num_tokens": num_tokens,
    }


def mars_uncertainty(
    answer: str,
    docs: List[Dict[str, Any]],
    mars_scorer: MarsScorer,
    top_docs: int = 5,
) -> Tuple[float, float]:
    """
    Black-box UE: convert MARS-style faithfulness into an uncertainty score.

    Returns (faithfulness_score, uncertainty), where:
    - faithfulness_score in [0, 1]: higher is more faithful / supported
    - uncertainty = 1 - faithfulness_score
    """
    faith = mars_scorer.score(answer, docs, top_docs=top_docs)
    # Clamp just in case the underlying model returns slightly out-of-range scores
    faith = max(0.0, min(1.0, float(faith)))
    return faith, 1.0 - faith


