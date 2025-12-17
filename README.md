## RAG-Uncertainty-Estimator

End-to-end pipeline for Retrieval-Augmented Generation (RAG) with Uncertainty Estimation (UE).  
The system uses **Contriever** for retrieval, **Qwen2.5-7B-Instruct** for generation, and **SAFE + MARS-style UE** to study how uncertainty correlates with factual correctness.

---

## What this repo gives you

- **Large-scale retrieval**: build sharded FAISS indices over corpora like `wiki-18`, then merge them into a single **merged index** for fast downstream retrieval.
- **RAG answering**: generate answers with Qwen (local) or API-based models.
- **SAFE scoring**: label answers with factual correctness scores.
- **Uncertainty Estimation**:
  - **White-box UE**: average LM NLL of the answer under a language model.
  - **Black-box UE**: \(1 -\) MARS-style faithfulness (entailment vs retrieved docs).
- **Correlation analysis**: compute Pearson/Spearman correlations between SAFE and UE.

---

## Requirements

- **Python**: 3.10 recommended (3.8+ should work).
- **GPU (recommended)**:
  - CUDA-capable GPU for Qwen2.5-7B-Instruct (INT4) and optionally MARS (RoBERTa MNLI).
  - CPU-only is supported but much slower for generation and UE.
- **Disk**:
  - Dataset (e.g., `wiki-18.jsonl.gz`).
  - Sharded FAISS indices under `outputs/` and a merged index under `outputs/merged/`.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU FAISS:

```bash
pip install faiss-gpu
```

---

## Project structure

```text
RAG-Uncertainty-Estimator/
  configs/
    default.yaml           # Main config (models, dataset, paths, sharding)

  data/
    wiki-18.jsonl.gz       # Downloaded corpus (git-ignored)

  scripts/
    download_dataset.py    # Download HF dataset to data/
    build_shard.py         # Build a single FAISS shard
    query_index.py         # Interactive shard queries

    utils.py               # Shared config/model/encoding utilities
    ue_utils.py            # UE + MARS helpers

    batch_rag.py           # Batch RAG answering over many queries
    rag_query.py           # Simple local RAG query
    rag_query_local.py     # RAG query used by UE pipeline
    rag_query_api.py       # RAG via external API (e.g., Qwen API)

    safe_with_openai.py    # SAFE scoring with OpenAI-style API
    safe_with_qwen.py      # SAFE scoring with Qwen
    merge_safe_labels.py   # Merge SAFE labels/checkpoints if needed

    eval_ue_phase2.py      # Phase 2 UE + correlation vs SAFE

  outputs/
    shard_0000/            # FAISS index + metadata per shard
    shard_0001/
    ...
    merged/                # Merged FAISS index + merged metadata (recommended for all downstream tasks)
      merged_index.index
      merged_metadata.jsonl.gz
    rag_answers_*.jsonl    # RAG answers (with qid/query/answer)
    safe_scores_*.jsonl    # SAFE scores (with qid/safe_score)
    ue_scores_*.jsonl      # Per-query UE outputs
    ue_correlation_summary.json  # Global correlation summary
```

---

## Key configuration (`configs/default.yaml`)

### Models

- **Retriever**:
  - **`model_name`**: path or HF id, e.g. `"scripts/models/contriever"` or `"facebook/contriever"`.
- **Generator / Answering model**:
  - **`answering_model_name`**: path or HF id for Qwen, e.g. `"scripts/models/qwen2.5-7B-instruct"`.

### Encoding & indexing

- **`batch_size`**: encoding batch size (reduce if OOM).
- **`max_length`**: max tokens per document.
- **`use_fp16`**: `true` to use fp16 on GPU.
- **`sharding.shard_size`**: docs per shard.
- **`index.normalize`**: `true` for L2-normalized embeddings (`IndexFlatIP` ≈ cosine).

### Paths

- **`paths.output_root`**: root for all outputs (indices, logs, scores).
- **`paths.logs_dir`**: logging directory.
- **`paths.hf_cache`**: HF cache dir (can point into `outputs/hf_cache`).

---

## End-to-end workflow: RAG + SAFE + UE

### 1. Download dataset

```bash
python scripts/download_dataset.py \
  --config configs/default.yaml
```

This writes the corpus to the `data/` folder defined in the config (e.g. `data/wiki-18.jsonl.gz`).

---

### 2. Build sharded FAISS indices

Build one shard (adjust `--shard_id` as needed):

```bash
python scripts/build_shard.py --shard_id 0 --config configs/default.yaml
```

Repeat for all shards you need:

```bash
python scripts/build_shard.py --shard_id 1
python scripts/build_shard.py --shard_id 2
# ...
```

Each shard folder under `outputs/` contains:
- `shard_XXXX.index` – FAISS `IndexFlatIP` over Contriever embeddings.
- `shard_XXXX.meta.jsonl.gz` – metadata (title, contents, ids) for RAG.

You can sanity-check retrieval with:

```bash
python scripts/query_index.py --shard_id 0 --top_k 5
```

---

### 3. Merge shards into a single merged index (recommended)

Once you have built all shard folders, merge them into a single index + metadata file:

```bash
python scripts/merge_indices.py \
  --output_root outputs \
  --total_shards 11 \
  --merged_dir outputs/merged
```

This produces:
- `outputs/merged/merged_index.index`
- `outputs/merged/merged_metadata.jsonl.gz`

From this point onward, **all downstream retrieval in this project uses the merged index** (faster and simpler than querying each shard).

---

### 4. Run RAG answering (uses merged index)

You can either:

- **Use local Qwen** (via `rag_query.py` / `rag_query_local.py`), or  
- **Use an API model** (e.g. Qwen API) via `rag_query_api.py`.

For batch answering over many queries (recommended for UE runs), retrieval is done from the merged index and the top documents are passed to the generator:

```bash
python scripts/batch_rag.py \
  --config configs/default.yaml
```

Your answers file must contain at least:
- `qid`
- `query`
- `answer`

This is the file you will feed into SAFE and UE.

---

### 5. SAFE scoring (factual correctness labels)

There are two SAFE front-ends:

- **`scripts/safe_with_qwen.py`** – uses Qwen as the SAFE model.
- **`scripts/safe_with_openai.py`** – uses an OpenAI-style API as SAFE model.

Example (adapt to your environment / API keys):

```bash
python scripts/safe_with_qwen.py \
  --answers_path outputs/rag_answers_...jsonl \
  --output_path outputs/safe_scores_...jsonl
```

The resulting SAFE file should contain:
- `qid`
- `query`
- `answer`
- `safe_score` (in \[0, 1\] or similar scale)

If you have multiple SAFE runs or checkpoints, you can consolidate them with:

```bash
python scripts/merge_safe_labels.py \
  --input_paths outputs/safe_scores_*.jsonl \
  --output_path outputs/safe_checkpoint.jsonl
```

---

### 6. Uncertainty Estimation (uses merged index for re-retrieval)

The main UE driver is `scripts/eval_ue_phase2.py`.

Run UE over a pair of (answers, SAFE) files:

```bash
python scripts/eval_ue_phase2.py \
  --answers_path outputs/rag_answers_...jsonl \
  --safe_path outputs/safe_scores_...jsonl \
  --config configs/default.yaml
```

What it does:
- Loads retriever (Contriever) and answering model (Qwen or fallback LM).
- For each shared `qid`:
  - **White-box UE**: computes average negative log-likelihood of the answer under the LM, conditioned on the question.
  - **Black-box UE**:
    - Re-retrieves documents for the query using the merged FAISS index.
    - Computes MARS-style faithfulness using a RoBERTa MNLI model.
    - Defines `ue_black = 1 - faithfulness`.
- Accumulates per-query records and global correlation stats.

Outputs:
- `outputs/ue_scores_...jsonl`:
  - `qid`, `query`, `answer`
  - `safe_score`
  - `ue_white_nll`, `ue_white_num_tokens`
  - `mars_faithfulness`, `ue_black_mars_uncertainty`
- `outputs/ue_correlation_summary.json`:
  - `num_examples`
  - `pearson_white_nll`, `spearman_white_nll`
  - `pearson_black_mars_uncertainty`, `spearman_black_mars_uncertainty`
  - paths + model metadata

You can inspect `ue_correlation_summary.json` directly to see how well uncertainty tracks SAFE.

---

## Script reference (quick)

- **`scripts/utils.py`**:
  - `load_config` – YAML config loader.
  - `load_model_and_tokenizer`, `load_model_and_tokenizer2` – load Contriever/Qwen with correct devices and quantization.
  - `encode_batch` – batched encoding with mean pooling.

- **Indexing**:
  - `download_dataset.py` – downloads HF dataset to `data/`.
  - `build_shard.py` – builds `IndexFlatIP` per shard from the corpus.
  - `query_index.py` – interactive debugging for retrieval quality.

- **RAG**:
  - `rag_query.py`, `rag_query_local.py` – local RAG for single queries / small batches.
  - `rag_query_api.py` – RAG over an external API model.
  - `batch_rag.py` – end-to-end batched answering for UE experiments.

- **SAFE & UE**:
  - `safe_with_openai.py`, `safe_with_qwen.py` – SAFE scoring front-ends.
  - `merge_safe_labels.py` – merge SAFE runs / checkpoints.
  - `eval_ue_phase2.py` – compute white-box and black-box UE and correlations.

---

## Troubleshooting notes

- **Slow runs / long UE time**:
  - UE is expensive: it runs LM NLL + RoBERTa MNLI + multi-shard retrieval per query.
  - For quick experiments, reduce the number of queries or shards, or lower `top_k_per_shard`.

- **GPU vs CPU**:
  - Qwen answering uses GPU INT4 in `load_model_and_tokenizer2`; if no GPU is found, scripts may fall back to a small CPU LM (e.g., `gpt2`) for NLL.
  - MARS (RoBERTa MNLI) defaults to CPU but can be switched to GPU inside `MarsScorer` if desired.

- **Out-of-memory**:
  - Lower `batch_size` in `configs/default.yaml`.
  - Disable fp16 (`use_fp16: false`) if you see stability issues.

---

## Notes

- Shards are independent; you can build and query them in any order.
- Metadata files store full document text for RAG, so they can be large.
- The code currently uses exact `IndexFlatIP`; swapping to IVF/HNSW would require minor FAISS changes in `build_shard.py`.
