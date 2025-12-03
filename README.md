# RAG-Uncertainty-Estimator

This project aims to build a Retrieval-Augmented Generation (RAG) system with 
Uncertainty Estimation (UE). It will use Contriever for retrieval, Qwen2.5-7B-Instruct 
for generation, and MARS &amp; Eccentricity for UE. The goal is to improve response 
accuracy using large-scale datasets like FactScore-Bio and LongFact-Objects.


## 🎯 Overview

This repository provides tools for:
- **Downloading datasets** from Hugging Face
- **Building distributed FAISS indices** by sharding large datasets
- **Querying indices** for semantic search
- **Scalable processing** suitable for cluster environments (Slurm)

The system is designed to handle large-scale datasets (e.g., wiki-18 corpus with millions of documents) by splitting them into manageable shards that can be processed in parallel.

## ✨ Features

- **Distributed Indexing**: Process large datasets in parallel by splitting into shards
- **Efficient Embeddings**: Uses Contriever for high-quality semantic embeddings
- **FAISS Integration**: Fast similarity search with normalized inner-product indices
- **Streaming Processing**: Memory-efficient processing of large JSONL.gz files
- **Metadata Preservation**: Stores document metadata alongside embeddings for retrieval
- **Cluster-Ready**: Includes Slurm job scripts for HPC environments
- **Configurable**: YAML-based configuration for easy customization

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster encoding)
- Sufficient disk space for dataset and indices

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAG-Uncertainty-Estimator
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: For GPU support, you may want to install `faiss-gpu` instead of `faiss-cpu`:
   ```bash
   pip install faiss-gpu
   ```

## 📁 Project Structure

```
RAG-Uncertainty-Estimator/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── configs/                 # Configuration files
│   ├── default.yaml         # Main configuration
│   ├── shard_config.yaml    # (Reserved for future use)
│   └── model_config.yaml    # (Reserved for future use)
│
├── data/                    # Dataset storage (git-ignored)
│   └── wiki-18.jsonl.gz     # Downloaded dataset
│
├── scripts/                 # Main Python scripts
│   ├── download_dataset.py  # Download dataset from Hugging Face
│   ├── build_shard.py      # Build a single shard index
│   ├── merge_indices.py    # (Reserved for future use)
│   ├── query_index.py      # Query a shard index interactively
│   └── utils.py            # Shared utilities (encoding, config loading)
│
├── slurm/                   # Slurm job scripts for cluster execution
│   ├── download.sh          # Download dataset job
│   └── build_shard.sh      # Build shard job
│
├── outputs/                 # Generated indices and metadata (git-ignored)
│   ├── shard_0000/
│   │   ├── shard_0000.index           # FAISS index file
│   │   └── shard_0000.meta.jsonl.gz   # Document metadata
│   ├── shard_0001/
│   ├── logs/                # Job logs
│   └── hf_cache/            # HuggingFace model cache
│
└── tests/                   # Unit tests
    └── test_encoding.py     # (Reserved for future use)
```

## ⚙️ Configuration

The main configuration file is `configs/default.yaml`. Key settings:

### Model & Encoding
- `model_name`: Hugging Face model identifier (default: `"facebook/contriever"`)
- `batch_size`: Batch size for encoding (default: `64`)
- `max_length`: Maximum tokens per document (default: `512`)
- `use_fp16`: Use half-precision for GPU (default: `true`)

### Dataset
- `dataset.repo_id`: Hugging Face dataset repository
- `dataset.filename`: Dataset filename
- `dataset.local_path`: Local storage path
- `text_field`: JSON key containing document text (default: `"contents"`)

### Sharding
- `sharding.shard_size`: Documents per shard (default: `2000000`)
- `sharding.save_texts`: Whether to store raw texts (default: `false`)

### Index
- `index.normalize`: L2-normalize embeddings (default: `true`)
- `index.index_type`: FAISS index type (default: `"flat_ip"`)

### Paths
- `paths.output_root`: Output directory (default: `"outputs"`)
- `paths.logs_dir`: Log directory
- `paths.hf_cache`: HuggingFace cache directory

## 📖 Usage Guide

### Step 1: Download Dataset

Download the dataset from Hugging Face to the local `data/` directory:

```bash
python scripts/download_dataset.py
```

Or with a custom config:
```bash
python scripts/download_dataset.py --config configs/default.yaml
```

**What it does:**
- Downloads the dataset specified in `configs/default.yaml` from Hugging Face
- Saves it to `data/wiki-18.jsonl.gz` (or path specified in config)
- Uses the `huggingface_hub` library for efficient downloading

### Step 2: Build Shard Indices

Build FAISS indices for each shard. Each shard processes a subset of documents:

```bash
python scripts/build_shard.py --shard_id 0
```

**Arguments:**
- `--shard_id` (required): Shard ID to build (0, 1, 2, ...)
- `--shard_size` (optional): Override shard size from config
- `--config` (optional): Path to config file (default: `configs/default.yaml`)

**What it does:**
1. Loads the encoder model (Contriever) and tokenizer
2. Reads the dataset file and processes documents in the shard's range
3. Encodes documents in batches using the encoder model
4. Normalizes embeddings (if configured)
5. Builds a FAISS index with inner-product similarity
6. Saves:
   - `shard_XXXX.index`: FAISS index file
   - `shard_XXXX.meta.jsonl.gz`: Compressed metadata (title, contents, IDs)

**Example: Building multiple shards**
```bash
# Build shard 0 (documents 0-2M)
python scripts/build_shard.py --shard_id 0

# Build shard 1 (documents 2M-4M)
python scripts/build_shard.py --shard_id 1

# Build shard 2 (documents 4M-6M)
python scripts/build_shard.py --shard_id 2
```

**Shard Range Calculation:**
- Shard 0: documents [0 × shard_size, 1 × shard_size)
- Shard 1: documents [1 × shard_size, 2 × shard_size)
- Shard N: documents [N × shard_size, (N+1) × shard_size)

### Step 3: Query the Index

Query a built shard index interactively:

```bash
python scripts/query_index.py --shard_id 0 --top_k 5
```

**Arguments:**
- `--shard_id` (required): Shard ID to query
- `--top_k` (optional): Number of results to return (default: `5`)
- `--config` (optional): Path to config file

**What it does:**
1. Loads the FAISS index and metadata for the specified shard
2. Loads the encoder model for query encoding
3. Enters an interactive loop where you can:
   - Type queries and get top-k results
   - See similarity scores, document IDs, titles, and snippets
   - Type `exit` or `quit` to stop

**Example session:**
```
Query> What is machine learning?
[Results table with rank, score, doc ID, title, snippet]

Query> How does neural network training work?
[Results table...]

Query> exit
Bye!
```

## 🔧 Core Components

### `scripts/utils.py`

Shared utility functions used across scripts:

- **`load_config(config_path)`**: Loads YAML configuration file
- **`get_device()`**: Returns CUDA device if available, else CPU
- **`load_model_and_tokenizer(cfg)`**: Loads tokenizer and model from config
- **`mean_pooling(token_embeddings, attention_mask)`**: Mean pooling over non-padding tokens
- **`encode_batch(texts, tokenizer, model, device, max_length)`**: Encodes a batch of texts into embeddings

### `scripts/download_dataset.py`

Downloads datasets from Hugging Face:
- Reads dataset configuration from YAML
- Uses `huggingface_hub` to download files
- Saves to local `data/` directory

### `scripts/build_shard.py`

Builds a single shard index:
- Streams through dataset file (memory-efficient)
- Processes documents in batches
- Encodes using Contriever model
- Builds FAISS index with normalized embeddings
- Saves index and metadata files

### `scripts/query_index.py`

Interactive query interface:
- Loads a shard index and metadata
- Encodes queries using the same model
- Performs similarity search
- Displays results in a formatted table

## 🖥️ Slurm Cluster Usage

For HPC environments, use the provided Slurm scripts:

### Download Dataset
```bash
sbatch slurm/download.sh
```

### Build Shard
```bash
sbatch slurm/build_shard.sh 0   # Build shard 0
sbatch slurm/build_shard.sh 1   # Build shard 1
```

**Note**: Update the `PROJECT_ROOT` path in the Slurm scripts to match your cluster setup.

The scripts:
- Set up the environment (activate venv, set HF_HOME)
- Run the Python scripts with appropriate arguments
- Save logs to `outputs/logs/`

## 🔍 Understanding the Workflow

### 1. Dataset Format

The system expects JSONL.gz files where each line is a JSON object:
```json
{"id": "doc_123", "title": "Document Title", "contents": "Document text here..."}
```

The `contents` field (or field specified by `text_field` in config) is used for encoding.

### 2. Sharding Strategy

Large datasets are split into shards for parallel processing:
- Each shard processes a contiguous range of documents
- Shards can be built independently (useful for parallel execution)
- Each shard produces its own FAISS index

### 3. Embedding Process

1. **Tokenization**: Text is tokenized using the model's tokenizer
2. **Encoding**: Tokenized text is passed through the encoder model
3. **Pooling**: Mean pooling over token embeddings (excluding padding)
4. **Normalization**: L2 normalization for cosine similarity
5. **Indexing**: Normalized embeddings are added to FAISS index

### 4. Query Process

1. **Query Encoding**: Query text is encoded using the same model
2. **Normalization**: Query embedding is normalized (if index was normalized)
3. **Search**: FAISS performs inner-product search (cosine similarity for normalized vectors)
4. **Retrieval**: Top-k results are retrieved with scores
5. **Metadata Lookup**: Document metadata is loaded from metadata file

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

- **Reduce batch size**: Set `batch_size: 32` or lower in `configs/default.yaml`
- **Use CPU**: The system will fall back to CPU if CUDA is unavailable
- **Disable FP16**: Set `use_fp16: false` if you encounter precision issues

### Slow Encoding

- **Use GPU**: Ensure CUDA is available (`torch.cuda.is_available()`)
- **Enable FP16**: Set `use_fp16: true` for faster GPU encoding
- **Increase batch size**: Larger batches are more efficient (if memory allows)

### Dataset Not Found

- Verify the `dataset.repo_id` and `dataset.filename` in config
- Check Hugging Face access (some datasets require authentication)
- Ensure the dataset repository exists and is accessible

### Index/Metadata Mismatch

- Ensure metadata file was written correctly during indexing
- Check that the metadata file wasn't corrupted
- Verify that documents were processed in order

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the correct Python version (3.8+)
- Verify virtual environment is activated

## 📝 Notes

- **Memory Efficiency**: The system streams through dataset files, so memory usage is independent of dataset size
- **Shard Independence**: Each shard is independent; you can build them in any order or in parallel
- **Metadata Storage**: Metadata includes full document text (for RAG), so metadata files can be large
- **Index Type**: Currently uses `IndexFlatIP` (exact search). For larger indices, consider `IndexIVFFlat` or `IndexHNSWFlat` (requires code modifications)

## 🔮 Future Enhancements

- Multi-shard querying (search across all shards)
- Index merging functionality
- Support for different FAISS index types (IVF, HNSW)
- Uncertainty estimation integration (MARS, Eccentricity)
- Generator model integration (Qwen2.5-7B-Instruct)

