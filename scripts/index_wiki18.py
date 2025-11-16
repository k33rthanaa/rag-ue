"""
Standalone Wikipedia Corpus Indexing Script (STREAMING VERSION)

Indexes PeterJinGo/wiki-18-corpus and stores embeddings in FAISS.

- Works on Mac (MPS), GPU clusters (CUDA) and CPU.
- Reads JSONL(.gz) directly and is robust to corrupted / weird lines.
- Does NOT assume a fixed JSON schema: will try the given text field,
  then fall back to other string fields, then the whole line.
- **STREAMING**: does NOT keep all embeddings in RAM at once.
- Logs progress to stdout AND to <output-dir>/progress.log
"""

import os
import sys
import json
import gzip
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Make project root importable if needed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------- helpers for text extraction -----------------

TEXT_KEYS = [
    "contents",
    "text",
    "content",
    "body",
    "passage",
    "paragraph",
    "article_body",
    "article",
    "doc",
    "document",
    "wiki_text",
]

TITLE_KEYS = ["title", "page_title", "name", "heading"]


def extract_text_title(obj, text_field: str):
    """
    Best effort extraction of (title, text) from a JSON object.

    Priority:
      1) If text_field is present and non-empty, use that as text.
      2) Else try common TEXT_KEYS.
      3) Else join all string values as text.

    Title:
      - Try TITLE_KEYS, else empty.
    """
    title = ""
    text = ""

    if isinstance(obj, dict):
        # title first
        for k in TITLE_KEYS:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                title = obj[k].strip()
                break

        # explicit text field
        if (
            text_field
            and text_field in obj
            and isinstance(obj[text_field], str)
            and obj[text_field].strip()
        ):
            text = obj[text_field].strip()
        else:
            # try common keys
            for k in TEXT_KEYS:
                if k in obj and isinstance(obj[k], str) and obj[k].strip():
                    text = obj[k].strip()
                    break

        # fallback: concat all string values
        if not text:
            parts = [
                v.strip()
                for v in obj.values()
                if isinstance(v, str) and v.strip()
            ]
            text = " ".join(parts)

    elif isinstance(obj, (list, tuple)):
        if len(obj) >= 3:
            title = str(obj[1]).strip()
            text = str(obj[2]).strip()
        elif len(obj) == 2:
            text = str(obj[1]).strip()
        elif len(obj) == 1:
            text = str(obj[0]).strip()

    return title.strip(), text.strip()


# --------------- streaming dataset iterator -----------------


def iter_entries_from_jsonl_gz(
    file_path,
    max_samples=None,
    text_field="contents",
    skip: int = 0,
):
    """
    Streaming version of JSONL(.gz) loader.

    Yields one {"title": ..., "text": ...} dict at a time instead of
    building a giant list in memory.

    Parameters
    ----------
    file_path : str
        Path to wiki-18.jsonl.gz
    max_samples : int or None
        Maximum number of *indexed* documents to yield (after skipping).
        None = no limit.
    text_field : str
        Preferred text field name in JSON.
    skip : int
        Number of *valid documents* to skip before yielding.
        This should match docs_indexed from a previous run if you resume.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f" Streaming from file: {file_path}", flush=True)
    print(f" [INFO] skip={skip} max_samples={max_samples}", flush=True)

    errors = 0
    num_indexed = 0        # docs actually yielded (after skipping)
    num_skipped = 0        # docs skipped (valid ones)
    num_valid_total = 0    # valid docs seen (skipped + indexed)

    print("📖 Reading and parsing JSONL (streaming)...", flush=True)

    with gzip.open(file_path, "rb") as f:
        for i, raw in enumerate(tqdm(f, desc="Reading lines")):
            # If we've already yielded the max number of docs, stop
            if max_samples is not None and num_indexed >= max_samples:
                break

            # decode safely, ignore broken bytes
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                errors += 1
                continue

            if not line:
                continue

            title = ""
            text = ""

            if line[0] in "{[":
                # looks like JSON
                try:
                    obj = json.loads(line)
                    title, text = extract_text_title(obj, text_field=text_field)
                except Exception:
                    errors += 1
                    if errors <= 5:
                        print(
                            f"  JSON parse error at line {i}, "
                            f"falling back to raw text",
                            flush=True,
                        )
                    text = line
            else:
                text = line

            if not text:
                continue

            # we now consider this a "valid doc" (text exists)
            num_valid_total += 1

            # handle skipping of valid docs
            if num_skipped < skip:
                num_skipped += 1
                if num_skipped % 100000 == 0:
                    print(
                        f"  → Skipped {num_skipped} valid docs so far...",
                        flush=True,
                    )
                continue

            # no more skipping -> yield entries
            yield {"title": title, "text": text}
            num_indexed += 1

            if (i + 1) % 10000 == 0 and num_indexed > 0:
                print(
                    f"  → Seen {i+1} lines, valid_docs={num_valid_total}, "
                    f"skipped={num_skipped}, indexed={num_indexed}, "
                    f"{errors} JSON errors",
                    flush=True,
                )

    print(
        f" Streamed {num_indexed} indexed docs after skipping {num_skipped} "
        f"valid docs (total_valid={num_valid_total}, JSON_errors={errors})",
        flush=True,
    )

    if num_indexed == 0:
        raise ValueError(
            "No valid entries were indexed (check skip / max-samples / dataset?)"
        )


# --------------- indexer class -----------------


class WikiIndexer:
    def __init__(self, model_name: str = "facebook/contriever", batch_size: int = 32):
        self.batch_size = batch_size
        self.device = self.get_device()

        print(f" Loading model '{model_name}' on device: {self.device}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(" Model loaded successfully", flush=True)

    def get_device(self):
        """Pick best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            print(" CUDA GPU available!", flush=True)
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(" MPS (Metal) GPU acceleration available!", flush=True)
            return torch.device("mps")
        print(" No GPU found, using CPU", flush=True)
        return torch.device("cpu")

    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode_batch(self, texts):
        """Encode a batch of texts to embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self.mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )

        return pooled.cpu().numpy()

    # --------------- dataset discovery -----------------

    def find_cached_dataset(self):
        """
        Try to locate wiki-18.jsonl.gz in HuggingFace cache.

        Checks HF_HOME, then home cache.
        """
        # 1) HF_HOME or HF_DATASETS_CACHE, if set
        for base_env in ("HF_DATASETS_CACHE", "HF_HOME"):
            base = os.environ.get(base_env)
            if not base:
                continue
            base_path = Path(base)
            if base_path.exists():
                for path in base_path.rglob("wiki-18.jsonl.gz"):
                    return str(path)

        # 2) Default HF hub cache (~/.cache/huggingface/hub)
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            for path in cache_dir.rglob("wiki-18.jsonl.gz"):
                return str(path)

        return None

    # --------------- legacy loader (kept for debugging) -----------------

    def load_from_jsonl_gz(self, file_path, max_samples=None, text_field="contents"):
        """
        Legacy non-streaming loader (kept for debugging / small subsets).

        Loads data from gzipped JSONL file into a list in memory.
        """
        print(f" Loading from file (non-streaming): {file_path}", flush=True)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        entries = []
        errors = 0

        print("📖 Reading and parsing JSONL...", flush=True)

        with gzip.open(file_path, "rb") as f:
            for i, raw in enumerate(tqdm(f, desc="Reading lines")):
                if max_samples is not None and len(entries) >= max_samples:
                    break

                # decode safely, ignore broken bytes
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                title = ""
                text = ""

                if line[0] in "{[":
                    # looks like JSON
                    try:
                        obj = json.loads(line)
                        title, text = extract_text_title(obj, text_field=text_field)
                    except Exception:
                        errors += 1
                        if errors <= 5:
                            print(
                                f"  JSON parse error at line {i}, "
                                f"falling back to raw text",
                                flush=True,
                            )
                        text = line
                else:
                    text = line

                if not text:
                    continue

                entries.append({"title": title, "text": text})

                if (i + 1) % 10000 == 0 and len(entries) > 0:
                    print(
                        f"  → Processed {i+1} lines, collected {len(entries)} valid entries, "
                        f"{errors} JSON errors",
                        flush=True,
                    )

        print(
            f" Loaded {len(entries)} valid entries (skipped {errors} JSON errors)",
            flush=True,
        )

        if len(entries) == 0:
            raise ValueError("No valid entries found in the dataset file!")

        return entries

    # --------------- streaming indexing -----------------

    def index_dataset_streaming(
        self,
        file_path,
        save_path="./wiki_index",
        max_samples=None,
        text_field="contents",
        skip: int = 0,
    ):
        """
        Stream the dataset from disk, index it, and save to FAISS.

        This avoids storing all embeddings in RAM at once:
        - we only keep one batch of embeddings in memory
        - we still store all texts + metadata (for retrieval)

        Parameters
        ----------
        skip : int
            Number of valid docs to skip before indexing (resume support).
        """
        os.makedirs(save_path, exist_ok=True)

        progress_path = os.path.join(save_path, "progress.log")
        prog = open(progress_path, "a", buffering=1, encoding="utf-8")

        def log(msg: str):
            ts = datetime.now().isoformat(timespec="seconds")
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            prog.write(line + "\n")

        log("==== START INDEXING (STREAMING) ====")
        log(f"params: skip={skip} max_samples={max_samples}")

        all_texts = []
        all_metadata = []

        index = None
        batch_entries = []
        total_seen = 0
        total_indexed = 0
        next_id = 0

        print(f"\n Starting streaming indexing process...", flush=True)

        # Stream entries from disk
        for entry in iter_entries_from_jsonl_gz(
            file_path=file_path,
            max_samples=max_samples,
            text_field=text_field,
            skip=skip,
        ):
            batch_entries.append(entry)
            total_seen += 1

            if len(batch_entries) >= self.batch_size:
                index, count_added, next_id = self._encode_and_add_batch(
                    batch_entries, index, all_texts, all_metadata, start_id=next_id
                )
                total_indexed += count_added
                batch_entries = []

                if total_indexed % 1000 == 0:
                    log(f"docs_indexed={total_indexed} seen={total_seen}")

        # last partial batch
        if batch_entries:
            index, count_added, next_id = self._encode_and_add_batch(
                batch_entries, index, all_texts, all_metadata, start_id=next_id
            )
            total_indexed += count_added
            batch_entries = []

        if index is None:
            prog.close()
            raise ValueError(
                "No embeddings were produced (empty dataset after cleaning / skip?)."
            )

        print(f"\n Total documents indexed: {total_indexed}", flush=True)
        log(f"docs_indexed={total_indexed} texts_stored={len(all_texts)}")

        # Save to disk
        self.save_index(index, all_texts, all_metadata, save_path)
        log("==== FINISHED INDEXING (STREAMING) ====")
        prog.close()
        return index, all_texts, all_metadata

    def _encode_and_add_batch(
        self,
        batch_entries,
        index,
        all_texts,
        all_metadata,
        start_id: int,
    ):
        """
        Helper: encode one batch and add to FAISS index.
        Returns (index, count_added, next_start_id).
        """
        texts = [e.get("text", "") or "" for e in batch_entries]
        if not any(t.strip() for t in texts):
            return index, 0, start_id

        embeddings = self.encode_batch(texts).astype("float32")
        faiss.normalize_L2(embeddings)

        # create index on first batch
        if index is None:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            print(f"  Created FAISS IndexFlatIP(dim={dim})", flush=True)

        index.add(embeddings)

        # store texts + metadata (much smaller than storing all embeddings)
        all_texts.extend(texts)
        for j, entry in enumerate(batch_entries):
            meta = {
                "id": start_id + j,
                "title": entry.get("title", ""),
            }
            all_metadata.append(meta)

        return index, len(texts), start_id + len(texts)

    # --------------- saving -----------------

    def save_index(self, index, texts, metadata, save_path):
        """Save FAISS index and metadata to disk."""
        os.makedirs(save_path, exist_ok=True)

        index_path = os.path.join(save_path, "faiss_index.bin")
        faiss.write_index(index, index_path)
        print(f" FAISS index saved: {index_path}", flush=True)

        data_path = os.path.join(save_path, "documents.pkl")
        with open(data_path, "wb") as f:
            pickle.dump({"texts": texts, "metadata": metadata}, f)
        print(f" Documents saved: {data_path}", flush=True)

        config_path = os.path.join(save_path, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"Total documents: {len(texts)}\n")
            f.write(f"Embedding dimension: {index.d}\n")
            f.write(f"Index type: {type(index).__name__}\n")
        print(f" Config saved: {config_path}", flush=True)


# --------------- main -----------------


def main():
    parser = argparse.ArgumentParser(
        description="Index Wikipedia corpus with Contriever (streaming)"
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to wiki-18.jsonl.gz file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/contriever",
        help="Model name from HuggingFace or local path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (lower if OOM)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to index (None for all)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="contents",
        help="Preferred text field name in JSON (used if present)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./wiki_index",
        help="Directory to save index and metadata",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of valid documents to skip before indexing (for resume)",
    )

    args = parser.parse_args()

    # ensure line-buffered prints (Python 3.7+)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60, flush=True)
    print(" WIKIPEDIA CORPUS INDEXER (STREAMING)", flush=True)
    print("=" * 60, flush=True)

    try:
        indexer = WikiIndexer(model_name=args.model, batch_size=args.batch_size)

        # Find dataset file
        if args.file_path:
            file_path = args.file_path
        else:
            print(" Auto-detecting cached dataset file...", flush=True)
            file_path = indexer.find_cached_dataset()
            if not file_path:
                print(" Could not find cached dataset file!", flush=True)
                print("\n💡 Please specify the file path manually:", flush=True)
                print(
                    "   python -m scripts.index_wiki18 --file-path /path/to/wiki-18.jsonl.gz",
                    flush=True,
                )
                return

        print(f" Using file: {file_path}", flush=True)
        print(f" Model: {args.model}", flush=True)
        print(f" Batch size: {args.batch_size}", flush=True)
        print(f"Max samples: {args.max_samples or 'All'}", flush=True)
        print(f" Preferred text field: {args.text_field}", flush=True)
        print(f" Output dir: {args.output_dir}", flush=True)
        print(f" Skip (valid docs): {args.skip}", flush=True)
        print("=" * 60 + "\n", flush=True)

        # Stream from file and index on the fly (memory friendly)
        index, texts, metadata = indexer.index_dataset_streaming(
            file_path=file_path,
            save_path=args.output_dir,
            max_samples=args.max_samples,
            text_field=args.text_field,
            skip=args.skip,
        )

        print("\n" + "=" * 60, flush=True)
        print("INDEXING COMPLETE!", flush=True)
        print("=" * 60, flush=True)
        print(f" Index location: {args.output_dir}/", flush=True)
        print(f" Total documents indexed: {len(texts)}", flush=True)
        print(f" Index size: {index.ntotal} vectors", flush=True)
        print("\n Files created:", flush=True)
        print(f"   - {args.output_dir}/faiss_index.bin", flush=True)
        print(f"   - {args.output_dir}/documents.pkl", flush=True)
        print(f"   - {args.output_dir}/config.txt", flush=True)
        print(f"   - {args.output_dir}/progress.log", flush=True)
        print("=" * 60 + "\n", flush=True)

    except KeyboardInterrupt:
        print("\n\n Indexing interrupted by user", flush=True)
    except Exception as e:
        print(f"\n\n ERROR: {e}", flush=True)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
