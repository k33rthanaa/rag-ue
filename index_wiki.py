"""
Standalone Wikipedia Corpus Indexing Script for M4 MacBook Air
Indexes PeterJinGo/wiki-18-corpus and stores embeddings in FAISS
Reads JSONL files directly to handle corrupted datasets
"""

import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
from tqdm import tqdm
import pickle
import os
import argparse
import gzip
import json
from pathlib import Path


class WikiIndexer:
    def __init__(self, model_name="facebook/contriever", batch_size=32):
        self.batch_size = batch_size
        self.device = self.get_device()
        
        print(f"🔧 Loading model on device: {self.device}", flush=True)
        
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        print("✅ Model loaded successfully", flush=True)
        
    def get_device(self):
        """Get best available device for M4 Mac"""
        if torch.backends.mps.is_available():
            print("🚀 MPS (Metal) GPU acceleration available!", flush=True)
            return torch.device("mps")
        print("⚠️  MPS not available, using CPU", flush=True)
        return torch.device("cpu")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_batch(self, texts):
        """Encode a batch of texts to embeddings"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        
        return embeddings.cpu().numpy()
    
    def find_cached_dataset(self):
        """Find the downloaded dataset file in HuggingFace cache"""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        pattern = "datasets--PeterJinGo--wiki-18-corpus"
        
        for path in cache_dir.glob(f"*{pattern}*/snapshots/*/wiki-18.jsonl.gz"):
            return str(path)
        
        return None
    
    def load_from_jsonl_gz(self, file_path, max_samples=None, text_field="contents"):
        """Load data from gzipped JSONL file, skipping corrupted lines"""
        print(f"📚 Loading from file: {file_path}", flush=True)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data_list = []
        errors = 0
        
        print("📖 Reading and parsing JSONL...", flush=True)
        
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(tqdm(f, desc="Reading lines")):
                if max_samples and len(data_list) >= max_samples:
                    break
                
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                    
                    if text_field in entry and entry[text_field]:
                        data_list.append(entry)
                    
                except (json.JSONDecodeError, UnicodeDecodeError):
                    errors += 1
                    if errors <= 10:
                        print(f"⚠️  Skipping corrupted entry at line {i}", flush=True)
                    continue
                
                if (i + 1) % 10000 == 0 and len(data_list) > 0:
                    print(f"  → Processed {i+1} lines, collected {len(data_list)} valid entries, {errors} errors", flush=True)
        
        print(f"✅ Loaded {len(data_list)} valid entries (skipped {errors} corrupted entries)", flush=True)
        
        if len(data_list) == 0:
            raise ValueError("No valid entries found in the dataset file!")
        
        return data_list
    
    def index_dataset(self, data_list, text_field="contents", save_path="./wiki_index"):
        """Index the dataset and save to FAISS"""
        print(f"\n🔨 Starting indexing process...", flush=True)
        print(f"📝 Total documents to index: {len(data_list)}", flush=True)
        
        all_embeddings = []
        all_texts = []
        all_metadata = []
        
        num_batches = (len(data_list) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(data_list), self.batch_size), 
                     total=num_batches, 
                     desc="🚀 Encoding batches"):
            
            batch = data_list[i:i + self.batch_size]
            
            texts = []
            for entry in batch:
                text = entry.get(text_field, "")
                texts.append(text if text else "")
            
            if not any(texts):
                continue
            
            try:
                embeddings = self.encode_batch(texts)
                all_embeddings.append(embeddings)
                all_texts.extend(texts)
                
                for j, entry in enumerate(batch):
                    meta = {'id': i + j}
                    for key, value in entry.items():
                        if key != text_field:
                            meta[key] = value
                    all_metadata.append(meta)
                    
            except Exception as e:
                print(f"\n⚠️  Error processing batch {i}: {e}", flush=True)
                continue
        
        all_embeddings = np.vstack(all_embeddings).astype('float32')
        print(f"\n✅ Embeddings generated: {all_embeddings.shape}", flush=True)
        
        print("\n🏗️  Building FAISS index...", flush=True)
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(all_embeddings)
        index.add(all_embeddings)
        
        print(f"✅ FAISS index built with {index.ntotal} vectors", flush=True)
        
        self.save_index(index, all_texts, all_metadata, save_path)
        
        return index, all_texts, all_metadata
    
    def save_index(self, index, texts, metadata, save_path):
        """Save FAISS index and metadata to disk"""
        os.makedirs(save_path, exist_ok=True)
        
        index_path = os.path.join(save_path, "faiss_index.bin")
        faiss.write_index(index, index_path)
        print(f"💾 FAISS index saved: {index_path}", flush=True)
        
        data_path = os.path.join(save_path, "documents.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({'texts': texts, 'metadata': metadata}, f)
        print(f"💾 Documents saved: {data_path}", flush=True)
        
        config_path = os.path.join(save_path, "config.txt")
        with open(config_path, 'w') as f:
            f.write(f"Total documents: {len(texts)}\n")
            f.write(f"Embedding dimension: {index.d}\n")
            f.write(f"Index type: {type(index).__name__}\n")
        print(f"💾 Config saved: {config_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Index Wikipedia corpus with Contriever')
    parser.add_argument('--file-path', type=str, default=None,
                       help='Path to wiki-18.jsonl.gz file (auto-detected if not provided)')
    parser.add_argument('--model', type=str, default='facebook/contriever',
                       help='Model name from HuggingFace')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding (lower if OOM)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to index (None for all)')
    parser.add_argument('--text-field', type=str, default='contents',
                       help='Name of the text field in JSON')
    parser.add_argument('--output-dir', type=str, default='./wiki_index',
                       help='Directory to save index and metadata')
    
    args = parser.parse_args()
    
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    print("=" * 60, flush=True)
    print("📇 WIKIPEDIA CORPUS INDEXER", flush=True)
    print("=" * 60, flush=True)
    
    try:
        indexer = WikiIndexer(model_name=args.model, batch_size=args.batch_size)
        
        # Find dataset file
        if args.file_path:
            file_path = args.file_path
        else:
            print("🔍 Auto-detecting cached dataset file...", flush=True)
            file_path = indexer.find_cached_dataset()
            if not file_path:
                print("❌ Could not find cached dataset file!", flush=True)
                print("\n💡 Please specify the file path manually:", flush=True)
                print("   python index_wiki.py --file-path /path/to/wiki-18.jsonl.gz", flush=True)
                return
        
        print(f"📁 Using file: {file_path}", flush=True)
        print(f"🤖 Model: {args.model}", flush=True)
        print(f"📦 Batch size: {args.batch_size}", flush=True)
        print(f"📊 Max samples: {args.max_samples or 'All'}", flush=True)
        print(f"🔑 Text field: {args.text_field}", flush=True)
        print("=" * 60 + "\n", flush=True)
        
        # Load data from file
        data_list = indexer.load_from_jsonl_gz(
            file_path=file_path,
            max_samples=args.max_samples,
            text_field=args.text_field
        )
        
        # Index and save
        index, texts, metadata = indexer.index_dataset(
            data_list=data_list,
            text_field=args.text_field,
            save_path=args.output_dir
        )
        
        print("\n" + "=" * 60, flush=True)
        print("✅ INDEXING COMPLETE!", flush=True)
        print("=" * 60, flush=True)
        print(f"📁 Index location: {args.output_dir}/", flush=True)
        print(f"📊 Total documents indexed: {len(texts)}", flush=True)
        print(f"💾 Index size: {index.ntotal} vectors", flush=True)
        print("\n💡 Next steps:", flush=True)
        print("   1. Use the saved index for retrieval", flush=True)
        print("   2. Files created:", flush=True)
        print(f"      - {args.output_dir}/faiss_index.bin", flush=True)
        print(f"      - {args.output_dir}/documents.pkl", flush=True)
        print(f"      - {args.output_dir}/config.txt", flush=True)
        print("=" * 60 + "\n", flush=True)
        
    except KeyboardInterrupt:
        print("\n\n❌ Indexing interrupted by user", flush=True)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()