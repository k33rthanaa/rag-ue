import faiss
import json
import gzip
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def merge_shards(output_root, total_shards, merged_dir):
    """
    Merge all FAISS shards into a single index with metadata
    """
    output_root = Path(output_root)
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🔄 MERGING FAISS SHARDS")
    print("="*80)
    print(f"📂 Source: {output_root}")
    print(f"📂 Target: {merged_dir}")
    print(f"📦 Total shards: {total_shards}")
    print("="*80 + "\n")
    
    # Step 1: Get dimension from first shard
    print("📏 Step 1/4: Reading first shard to get dimensions...")
    first_shard_path = output_root / "shard_0000" / "shard_0000.index"
    
    if not first_shard_path.exists():
        raise FileNotFoundError(f"First shard not found: {first_shard_path}")
    
    first_index = faiss.read_index(str(first_shard_path))
    dimension = first_index.d
    print(f"   ✅ Embedding dimension: {dimension}")
    print(f"   ✅ First shard has {first_index.ntotal} vectors\n")
    
    # Step 2: Create merged index
    print("🔧 Step 2/4: Creating merged FAISS index...")
    merged_index = faiss.IndexFlatIP(dimension)
    print(f"   ✅ Created empty IndexFlatIP with dimension {dimension}\n")
    
    # Step 3: Merge all shard embeddings
    print("📦 Step 3/4: Merging embeddings from all shards...")
    total_vectors = 0
    
    for shard_id in tqdm(range(total_shards), desc="Processing shards"):
        shard_dir = output_root / f"shard_{shard_id:04d}"
        index_path = shard_dir / f"shard_{shard_id:04d}.index"
        
        if not index_path.exists():
            print(f"   ⚠️  Shard {shard_id} missing, skipping...")
            continue
        
        # Load shard index
        shard_index = faiss.read_index(str(index_path))
        num_vectors = shard_index.ntotal
        
        # Extract embeddings
        embeddings = np.empty((num_vectors,dimension),dtype='float32')
        shard_index.reconstruct_n(0,num_vectors,embeddings)
        # Add to merged index
        merged_index.add(embeddings)
        total_vectors += num_vectors
        
        tqdm.write(f"   ✅ Shard {shard_id:04d}: Added {num_vectors:,} vectors (Total: {total_vectors:,})")
    
    print(f"\n   ✅ Total vectors in merged index: {merged_index.ntotal:,}\n")
    
    # Step 4: Merge all metadata files
    print("📝 Step 4/4: Merging metadata files...")
    merged_meta_path = merged_dir / "merged_metadata.jsonl.gz"
    total_records = 0
    
    with gzip.open(merged_meta_path, "wt", encoding="utf-8") as out_f:
        for shard_id in tqdm(range(total_shards), desc="Processing metadata"):
            shard_dir = output_root / f"shard_{shard_id:04d}"
            meta_path = shard_dir / f"shard_{shard_id:04d}.meta.jsonl.gz"
            
            if not meta_path.exists():
                print(f"   ⚠️  Metadata for shard {shard_id} missing, skipping...")
                continue
            
            shard_records = 0
            with gzip.open(meta_path, "rt", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
                        shard_records += 1
                        total_records += 1
            
            tqdm.write(f"   ✅ Shard {shard_id:04d}: Copied {shard_records:,} records (Total: {total_records:,})")
    
    print(f"\n   ✅ Total metadata records: {total_records:,}\n")
    
    # Step 5: Verify and save
    print("💾 Step 5/4: Saving merged index...")
    
    # Sanity check
    if merged_index.ntotal != total_records:
        print(f"   ⚠️  WARNING: Mismatch detected!")
        print(f"      Vectors: {merged_index.ntotal:,}")
        print(f"      Metadata: {total_records:,}")
    else:
        print(f"   ✅ Verification passed: {merged_index.ntotal:,} vectors = {total_records:,} metadata records")
    
    # Save merged index
    merged_index_path = merged_dir / "merged_index.index"
    faiss.write_index(merged_index, str(merged_index_path))
    
    print(f"\n" + "="*80)
    print("✅ MERGING COMPLETE!")
    print("="*80)
    print(f"📊 Final Statistics:")
    print(f"   - Total vectors: {merged_index.ntotal:,}")
    print(f"   - Total metadata records: {total_records:,}")
    print(f"   - Dimension: {dimension}")
    print(f"\n📁 Output Files:")
    print(f"   - Index:    {merged_index_path}")
    print(f"   - Metadata: {merged_meta_path}")
    print("="*80 + "\n")
    
    return merged_index_path, merged_meta_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge FAISS shards into single index")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Directory containing shard_XXXX folders (default: outputs)"
    )
    parser.add_argument(
        "--total_shards",
        type=int,
        required=True,
        help="Total number of shards (e.g., 11 for shard_0000 to shard_0010)"
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default="outputs/merged",
        help="Where to save merged files (default: outputs/merged)"
    )
    
    args = parser.parse_args()
    
    merge_shards(args.output_root, args.total_shards, args.merged_dir)
