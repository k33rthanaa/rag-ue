"""
Simple script to download wiki-18-corpus and check its contents
"""

import gzip
import json
import os
from pathlib import Path
from tqdm import tqdm
import urllib.request

def download_dataset(url, output_path):
    """Download dataset with progress bar"""
    print(f"📥 Downloading dataset...")
    print(f"   URL: {url}")
    print(f"   Saving to: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download with progress bar
    def progress_hook(count, block_size, total_size):
        progress = count * block_size
        percent = min(100, progress * 100 / total_size)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r  [{bar}] {percent:.1f}% ({progress/(1024**3):.2f}GB / {total_size/(1024**3):.2f}GB)', end='', flush=True)
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print("\n✅ Download complete!")


def check_file_contents(file_path, num_samples=10):
    """Read and display sample entries from the JSONL file"""
    print(f"\n📖 Checking file contents: {file_path}")
    print(f"   File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
    
    print(f"\n🔍 Reading first {num_samples} valid entries...\n")
    
    valid_count = 0
    error_count = 0
    total_lines = 0
    
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            total_lines += 1
            
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                valid_count += 1
                
                # Display first few entries
                if valid_count <= num_samples:
                    print("=" * 80)
                    print(f"Entry #{valid_count}:")
                    print("-" * 80)
                    print(f"Fields: {list(entry.keys())}")
                    
                    for key, value in entry.items():
                        if isinstance(value, str):
                            preview = value[:200] + "..." if len(value) > 200 else value
                            print(f"\n{key}:")
                            print(f"  {preview}")
                        else:
                            print(f"\n{key}: {value}")
                    print()
                
                # Stop after reading enough samples + extra for stats
                if valid_count >= num_samples + 100:
                    break
                    
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                error_count += 1
                if error_count <= 5:
                    print(f"⚠️  Line {total_lines}: Skipping corrupted entry")
                continue
    
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Total lines processed: {total_lines}")
    print(f"   Valid entries: {valid_count}")
    print(f"   Corrupted entries: {error_count}")
    print(f"   Success rate: {valid_count/(valid_count+error_count)*100:.1f}%")
    
    return valid_count > 0


def find_cached_file():
    """Check if file is already downloaded in HuggingFace cache"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    pattern = "datasets--PeterJinGo--wiki-18-corpus"
    
    for path in cache_dir.glob(f"*{pattern}*/snapshots/*/wiki-18.jsonl.gz"):
        return str(path)
    
    return None


def main():
    print("=" * 80)
    print("📚 WIKI-18-CORPUS DOWNLOADER & CHECKER")
    print("=" * 80)
    
    # Check if already downloaded
    cached_file = find_cached_file()
    
    if cached_file and os.path.exists(cached_file):
        print(f"\n✅ Found cached file:")
        print(f"   {cached_file}")
        print(f"   Size: {os.path.getsize(cached_file) / (1024**3):.2f} GB")
        
        use_cached = input("\n   Use this file? (y/n): ").strip().lower()
        
        if use_cached == 'y':
            file_path = cached_file
        else:
            file_path = None
    else:
        print("\n📂 No cached file found")
        file_path = None
    
    # Download if needed
    if not file_path:
        print("\n🌐 Dataset URL:")
        print("   https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus")
        
        download_dir = "./data"
        file_path = os.path.join(download_dir, "wiki-18.jsonl.gz")
        
        if os.path.exists(file_path):
            print(f"\n✅ File already exists at: {file_path}")
        else:
            print("\n⚠️  Note: This is a 5GB+ file and will take time to download")
            proceed = input("   Proceed with download? (y/n): ").strip().lower()
            
            if proceed != 'y':
                print("\n❌ Download cancelled")
                return
            
            # HuggingFace direct download URL
            url = "https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz"
            download_dataset(url, file_path)
    
    # Check the file contents
    print("\n" + "=" * 80)
    if check_file_contents(file_path, num_samples=5):
        print("\n✅ Dataset is readable!")
        print(f"\n💾 File location: {file_path}")
        print("\n🚀 Next steps:")
        print("   1. Use this file with the manual indexer:")
        print(f"      python index_wiki_manual.py --file-path \"{file_path}\" --max-samples 10000")
        print("\n   2. Or index the full dataset:")
        print(f"      python index_wiki_manual.py --file-path \"{file_path}\"")
    else:
        print("\n❌ Could not read any valid entries from the file")
        print("   The dataset file may be corrupted")
    
    print("=" * 80)


if __name__ == "__main__":
    main()