
#!/bin/bash
#SBATCH --job-name=wiki18_chunk10_gpu
#SBATCH --output=logs/wiki18_chunk10_gpu_%j.out
#SBATCH --error=logs/wiki18_chunk10_gpu_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p csedu
#SBATCH --gres=gpu:1

BASE=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv
REPO="$BASE/rag-ue"
PYTHON="$REPO/.venv/bin/python"

cd "$REPO" || { echo "cd failed"; exit 1; }

export HF_HOME="$BASE/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "CUDA available?"; "$PYTHON" -c "import torch; print(torch.cuda.is_available())"
echo "nvidia-smi:"; nvidia-smi || echo "nvidia-smi not available"

# Chunk 10
srun "$PYTHON" -m scripts.index_wiki18 \
  --file-path "$BASE/data/wiki18/wiki-18.jsonl.gz" \
  --model "$BASE/models/contriever" \
  --batch-size 32 \
  --text-field contents \
  --skip 18000000 \
  --max-samples 2000000 \
  --output-dir "$BASE/index/wiki_index_chunk10"

echo "Finished: $(date)"



