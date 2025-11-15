#!/bin/bash
#SBATCH --job-name=wiki18_index
#SBATCH --output=logs/wiki18_index_%j.out
#SBATCH --error=logs/wiki18_index_%j.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p <YOUR_PARTITION_NAME>      # ← replace this!
# If you ever want a GPU, add something like:
# #SBATCH --gres=gpu:1

BASE=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv

cd "$BASE/rag-ue" || { echo "cd failed"; exit 1; }

source venv/bin/activate

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
echo "Python: $(which python)"
python --version

srun python -m scripts.index_wiki18 \
  --file-path "$BASE/data/wiki18/wiki-18.jsonl.gz" \
  --model "$BASE/models/contriever" \
  --batch-size 2 \
  --max-samples 1000 \
  --text-field contents \
  --output-dir "$BASE/index/wiki_index_full_debug"

echo "Finished: $(date)"
