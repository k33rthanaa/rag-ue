#!/bin/bash
#SBATCH --job-name=wiki18_full_index
#SBATCH --output=logs/wiki18_full_%j.out
#SBATCH --error=logs/wiki18_full_%j.err
#SBATCH --time=1-00:00:00          # 1 day; csedu has infinite limit, so this is safe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p csedu                   # use the csedu partition

BASE=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv

cd "$BASE/rag-ue" || { echo "cd failed"; exit 1; }

source venv/bin/activate

export HF_HOME="$BASE/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
python --version

# FULL CORPUS: no --max-samples -> indexes everything in wiki-18.jsonl.gz
srun python -m scripts.index_wiki18 \
  --file-path "$BASE/data/wiki18/wiki-18.jsonl.gz" \
  --model "$BASE/models/contriever" \
  --batch-size 32 \
  --text-field contents \
  --output-dir "$BASE/index/wiki_index_full"

echo "Finished: $(date)"
