#!/bin/bash
#SBATCH -J wiki18-full
#SBATCH -p csedu
#SBATCH -A csedui00041
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

echo "[START] Job running on: \$(hostname)"
echo "[TIME] \$(date)"

# 1) activate venv
source /scratch/\$USER/venvs/rag-ue/bin/activate

# 2) go to project
cd /scratch/\$USER/proj/rag-ue

# 3) HF and thread env
export HF_HOME=/scratch/\$USER/hf
export HF_DATASETS_CACHE=/scratch/\$USER/hf_datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

# 4) run full indexing
python -m scripts.index_wiki18 \
  --file-path /scratch/\$USER/data/wiki18/wiki-18.jsonl.gz \
  --model /scratch/\$USER/models/contriever \
  --batch-size 8 \
  --text-field contents \
  --output-dir /scratch/\$USER/index/wiki_index_full

echo "[END] Finished at \$(date)"
