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
set -x   # echo commands so we see what runs

echo "[START] Job running on: $(hostname)"
echo "[TIME] $(date)"
echo "User: $(whoami)"
echo "Initial PWD: $(pwd)"

# 1) go to project
cd /scratch/$USER/proj/rag-ue
echo "PWD after cd: $(pwd)"
ls

# 2) HF and thread env
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=/scratch/$USER/hf_datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

# 3) show which python we will use
/scratch/$USER/venvs/rag-ue/bin/python -V

# 4) run full indexing
/scratch/$USER/venvs/rag-ue/bin/python -m scripts.index_wiki18 \
  --file-path /scratch/$USER/data/wiki18/wiki-18.jsonl.gz \
  --model /scratch/$USER/models/contriever \
  --batch-size 8 \
  --text-field contents \
  --output-dir /scratch/$USER/index/wiki_index_full

echo "[END] Finished at $(date)"
