#!/bin/bash
#SBATCH -J index-wiki18
#SBATCH -p csedu
#SBATCH -A csedui00041
#SBATCH -c 2                 # 2 CPU cores
#SBATCH --mem=8G             # 8 GB RAM
#SBATCH -t 02:00:00          # max runtime 2 hours
#SBATCH -o /scratch/$USER/proj/rag-ue/logs/%x.%j.out
#SBATCH -e /scratch/$USER/proj/rag-ue/logs/%x.%j.err

set -euo pipefail

# 1) activate your env
source /scratch/$USER/venvs/rag-ue/bin/activate

# 2) go to project
cd /scratch/$USER/proj/rag-ue

# 3) HF caches on scratch
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=/scratch/$USER/hf_datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export TQDM_MININTERVAL=5

# 4) run indexer on the whole wiki18
python -m scripts.index_wiki18 \
  --corpus_dir /scratch/$USER/data/wiki18 \
  --index_dir  /scratch/$USER/index/contriever \
  --model_dir  /scratch/$USER/models/contriever \
  --bs 32 --max_length 256 --max_files 0
