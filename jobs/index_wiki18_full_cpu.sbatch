#!/bin/bash
#SBATCH --job-name=wiki18-full
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# If you want a GPU, uncomment the next line:
#SBATC H --gres=gpu:1
#SBATCH --output=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv/wiki18-full.%j.out
#SBATCH --error=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv/wiki18-full.%j.err

set -euo pipefail
set -x

echo "[START] Job running on: $(hostname)"
echo "[TIME] $(date)"
echo "User: $(whoami)"
echo "PWD at start: $(pwd)"

cd /scratch/kyv/proj/rag-ue
echo "PWD after cd: $(pwd)"
ls

# HF caches (you can also move these to /vol/csedu-nobackup if /scratch becomes a problem)
export HF_HOME=/scratch/kyv/hf
export HF_DATASETS_CACHE=/scratch/kyv/hf_datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

/scratch/kyv/venvs/rag-ue/bin/python -V

/scratch/kyv/venvs/rag-ue/bin/python -m scripts.index_wiki18 \
  --file-path /scratch/kyv/data/wiki18/wiki-18.jsonl.gz \
  --model /scratch/kyv/models/contriever \
  --batch-size 8 \
  --text-field contents \
  --output-dir /scratch/kyv/index/wiki_index_full

echo "[END] Finished at $(date)"
