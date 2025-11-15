#!/bin/bash
#SBATCH --job-name=wiki18-full
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv/wiki18-full.%j.out
#SBATCH --error=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv/wiki18-full.%j.err

set -euo pipefail
set -x

BASE=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv

echo "[START] Job running on: $(hostname)"
echo "[TIME] $(date)"
echo "User: $(whoami)"
echo "PWD at start: $(pwd)"

# 1) Go to your project on /vol (shared across nodes)
cd "$BASE/rag-ue"
echo "PWD after cd: $(pwd)"
ls

# 2) HuggingFace caches on /vol (avoid home quota)
/usr/bin/mkdir -p "$BASE/hf" "$BASE/hf/datasets" "$BASE/hf/transformers" "$BASE/hf/hub" || true
export HF_HOME="$BASE/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# 3) Activate the /vol venv and run python from it
source venv/bin/activate

python -V

# 4) Call your indexing script with real arguments
python -m scripts.index_wiki18 \
  --file-path "$BASE/data/wiki18/wiki-18.jsonl.gz" \
  --model "$BASE/models/contriever" \
  --batch-size 8 \
  --text-field contents \
  --output-dir "$BASE/index/wiki_index_full"

echo "[END] Finished at $(date)"
