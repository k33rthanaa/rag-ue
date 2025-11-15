cd /vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv/rag-ue

cat > jobs/index_wiki18_full.sbatch << 'EOF'
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

cd "$BASE/rag-ue"
echo "PWD after cd: $(pwd)"
ls

# HuggingFace caches on /vol
export HF_HOME="$BASE/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Activate the /vol venv exactly like in your shell
source venv/bin/activate

python -V

python -m scripts.index_wiki18 \
  --file-path "$BASE/data/wiki18/wiki-18.jsonl.gz" \
  --model "$BASE/models/contriever" \
  --batch-size 8 \
  --text-field contents \
  --output-dir "$BASE/index/wiki_index_full"

echo "[END] Finished at $(date)"
EOF
