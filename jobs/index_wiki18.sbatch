#!/bin/bash
#SBATCH --job-name=index_wiki
#SBATCH --output=logs/index_wiki_%j.out
#SBATCH --error=logs/index_wiki_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# If you actually want a GPU, uncomment and adjust:
# #SBATCH --gres=gpu:1

# === 0) Sanity: echo some basic info ===
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"

# === 1) Recreate your BASE ===
BASE=/vol/csedu-nobackup/course/I00041_informationretrieval/users/kyv

# === 2) cd to project, like in your working shell ===
cd "$BASE/rag-ue" || { echo "cd to $BASE/rag-ue failed"; exit 1; }

# If your cluster uses modules normally, load them here
# module load Python/3.10

# === 3) Activate your venv ===
source venv/bin/activate

# Debug: confirm which Python we're using
echo "Python path: $(which python)"
python --version

# === 4) Re-export your HF env vars (batch jobs don't inherit them) ===
export HF_HOME="$BASE/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Debug: show the relevant env vars
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# === 5) Actually run your script ===
# Use srun if that’s the convention on your cluster
srun python -m scripts.index_wiki18 \
  --file-path "$BASE/data/wiki18/wiki-18.jsonl.gz" \
  --model "$BASE/models/contriever" \
  --batch-size 2 \
  --max-samples 1000 \
  --text-field contents \
  --output-dir "$BASE/index/wiki_index_full_debug"

echo "Finished at: $(date)"
