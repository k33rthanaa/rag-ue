#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=build_shard
#SBATCH --output=/vol/csedu-nobackup/course/I00041_informationretrieval/users/aditya/RAG-Uncertainty-Estimator/outputs/logs/build_shard_%j.out

# 1) Read shard id from first argument
SHARD_ID=$1

if [ -z "$SHARD_ID" ]; then
  echo "Error: You must pass a shard id, e.g.:"
  echo "  sbatch slurm/build_shard.sh 0"
  exit 1
fi

echo "Running shard id: $SHARD_ID"

# 2) Go to your project folder
PROJECT_ROOT=/vol/csedu-nobackup/course/I00041_informationretrieval/users/aditya/RAG-Uncertainty-Estimator
cd "$PROJECT_ROOT"

# 3) (Optional) Set HuggingFace cache folder on shared disk
export HF_HOME="$PROJECT_ROOT/outputs/hf_cache"

# 4) Activate your virtualenv
source venv/bin/activate

# 5) Run the Python indexing script
python scripts/build_shard.py \
  --shard_id "$SHARD_ID" \
  --config configs/default.yaml
