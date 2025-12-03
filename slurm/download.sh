#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --job-name=download_wiki
#SBATCH --output=/vol/csedu-nobackup/course/I00041_informationretrieval/users/aditya/RAG-Uncertainty-Estimator/outputs/logs/download_%j.out

# 1) Go to your project folder on the cluster
PROJECT_ROOT=/vol/csedu-nobackup/course/I00041_informationretrieval/users/aditya/RAG-Uncertainty-Estimator
cd "$PROJECT_ROOT"

# 2) (Optional) set HF cache folder (not required, but good practice)
export HF_HOME="$PROJECT_ROOT/outputs/hf_cache"

# 3) Activate your virtual environment
source venv/bin/activate

# 4) Run the Python download script
python scripts/download_dataset.py --config configs/default.yaml
