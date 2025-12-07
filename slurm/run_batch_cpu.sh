#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --time=10:0:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --job-name=rag_batch_cpu
#SBATCH --output=logs/rag_batch_%j.out
#SBATCH --error=logs/rag_batch_%j.err

# Create logs directory
mkdir -p logs

# Activate environment
cd /vol/csedu-nobackup/course/I00041_informationretrieval/users/aditya/RAG-Uncertainty-Estimator
source venv/bin/activate

# Run batch processing
python scripts/batch_rag.py

echo "✅ Batch processing complete!"
