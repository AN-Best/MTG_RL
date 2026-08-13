#!/bin/bash
#SBATCH --account=def-rzheng_gpu          # GPU allocation account
#SBATCH --gpus-per-node=a100:1            # 1x A100 (40 GB VRAM) — Narval standard
#SBATCH --cpus-per-task=6                 # 6 CPU cores (env sim + data loading)
#SBATCH --mem=48G                         # 48 GB RAM (2048 envs + ref motion buffer)
#SBATCH --time=23:59:00                   # 24h walltime limit
#SBATCH --output=logs/biped_%j.out        # Stdout → logs/biped_<jobid>.out
#SBATCH --error=logs/biped_%j.err         # Stderr → logs/biped_<jobid>.err
#SBATCH --job-name=biped_diffphys         # Job name visible in squeue

# -----------------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------------
mkdir -p logs

echo "============================================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $SLURMD_NODENAME"
echo "  GPUs:        $CUDA_VISIBLE_DEVICES"
echo "  Start:       $(date)"
echo "  Working dir: $(pwd)"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Load Cluster Modules
# -----------------------------------------------------------------------------
echo "Loading modules..."
module load StdEnv/2023
module load intel/2023.2.1
module load python/3.10
module load scipy-stack
module load cuda/11.8

# -----------------------------------------------------------------------------
# 2. Activate Virtual Environment
# -----------------------------------------------------------------------------
# One-time setup on the login node:
#   module load StdEnv/2023 intel/2023.2.1 python/3.10 scipy-stack cuda/11.8
#   virtualenv --no-download ~/env_biped
#   source ~/env_biped/bin/activate
#   pip install --no-index torch torchvision
#   pip install sympy gymnasium skrl
#
ENV_PATH="$HOME/env_biped"
if [ -d "$ENV_PATH" ]; then
    echo "Activating virtualenv: $ENV_PATH"
    source "$ENV_PATH/bin/activate"
else
    echo "ERROR: Virtual environment not found at $ENV_PATH"
    echo "Please create it on the login node first (see comments above)."
    exit 1
fi

# Verify GPU is visible
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found!'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || exit 1

# -----------------------------------------------------------------------------
# 3. Pre-download Reference Motion (avoids timeout during compute job)
# -----------------------------------------------------------------------------
MOT_FILE="subject01_walk1_ik.mot"
if [ ! -f "$MOT_FILE" ]; then
    echo "Downloading reference motion..."
    wget -q -O "$MOT_FILE" \
        "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Pipelines/Gait2392_Simbody/subject01_walk1_ik.mot" \
        || { echo "WARNING: Download failed, fallback generator will be used."; rm -f "$MOT_FILE"; }
fi

# -----------------------------------------------------------------------------
# 4. Run Training
# -----------------------------------------------------------------------------
echo "Starting Differentiable Physics training..."
srun python -u Biped_Train_DiffPhys.py

EXIT_CODE=$?
echo "============================================================"
echo "  Job finished: $(date)"
echo "  Exit code:    $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
