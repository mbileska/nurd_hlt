#!/bin/bash
#SBATCH --job-name=nurd_eval_dual
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Compatibility entrypoint. Every Slurm evaluation now runs both the strict
# held-out protocol and origin/main's legacy QCD protocol.
set -euo pipefail
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
exec bash "$CODE_DIR/slurm/submit_eval_latest.sbatch"
