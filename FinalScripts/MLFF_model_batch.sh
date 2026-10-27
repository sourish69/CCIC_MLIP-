#!/bin/bash
#SBATCH --job-name=mlff4
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # one task = one independent model
#SBATCH --cpus-per-task=16         # cores per model
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
##SBATCH --partition=cpu           # <-- set for your cluster
##SBATCH --account=myproject       # <-- set for your cluster

# Submit with ./submit.sh, which passes CONFIG, TRAIN_SCRIPT and RESTART (0/1).
set -euo pipefail

CONFIG="${CONFIG:-config.json}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SLURM_SUBMIT_DIR/train_mlff.py}"
RESTART="${RESTART:-0}"

# ---- activate your environment (edit) --------------------------------------
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate deepmd
# -----------------------------------------------------------------------------
command -v dp >/dev/null || { echo "'dp' not on PATH - activate the DeePMD env" >&2; exit 1; }

# Each model uses all the cores of its own task
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_INTRA_OP_PARALLELISM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_INTER_OP_PARALLELISM_THREADS=2

# usage: run <n_tasks> <train_mlff.py args...>
run() {
    local ntasks=$1; shift
    srun --ntasks="$ntasks" --cpus-per-task="$SLURM_CPUS_PER_TASK" --exact --cpu-bind=cores \
         python "$TRAIN_SCRIPT" --config "$CONFIG" "$@"
}

echo "Job $SLURM_JOB_ID on $(hostname): 4 models x $SLURM_CPUS_PER_TASK cores, restart=$RESTART"

if [ "$RESTART" = "1" ]; then
    run 4 --stage train --restart
else
    # Convert the ASE DBs once (shared by all models), then train 4 in parallel.
    # Task k trains model_k (index from $SLURM_PROCID, seed = seed_base + k).
    run 1 --stage prepare
    run 4 --stage train
fi
echo "All models finished."
