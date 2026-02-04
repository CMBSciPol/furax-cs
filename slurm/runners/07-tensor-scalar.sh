#!/bin/bash
# Tensor-to-scalar ratio runs (Redone)

# Collect all job IDs here
job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --time=05:00:00 --parsable"
OUTPUT_DIR="RESULTS/RUNS/TENSOR_TO_SCALAR_34"

echo "=== Running Tensor-to-Scalar Runs (Redo) ==="

# Common Parameters
NS=10
NR=0.2
MASK="ALL"
SOLVER="ADABK0"
MAX_ITER=2000
INSTRUMENT="LiteBIRD"
NSIDE=64

submit_job() {
    local TAG=$1
    local B_DUST=$2
    local T_DUST=$3
    local B_SYNC=$4
    
    local JOB_NAME="KM_${TAG}_${B_DUST}"
    # Construct name to match pattern
    local NAME="kmeans_${TAG}_BD${B_DUST}_TD${T_DUST}_BS${B_SYNC}_${MASK}"
    
    if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
        echo "Submitting ${TAG} with ${B_DUST} ${T_DUST} ${B_SYNC}..."
        
        jid=$(sbatch $BATCH_PARAMS --job-name=$JOB_NAME \
            $SLURM_SCRIPT $OUTPUT_DIR \
            kmeans-model -n $NSIDE -ns $NS -nr $NR \
            -pc $B_DUST $T_DUST $B_SYNC \
            -tag $TAG -m $MASK -i $INSTRUMENT \
            -s $SOLVER -mi $MAX_ITER \
            --name $NAME -o $OUTPUT_DIR)
            
        job_ids+=("$jid")
    else
        echo "Skipping $NAME (already done)"
    fi
}

# 1. cr4d0s0 10 10 10
submit_job "cr4d0s0" 10 10 10

# 2. c1d0s0 10 10 10
submit_job "c1d0s0" 10 10 10

# 3. cr4d0s0 30000 1500 1500
submit_job "cr4d0s0" 30000 1500 1500

# 4. c1d0s0 30000 1500 1500
submit_job "c1d0s0" 30000 1500 1500

# Dependencies for r_analysis
deps=$(IFS=:; echo "${job_ids[*]}")

if [ -n "$deps" ]; then
    echo "Submitting analysis jobs..."

    # Analysis for cr4d0s0
    sbatch --dependency=afterany:$deps \
        $BATCH_PARAMS \
        --job-name=ANA_CR4 \
        $SLURM_SCRIPT $OUTPUT_DIR \
        r_analysis snap -r "kmeans_cr4d0s0_*" -ird $OUTPUT_DIR \
        -mi $MAX_ITER -s optax_lbfgs -n $NSIDE -i $INSTRUMENT -o SNAP

    # Analysis for c1d0s0
    sbatch --dependency=afterany:$deps \
        $BATCH_PARAMS \
        --job-name=ANA_C1 \
        $SLURM_SCRIPT $OUTPUT_DIR \
        r_analysis snap -r "kmeans_c1d0s0_*" -ird $OUTPUT_DIR \
        -mi $MAX_ITER -s optax_lbfgs -n $NSIDE -i $INSTRUMENT -o SNAP
else
    echo "No new jobs submitted. Skipping analysis."
fi