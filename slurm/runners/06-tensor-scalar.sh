#!/bin/bash
# RUN_LOCALLY: true (local direct), false (sbatch), dryrun (print only)
#!/bin/bash
# RUN_LOCALLY: true (local direct), false (sbatch), dryrun (print only)
RUN_LOCALLY=dryrun

ACCOUNT="rzt@v100"
CONSTRAINT="v100-32g"
GPUS_PER_NODE=1
CPUS_PER_NODE=10
TASKS_PER_NODE=1
NODES=1
QOS="qos_gpu-t3"
TIME_LIMIT="05:00:00"
CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT \
    --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK \
    --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE"
[ -n "$QOS" ] && BASE_SBATCH_ARGS="$BASE_SBATCH_ARGS --qos=$QOS"

if [ "$RUN_LOCALLY" = false ] && [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."; exit 1
fi
mkdir -p SLURM_LOGS

dry_run_submit() {
    local job_name="$1"; local dep="${2:-}"; shift 2
    echo "======================================================="
    echo "Submitting job $job_name"
    echo "  ACCOUNT:    $ACCOUNT"
    echo "  CONSTRAINT: $CONSTRAINT"
    echo "  TIME_LIMIT: $TIME_LIMIT"
    echo "  GPU:        $GPUS_PER_NODE"
    echo "  CPU:        $CPUS_PER_TASK"
    echo "  NODES:      $NODES"
    echo "  QOS:        ${QOS:-none}"
    echo "  DEP:        ${dep:-none}"
    echo "  CMD:        $*"
    echo 0
}

submit_job() {
    local job_name="$1"; local dep_args="${2:-}"; local run_cat="$3"; shift 3
    if   [ "$RUN_LOCALLY" = true ];   then "$@"; echo 0
    elif [ "$RUN_LOCALLY" = dryrun ]; then dry_run_submit "$job_name" "$dep_args" "$@"
    else
        sbatch $BASE_SBATCH_ARGS --job-name="$job_name" $dep_args \
            --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err \
            "$SLURM_SCRIPT" "$run_cat" "$@" | awk '{print $NF}'
    fi
}

# =============================================================================
# Configuration
# =============================================================================

RTOL=1e-16
ATOL=1e-18

job_ids=()
OUTPUT_DIR="RESULTS/TENSOR_TO_SCALAR_34"

echo "=== Running Tensor-to-Scalar Runs ==="

NS=40
NR=0.3
MASK="ALL"
SOLVER="ADABK2"
MAX_ITER=2000
INSTRUMENT="LiteBIRD"
NSIDE=64

run_kmeans() {
    local TAG=$1
    local B_DUST=$2
    local T_DUST=$3
    local B_SYNC=$4

    local JOB_NAME="KM_${TAG}_${B_DUST}"
    local NAME="kmeans_${TAG}_BD${B_DUST}_TD${T_DUST}_BS${B_SYNC}_${MASK}"

    if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
        echo "Submitting ${TAG} with ${B_DUST} ${T_DUST} ${B_SYNC}..."
        jid=$(submit_job "$JOB_NAME" "" KMEANS \
            kmeans-model -n $NSIDE -ns $NS -nr $NR \
            -pc $B_DUST $T_DUST $B_SYNC \
            -tag $TAG -m $MASK -i $INSTRUMENT \
            -s $SOLVER -mi $MAX_ITER \
            --rtol $RTOL --atol $ATOL \
            --name $NAME -o $OUTPUT_DIR)
        job_ids+=("$jid")
    else
        echo "Skipping $NAME (already done)"
    fi
}

# 1. cr4d1s1 10 10 10
run_kmeans "cr4d1s1" 10 10 10

# 2. c1d1s1 10 10 10
run_kmeans "c1d1s1" 10 10 10

# 3. cr4d1s1 30000 1500 1500
run_kmeans "cr4d1s1" 30000 1500 1500

# 4. c1d1s1 30000 1500 1500
run_kmeans "c1d1s1" 30000 1500 1500

# =============================================================================
# Analysis
# =============================================================================

deps=$(IFS=:; echo "${job_ids[*]}")

if [ -n "$deps" ]; then
    echo "Submitting analysis jobs..."

    # Analysis for cr4d1s1
    submit_job ANA_CR4 "--dependency=afterany:$deps" ANA \
        r_analysis snap -r "kmeans_cr4d1s1" -ird $OUTPUT_DIR \
        -mi $MAX_ITER -s optax_lbfgs -n $NSIDE -i $INSTRUMENT -o $OUTPUT_DIR/SNAP/tensor_to_scalar_cr4d1s1.parquet

    # Analysis for c1d1s1
    submit_job ANA_C1 "--dependency=afterany:$deps" ANA \
        r_analysis snap -r "kmeans_c1d1s1" -ird $OUTPUT_DIR \
        -mi $MAX_ITER -s optax_lbfgs -n $NSIDE -i $INSTRUMENT -o $OUTPUT_DIR/SNAP/tensor_to_scalar_c1d1s1.parquet
else
    echo "No new jobs submitted. Skipping analysis."
fi
