#!/bin/bash
# RUN_LOCALLY: true (local direct), false (sbatch), dryrun (print only)
RUN_LOCALLY=false

ACCOUNT="tkc@h100"
CONSTRAINT="h100"
GPUS_PER_NODE=1
CPUS_PER_NODE=10
TASKS_PER_NODE=1
NODES=1
QOS="qos_gpu_h100-t3"
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

RTOL=1e-18
ATOL=1e-18
COOLDOWN=50
MIN_STEPS=200
VERBOSE=true
[ "$VERBOSE" = true ] && verbose_arg="--verbose" || verbose_arg=""

SKY=c1d1s1
SOLVER="ADABK0"
SUFFIX="_BETTER_TERMINATE_FINAL"
OUTPUT_DIR="RESULTS/KMEANS_TRUE_CLUSTERS${SUFFIX}"

# =============================================================================
# K-Means runs with precomputed true-parameter clusters
# =============================================================================

job_ids=()
MASK="ALL-GALACTIC"
NAME="kmeans_${SKY}_BDtrue_TDtrue_BStrue_${MASK}"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(submit_job "KM_TRUE_${MASK}${SUFFIX}" "" KMEANS_TRUE${SUFFIX} \
        kmeans-model -n 64 -ns 40 -nr 1.0 \
        -c true true true \
        -tag ${SKY} -m $MASK -i LiteBIRD \
        -s $SOLVER -mi 2000 \
        --rtol $RTOL --atol $ATOL \
        --cooldown $COOLDOWN --min-steps $MIN_STEPS $verbose_arg \
        --name $NAME -o $OUTPUT_DIR)
    job_ids+=("$jid")
else
    echo "Skipping $NAME (already done)"
fi

# Snapshot step
deps=$(IFS=:; echo "${job_ids[*]}")
if [ -n "$deps" ]; then
    DEP_ARGS="--dependency=afterok:$deps"
else
    DEP_ARGS=""
fi
submit_job "ANA_TRUE${SUFFIX}" "$DEP_ARGS" ANA_TRUE${SUFFIX} \
    r_analysis snap -r "kmeans_${SKY}_BDtrue_TDtrue_BStrue" -ird $OUTPUT_DIR \
    -o $OUTPUT_DIR/SNAPSHOT/kmeans_c1d1s1_true.parquet \
    -mi 2000 -s optax_lbfgs -n 64 -i LiteBIRD
