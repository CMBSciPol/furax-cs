#!/bin/bash
# RUN_LOCALLY: true (local direct), false (sbatch), dryrun (print only)
RUN_LOCALLY=false

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

parquets_exist() {
    local dir="$1"
    [ -n "$(find "$dir" -name '*.parquet' 2>/dev/null | head -1)" ]
}

# =============================================================================
# Configuration
# =============================================================================

RTOL=1e-16
ATOL=1e-18

SOLVER="ADABK0"
OUTPUT_DIR="RESULTS/MULTIRES"
SKY="c1d0s0"
NS=100

# =============================================================================
# MultiRes MODELS (1)
# =============================================================================

job_ids=()

# Zone 1 Low galactic represented by GAL020
NAME="ptep1_${SKY}_BD64_TD0_BS2_LiteBIRD_GAL020"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(submit_job PTEP_GAL020_M1 "" PTEP \
        ptep-model -n 64 -ns $NS -nr 1.0 -ud 64 0 2 \
        -tag $SKY -m GAL020 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
        --rtol $RTOL --atol $ATOL \
        --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# Zone 2 Medium galactic represented by GAL040 - GAL020
NAME="ptep1_${SKY}_BD64_TD4_BS2_LiteBIRD_GAL040"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(submit_job PTEP_GAL040_M1 "" PTEP \
        ptep-model -n 64 -ns $NS -nr 1.0 -ud 64 4 2 \
        -tag $SKY -m GAL040 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
        --rtol $RTOL --atol $ATOL \
        --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# Zone 3 High galactic represented by GAL060 - GAL040
NAME="ptep1_${SKY}_BD64_TD8_BS4_LiteBIRD_GAL060"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(submit_job PTEP_GAL060_M1 "" PTEP \
        ptep-model -n 64 -ns $NS -nr 1.0 -ud 64 8 4 \
        -tag $SKY -m GAL060 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
        --rtol $RTOL --atol $ATOL \
        --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# =============================================================================
# Snapshot (skip if parquets already present — local mode optimization)
# =============================================================================

deps=$(IFS=:; echo "${job_ids[*]}")
if [ -n "$deps" ]; then
    DEP_ARGS="--dependency=afterok:$deps"
else
    DEP_ARGS=""
fi

SNAP_DIR="$OUTPUT_DIR/SNAPSHOT"
if [ "$RUN_LOCALLY" = true ] && parquets_exist "$SNAP_DIR"; then
    echo "Skipping snap step: parquets already present in $SNAP_DIR"
else
    submit_job CACHE_PTEP "$DEP_ARGS" SNAP \
        r_analysis snap -r ptep1 -ird $OUTPUT_DIR \
        -mi 2000 -n 64 -i LiteBIRD -s optax_lbfgs -o $SNAP_DIR/multires.parquet
fi
