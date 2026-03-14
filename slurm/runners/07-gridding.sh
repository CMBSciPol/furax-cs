#!/bin/bash
# RUN_LOCALLY: true (local direct), false (sbatch), dryrun (print only)
RUN_LOCALLY=false

ACCOUNT="nih@h100"
CONSTRAINT="h100"
GPUS_PER_NODE=8
CPUS_PER_NODE=80
TASKS_PER_NODE=8
NODES=4
QOS=""
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
OUTPUT_DIR="RESULTS/GRID_SEARCH"

# =============================================================================
# Grid search runs on galactic mask zones
# =============================================================================

# Zone 1 mask of GAL020
jid=$(submit_job GAL020_GRID "" GRIDDING \
    distributed-gridding -n 64 -ns 100 -nr 1.0 \
    -tag c1d1s1 -m GAL020 -i LiteBIRD -cond \
    -ss SEARCH_SPACE.yml -s active_set -mi 1000 \
    --rtol $RTOL --atol $ATOL \
    -o $OUTPUT_DIR)
job_ids+=($jid)

# Zone 2 mask of GAL040 - GAL020
jid=$(submit_job GAL040_GRID "" GRIDDING \
    distributed-gridding -n 64 -ns 100 -nr 1.0 \
    -tag c1d1s1 -m GAL040 -i LiteBIRD -cond \
    -ss SEARCH_SPACE.yml -s active_set -mi 1000 \
    --rtol $RTOL --atol $ATOL \
    -o $OUTPUT_DIR)
job_ids+=($jid)

# Zone 3 mask of GAL060 - GAL040
jid=$(submit_job GAL060_GRID "" GRIDDING \
    distributed-gridding -n 64 -ns 100 -nr 1.0 \
    -tag c1d1s1 -m GAL060 -i LiteBIRD -cond \
    -ss SEARCH_SPACE.yml -s active_set -mi 1000 \
    --rtol $RTOL --atol $ATOL \
    -o $OUTPUT_DIR)
job_ids+=($jid)

# Zone 4 mask of GALACTIC
jid=$(submit_job GALACTIC_GRID "" GRIDDING \
    distributed-gridding -n 64 -ns 100 -nr 1.0 \
    -tag c1d1s1 -m GALACTIC -i LiteBIRD -cond \
    -ss SEARCH_SPACE.yml -s active_set -mi 1000 \
    --rtol $RTOL --atol $ATOL \
    -o $OUTPUT_DIR)
job_ids+=($jid)

# =============================================================================
# Snapshot
# =============================================================================

deps=$(IFS=:; echo "${job_ids[*]}")
submit_job PTEP_SNAP "--dependency=afterok:$deps" SNAP \
    r_analysis snap -n 64 -i LiteBIRD -ird RESULTS/MULTIRES \
    -r ptep -o $OUTPUT_DIR/multires.parquet -s active_set -mi 1000
