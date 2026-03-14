#!/bin/bash
# RUN_LOCALLY: true (local direct), false (sbatch), dryrun (print only)
RUN_LOCALLY=false

ACCOUNT="nih@h100"
CONSTRAINT="h100"
GPUS_PER_NODE=1
CPUS_PER_NODE=10
TASKS_PER_NODE=1
NODES=1
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

SKY=bd100_td15_bs5
MASK="ALL-GALACTIC"
OUTPUT_DIR="RESULTS/NOISE_MODEL"

# =============================================================================
# Noise model: grid over BD patches (truth: BD=100, TD=15, BS=5)
# =============================================================================

job_ids=()

for BD in $(seq 10 10 300); do
    NAME="kmeans_${SKY}_BD${BD}_TD15_BS5"
    if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
        jid=$(submit_job "KM_BD${BD}" "" KMEANS \
            kmeans-model -n 64 -ns 10 -nr 1.0 \
            -pc $BD 15 5 \
            -tag ${SKY} -m $MASK -i LiteBIRD \
            -s optax_lbfgs -mi 2000 \
            --rtol $RTOL --atol $ATOL \
            --name $NAME -o $OUTPUT_DIR)
        job_ids+=("$jid")
    else
        echo "Skipping $NAME (already done)"
    fi
done

# =============================================================================
# Snapshot (depends on all kmeans jobs)
# =============================================================================

deps=$(IFS=:; echo "${job_ids[*]}")
DEP_ARGS=""
[ -n "$deps" ] && DEP_ARGS="--dependency=afterok:$deps"

REGEX="kmeans_bd100_td15_bs5_BD(\d+)_TD15_BS5"
submit_job "ANA_NOISE" "$DEP_ARGS" ANA \
    r_analysis snap -r "$REGEX" -ird $OUTPUT_DIR \
    -o $OUTPUT_DIR/SNAPSHOT/noise_model.parquet \
    -mi 2000 -s optax_lbfgs -n 64 -i LiteBIRD
