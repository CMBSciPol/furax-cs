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

SKY=c1d0s0
SOLVER="ADABK0"
OUTPUT_DIR="RESULTS/KMEANS_C1D0S0"

CONFIGS=(
    "4000 10 0 3"    # Case 1: B_DUST=4000, T_DUST=10, Vary B_SYNC
    "4000 0 10 2"    # Case 2: B_DUST=4000, B_SYNC=10, Vary T_DUST
    "10000 500 0 3"  # Case 3: B_DUST=10000, T_DUST=500, Vary B_SYNC
    "10000 0 500 2"  # Case 4: B_DUST=10000, B_SYNC=500, Vary T_DUST
    "0 500 500 1"    # Case 5: T_DUST=500, B_SYNC=500, Vary B_DUST
    "10000 3500 0 3" # Case 6: B_DUST=10000, T_DUST=3500, Vary B_SYNC
    "10000 0 300 2"  # Case 7: B_DUST=10000, B_SYNC=300, Vary T_DUST
)

RANGE_LOW="1 10 20 30 40 50 60 70 80 90"
RANGE_HIGH="1 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000"

# =============================================================================
# K-Means runs
# =============================================================================

job_ids=()

for config in "${CONFIGS[@]}"; do
    read -r B_DUST_BASE T_DUST_BASE B_SYNC_BASE VARY_IDX <<< "$config"

    if [ "$VARY_IDX" -eq 1 ]; then
        VARY_NAME="B_DUST"
        RANGE=$RANGE_HIGH
        OUTPUT_BASE="BDXXX_TD${T_DUST_BASE}_BS${B_SYNC_BASE}"
    elif [ "$VARY_IDX" -eq 2 ]; then
        VARY_NAME="T_DUST"
        if [ "$B_DUST_BASE" -eq 4000 ]; then RANGE=$RANGE_LOW; else RANGE=$RANGE_HIGH; fi
        OUTPUT_BASE="BD${B_DUST_BASE}_TDXXX_BS${B_SYNC_BASE}"
    elif [ "$VARY_IDX" -eq 3 ]; then
        VARY_NAME="B_SYNC"
        if [ "$B_DUST_BASE" -eq 4000 ]; then RANGE=$RANGE_LOW; else RANGE=$RANGE_HIGH; fi
        OUTPUT_BASE="BD${B_DUST_BASE}_TD${T_DUST_BASE}_BSXXX"
    fi

    RUN_OUTPUT_DIR="$OUTPUT_DIR/$OUTPUT_BASE"
    current_job_ids=()

    echo "=== Running Configuration: $OUTPUT_BASE (Varying $VARY_NAME) ==="

    for VAL in $RANGE; do
        B_DUST=$B_DUST_BASE
        T_DUST=$T_DUST_BASE
        B_SYNC=$B_SYNC_BASE

        if [ "$VARY_IDX" -eq 1 ]; then B_DUST=$VAL; fi
        if [ "$VARY_IDX" -eq 2 ]; then T_DUST=$VAL; fi
        if [ "$VARY_IDX" -eq 3 ]; then B_SYNC=$VAL; fi

        JOB_NAME="KM_${VARY_NAME}_${VAL}"

        for MASK in ALL-GALACTIC; do
            NAME="kmeans_${SKY}_BD${B_DUST}_TD${T_DUST}_BS${B_SYNC}_${MASK}"
            if [ ! -f "$RUN_OUTPUT_DIR/$NAME/best_params.npz" ]; then
                jid=$(submit_job "${JOB_NAME}_${MASK}" "" KMEANS \
                    kmeans-model -n 64 -ns 10 -nr 1.0 \
                    -pc $B_DUST $T_DUST $B_SYNC \
                    -tag ${SKY} -m $MASK -i LiteBIRD \
                    -s $SOLVER -mi 2000 \
                    --rtol $RTOL --atol $ATOL \
                    --name $NAME -o $RUN_OUTPUT_DIR)
                current_job_ids+=("$jid")
            else
                echo "Skipping $NAME (already done)"
            fi
        done
    done

    deps=$(IFS=:; echo "${current_job_ids[*]}")
    if [ "$VARY_IDX" -eq 1 ]; then REGEX="kmeans_${SKY}_BD(\d+)_TD${T_DUST_BASE}_BS${B_SYNC_BASE}"; fi
    if [ "$VARY_IDX" -eq 2 ]; then REGEX="kmeans_${SKY}_BD${B_DUST_BASE}_TD(\d+)_BS${B_SYNC_BASE}"; fi
    if [ "$VARY_IDX" -eq 3 ]; then REGEX="kmeans_${SKY}_BD${B_DUST_BASE}_TD${T_DUST_BASE}_BS(\d+)"; fi

    if [ -n "$deps" ]; then
        DEP_ARGS="--dependency=afterok:$deps"
    else
        DEP_ARGS=""
    fi
    submit_job "ANA_${OUTPUT_BASE}" "$DEP_ARGS" ANA \
        r_analysis snap -r "$REGEX" -ird $RUN_OUTPUT_DIR \
        -o $RUN_OUTPUT_DIR/SNAPSHOT \
        -mi 2000 -s optax_lbfgs -n 64 -i LiteBIRD

    job_ids+=("${current_job_ids[@]}")
done
