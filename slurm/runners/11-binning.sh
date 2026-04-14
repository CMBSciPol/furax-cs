#!/bin/bash
# Post-clustering binning workflow:
#   1. Bin the 3 optimal runs (combined → ALL-GALACTIC) into patches .npy
#   2. Run kmeans-model with binned patches for each bin config
#   3. Snap all results
# RUN_LOCALLY: true (local direct), false (sbatch), dryrun (print only)
RUN_LOCALLY=false

ACCOUNT="tkc@h100"
CONSTRAINT="h100"
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

RTOL=1e-18
ATOL=1e-18
COOLDOWN=50
MIN_STEPS=200
VERBOSE=true
[ "$VERBOSE" = true ] && verbose_arg="--verbose" || verbose_arg=""

SKY=c1d1s1
SOLVER="ADABK0"
RESULTS_DIR="RESULTS/KMEANS_C1D1S1"
OUTPUT_DIR="RESULTS/BINNING_C1D1S1"

# 3 optimal runs (same as section_45.sh)
RUNS=(-r 'BD7000_TD500_BS500_GAL020' 'BD10000_TD500_BS150_GAL040' 'BD10000_TD2500_BS300_GAL060')


# =============================================================================
# Phase 1 & 2: For each bin config, bin then run kmeans
# =============================================================================

job_ids=()

for BIN in 10 100 1000; do
    BIN_DIR="$OUTPUT_DIR/BINNING/bin_${BIN}"
    NAME="binned_COMBINED_bin${BIN}"

    # Phase 1: Bin combined 3 runs → single patches set covering ALL-GALACTIC
    bin_jid=$(submit_job "BIN_${BIN}" "" BIN_COMBINED \
        r_analysis bin \
        "${RUNS[@]}" \
        -ird "$RESULTS_DIR" \
        -n 64 -i LiteBIRD \
        --bin-bd $BIN --bin-td $BIN --bin-bs $BIN \
        -o "$BIN_DIR")

    if [ -n "$bin_jid" ] && [ "$bin_jid" != "0" ]; then
        BIN_DEP="--dependency=afterok:$bin_jid"
    else
        BIN_DEP=""
    fi

    # Phase 2: Run kmeans with combined binned patches + ALL-GALACTIC mask
    if [ ! -f "$OUTPUT_DIR/KMEANS/$NAME/best_params.npz" ]; then
        jid=$(submit_job "KM_BIN${BIN}" "$BIN_DEP" KMEANS_BINNED \
            kmeans-model -n 64 -ns 40 -nr 1.0 \
            -c "$BIN_DIR/patches_beta_dust.npy" \
               "$BIN_DIR/patches_temp_dust.npy" \
               "$BIN_DIR/patches_beta_pl.npy" \
            -tag $SKY -m ALL-GALACTIC -i LiteBIRD \
            -s $SOLVER -mi 2000 \
            --rtol $RTOL --atol $ATOL \
            --cooldown $COOLDOWN --min-steps $MIN_STEPS $verbose_arg \
            --name $NAME -o "$OUTPUT_DIR/KMEANS")
        job_ids+=("$jid")
    else
        echo "Skipping $NAME (already done)"
    fi
done

# =============================================================================
# Phase 3: Snap all binned results
# =============================================================================

deps=$(IFS=:; echo "${job_ids[*]}")
if [ -n "$deps" ]; then
    DEP_ARGS="--dependency=afterok:$deps"
else
    DEP_ARGS=""
fi
submit_job "ANA_BINNING" "$DEP_ARGS" ANA_BINNING \
    r_analysis snap \
    -r 'binned' \
    -ird "$OUTPUT_DIR/KMEANS" \
    -o "$OUTPUT_DIR/SNAPSHOT/binning.parquet" \
    -mi 2000 -s optax_lbfgs --sky $SKY -n 64 -i LiteBIRD --no-images

echo "=== Done ==="
