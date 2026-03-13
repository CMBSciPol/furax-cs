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

OUTPUT_DIR="RESULTS/MINIMIZE/VALIDATE_FURAX"
mkdir -p $OUTPUT_DIR

job_ids=()
noise="1.0"
sky="c1d0s0"
tag=$sky
mask="ALL-GALACTIC"

COMMON_ARGS="-n 64 -ns 20 -nr $noise -pc 140 140 140 -tag $tag -m $mask -i LiteBIRD -sp 1.54 20.0 -3.0 -mi 2000 -o $OUTPUT_DIR --rtol $RTOL --atol $ATOL"

# =============================================================================
# FURAX Runs (L-BFGS, SciPy TNC, AdaBelief, Active Set)
# =============================================================================

# 1. Optax L-BFGS zoom + cond
ls_lbfgs="zoom"
flag="-cond"
cname="cond"
run_name_lbfgs="kmeans_lbfgs_ls${ls_lbfgs}_${cname}"

if [ ! -f "$OUTPUT_DIR/$run_name_lbfgs/best_params.npz" ]; then
    echo "Submitting $run_name_lbfgs"
    jid=$(submit_job FX_LB "" FURAX \
        kmeans-model $COMMON_ARGS \
        -s optax_lbfgs -ls $ls_lbfgs $flag --name $run_name_lbfgs)
    job_ids+=("$jid")
else
    echo "Skipping $run_name_lbfgs (already done)"
fi

# 2. SciPy TNC
run_name_tnc="kmeans_scipy_tnc"

if [ ! -f "$OUTPUT_DIR/$run_name_tnc/best_params.npz" ]; then
    echo "Submitting $run_name_tnc"
    jid=$(submit_job FX_TNC "" FURAX \
        kmeans-model $COMMON_ARGS \
        -s scipy_tnc --name $run_name_tnc)
    job_ids+=("$jid")
else
    echo "Skipping $run_name_tnc (already done)"
fi

# 3. AdaBelief (Conditioned + Zoom)
run_name_ada="kmeans_adabelief_lszoom_cond"

if [ ! -f "$OUTPUT_DIR/$run_name_ada/best_params.npz" ]; then
    echo "Submitting $run_name_ada"
    jid=$(submit_job FX_ADA "" FURAX \
        kmeans-model $COMMON_ARGS \
        -s adabelief -ls zoom -cond --name $run_name_ada)
    job_ids+=("$jid")
else
    echo "Skipping $run_name_ada (already done)"
fi

# 4. Active Set AdaBelief (Zoom + k values)
ks=("0" "1" "2" "4" "6" "8")
AS_ADA_PATS=""

for k in "${ks[@]}"; do
    run_name="kmeans_topk${k}_active_set_adabelief_lszoom"

    if [ ! -f "$OUTPUT_DIR/$run_name/best_params.npz" ]; then
        echo "Submitting $run_name"
        jid=$(submit_job "FX_AS_ADA_${k}" "" FURAX \
            kmeans-model $COMMON_ARGS \
            -s ADABK$k --name $run_name)
        job_ids+=("$jid")
    else
        echo "Skipping $run_name (already done)"
    fi

    AS_ADA_PATS="$AS_ADA_PATS $run_name"
done

AS_ADA_REG="kmeans_ADABK(\d+)_active_set_adabelief_lszoom"

# =============================================================================
# Analysis
# =============================================================================

deps=$(IFS=:; echo "${job_ids[*]}")
if [ -z "$deps" ]; then
    echo "No new jobs submitted. Skipping analysis."
else
    echo "Submitting analysis jobs depending on: $deps"

    RUN_PATS="$run_name_lbfgs $run_name_tnc $run_name_ada $AS_ADA_REG"

    # Validate
    submit_job FX_val_FURAX "--dependency=afterany:$deps" ANA \
        r_analysis validate \
        -r $RUN_PATS \
        -ird $OUTPUT_DIR \
        --scales 1e-4 1e-5

    # Snapshot
    submit_job FX_snap_FURAX "--dependency=afterany:$deps" SNAP \
        r_analysis snap \
        -r $RUN_PATS \
        -ird $OUTPUT_DIR \
        -s optax_lbfgs \
        -o $OUTPUT_DIR/SNAPSHOT_FURAX \
        -mi 2000 \
        -n 64 \
        -i LiteBIRD
fi
