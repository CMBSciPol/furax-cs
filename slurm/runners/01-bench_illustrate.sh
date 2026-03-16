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
TIME_LIMIT="08:00:00"
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
# Run Benchmarks  (bench-clusters uses --tol, not --rtol/--atol)
# =============================================================================

# 1. FGBuster (TNC)
submit_job BENCH_FGB "" BENCH bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --fgbuster-solver TNC --noise 1.0 --tol 1e-16 --n-sims 20  --max-iter 2000

# 2. ADABK5 (Top K = 0.5)
submit_job BENCH_ADA5 "" BENCH \
    bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
    --jax-solver ADABK5 --noise 1.0 --tol 1e-2 --n-sims 20  --max-iter 2000

# 3. ADABK0 (Top K = 0.0)
submit_job BENCH_ADA0 "" BENCH \
    bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
    --jax-solver ADABK0 --noise 1.0 --tol 1e-16 --n-sims 20  --max-iter 2000

# 4. Conditioned AdaBelief Active Set
submit_job BENCH_ADAC "" BENCH \
    bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
    --jax-solver adabelief --precondition --noise 1.0 --tol 1e-16 --n-sims 20  --max-iter 2000
