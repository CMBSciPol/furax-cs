#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --exclusive"

# =============================================================================
# Run Benchmarks
# =============================================================================

# 1. FGBuster (TNC)
sbatch $SBATCH_ARGS --job-name=BENCH_FGB $SLURM_SCRIPT BENCH bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --fgbuster-solver TNC --noise 1.0

# 2. ADABK5 (Top K = 0.5)
sbatch $SBATCH_ARGS --job-name=BENCH_ADA5 $SLURM_SCRIPT BENCH bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --jax-solver ADABK5 --noise 1.0

# 3. ADABK0 (Top K = 0.0)
sbatch $SBATCH_ARGS --job-name=BENCH_ADA0 $SLURM_SCRIPT BENCH bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --jax-solver ADABK0 --noise 1.0

# 4. Conditioned AdaBelief Active Set
sbatch $SBATCH_ARGS --job-name=BENCH_ADAC $SLURM_SCRIPT BENCH bench-clusters -n 64 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --jax-solver adabelief --precondition --noise 1.0