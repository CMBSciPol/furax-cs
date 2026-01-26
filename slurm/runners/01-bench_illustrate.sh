SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --exclusive"
# =============================================================================
# Run Benchmarks
# =============================================================================
sbatch $SBATCH_ARGS --job-name=BENCH_CLUS-N-a100 $SLURM_SCRIPT BENCH bench-clusters -n 32 64 128 256 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --fgbuster-solver TNC  --noise 1.0
sbatch $SBATCH_ARGS --job-name=BENCH_CLUS-N-a100 $SLURM_SCRIPT BENCH bench-clusters -n 32 64 128 256 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --jax-solver active_set --noise 1.0
sbatch $SBATCH_ARGS --job-name=BENCH_CLUS-N-a100 $SLURM_SCRIPT BENCH bench-clusters -n 32 64 128 256 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --jax-solver scipy_tnc --noise 1.0
