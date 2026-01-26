#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --parsable"
OUTPUT_DIR="RESULTS/MINIMIZE/VALIDATE_FG"
mkdir -p $OUTPUT_DIR

job_ids=()
noise="1.0"
sky="c1d0s0"
tag=$sky
mask="ALL-GALACTIC" # Assuming ALL-GALACTIC for FGBuster too, or revert to specific masks if needed

COMMON_ARGS="-n 64 -ns 20 -nr $noise -pc 140 140 140 -tag $tag -m $mask -i LiteBIRD -sp 1.54 20.0 -3.0 -mi 2000 -o $OUTPUT_DIR"

run_name="fgbuster_${tag}_${noise}"
job_name="FG_BUST"

echo "Submitting $run_name"
# Note: Using fgbuster-model script name. Ensure it exists in pyproject.toml / scripts.
jid=$(sbatch $SBATCH_ARGS --job-name=\"$job_name\" \
     $SLURM_SCRIPT $OUTPUT_DIR fgbuster-model $COMMON_ARGS)
job_ids+=("$jid")


# =============================================================================
# Analysis
# =============================================================================
deps=$(IFS=:; echo "${job_ids[*]}")
if [ -z "$deps" ]; then
    echo "No jobs submitted."
    exit 1
fi

echo "Submitting analysis jobs depending on: $deps"

# Validate
sbatch --dependency=afterany:$deps \
       $SBATCH_ARGS \
       --job-name=FG_validate \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis validate \
       -r fgbuster \
       -ird $OUTPUT_DIR \
       --scales 1e-4 1e-5

# Snapshot
snap_id=$(sbatch --dependency=afterany:$deps \
       $SBATCH_ARGS \
       --job-name=FG_snap \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap \
       -r fgbuster \
       -ird $OUTPUT_DIR \
       -s scipy_tnc \
       -o $OUTPUT_DIR/SNAPSHOT_FG \
       -mi 2000 \
       -n 64 \
       -i LiteBIRD)

# Plot
sbatch --dependency=afterany:$snap_id \
        $SBATCH_ARGS \
        --job-name=FG_plot \
        $SLURM_SCRIPT $OUTPUT_DIR r_analysis plot \
        -r fgbuster \
        -ird $OUTPUT_DIR \
        -a \
        --snapshot $OUTPUT_DIR/SNAPSHOT_FG 