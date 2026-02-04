#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --time=05:00:00 --parsable"

job_ids=()

if [ -z "$1" ]; then
    echo "Usage: $0 <seed_start>"
    exit 1
fi
SEED=$1
SKY="c1d0s0"
OUTPUT_DIR="RESULTS/MINIMIZE/VALIDATE_FG"
mkdir -p $OUTPUT_DIR

# =============================================================================
# MultiRes MODELS (1)
# =============================================================================
# Zone 1 Low galactic represented by GAL020
NAME="fgbuster_${SKY}_BD64_TD0_BS2_LiteBIRD_GAL020_seed$SEED"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(sbatch $SBATCH_ARGS --job-name=FG_GAL020_M1_$SEED \
           $SLURM_SCRIPT $OUTPUT_DIR fgbuster-model  -n 64 -ns 10 -nr 1.0 -ud 64 0 2 \
           -tag $SKY -m GAL020 -i LiteBIRD -o $OUTPUT_DIR -mi 2000 \
           -ss $SEED \
           --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# Zone 2 Medium galactic represented by GAL040 - GAL020
NAME="fgbuster_${SKY}_BD64_TD4_BS2_LiteBIRD_GAL040_seed$SEED"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(sbatch $SBATCH_ARGS --job-name=FG_GAL040_M1_$SEED \
           $SLURM_SCRIPT $OUTPUT_DIR fgbuster-model  -n 64 -ns 10 -nr 1.0 -ud 64 4 2 \
           -tag $SKY -m GAL040 -i LiteBIRD -o $OUTPUT_DIR -mi 2000 \
           -ss $SEED \
           --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# Zone 3 High galactic represented by GAL060 - GAL040
NAME="fgbuster_${SKY}_BD64_TD8_BS4_LiteBIRD_GAL060_seed$SEED"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(sbatch $SBATCH_ARGS --job-name=FG_GAL060_M1_$SEED \
           $SLURM_SCRIPT $OUTPUT_DIR fgbuster-model  -n 64 -ns 10 -nr 1.0 -ud 64 8 4 \
           -tag $SKY -m GAL060 -i LiteBIRD -o $OUTPUT_DIR -mi 2000 \
           -ss $SEED \
           --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi


# =============================================================================
# CACHE_RUNS
# =============================================================================


deps=$(IFS=:; echo "${job_ids[*]}")

if [ -n "$deps" ]; then
    jid=$(sbatch --dependency=afterok:$deps \
            $SBATCH_ARGS \
           --job-name=CACHE_FGBUSTER \
           $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap -r fgbuster -ird $OUTPUT_DIR \
          -mi 2000 -n 64 -i LiteBIRD -s optax_lbfgs -o SNAPSHOT)
fi