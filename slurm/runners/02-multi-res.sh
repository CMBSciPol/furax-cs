#!/bin/bash

SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --time=05:00:00 --parsable"

job_ids=()

SOLVER="ADABK0"
OUTPUT_DIR="RESULTS/MULTIRES"
SKY="c1d0s0"
NS=100
# =============================================================================
# MultiRes MODELS (1)
# =============================================================================
# Zone 1 Low galactic represented by GAL020
NAME="ptep1_${SKY}_BD64_TD0_BS2_LiteBIRD_GAL020"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL020_M1 \
           $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns $NS -nr 1.0 -ud 64 0 2 \
           -tag $SKY -m GAL020 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
           --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# Zone 2 Medium galactic represented by GAL040 - GAL020
NAME="ptep1_${SKY}_BD64_TD4_BS2_LiteBIRD_GAL040"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL040_M1 \
           $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns $NS -nr 1.0 -ud 64 4 2 \
           -tag $SKY -m GAL040 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
           --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# Zone 3 High galactic represented by GAL060 - GAL040
NAME="ptep1_${SKY}_BD64_TD8_BS4_LiteBIRD_GAL060"
if [ ! -f "$OUTPUT_DIR/$NAME/best_params.npz" ]; then
    jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL060_M1 \
           $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns $NS -nr 1.0 -ud 64 8 4 \
           -tag $SKY -m GAL060 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
           --name $NAME)
    job_ids+=($jid)
else
    echo "Skipping $NAME (already done)"
fi

# =============================================================================
# MultiRes MODELS (2)
# =============================================================================
if false; then
# Zone 1 Low galactic represented by GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL020_M2 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns $NS -nr 1.0 -ud 64 2 2 \
       -tag $SKY -m GAL020 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
       --name ptep2_${SKY}_BD64_TD2_BS2_LiteBIRD_GAL020)
job_ids+=($jid)
# Zone 2 Medium galactic represented by GAL040 - GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL040_M2 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns $NS -nr 1.0 -ud 64 8 2 \
       -tag $SKY -m GAL040 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
       --name ptep2_${SKY}_BD64_TD8_BS2_LiteBIRD_GAL040)
job_ids+=($jid)
# Zone 3 High galactic represented by GAL060 - GAL040
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL060_M2 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns $NS -nr 1.0 -ud 64 16 4 \
       -tag $SKY -m GAL060 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s $SOLVER \
       --name ptep2_${SKY}_BD64_TD16_BS4_LiteBIRD_GAL060)
job_ids+=($jid)
fi
# =============================================================================
# CACHE_RUNS
# =============================================================================


deps=$(IFS=:; echo "${job_ids[*]}")

if [ -n "$deps" ]; then
    jid=$(sbatch --dependency=afterok:$deps \
            $SBATCH_ARGS \
           --job-name=CACHE_PTEP \
           $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap -r ptep1 ptep2 -ird $OUTPUT_DIR \
          -mi 2000 -n 64 -i LiteBIRD -s optax_lbfgs -o SNAPSHOT)
fi
