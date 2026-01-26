#!/bin/bash

SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"

job_ids=()

OUTPUT_DIR="RESULTS/MULTIRES"
# =============================================================================
# MultiRes MODELS (1)
# =============================================================================
# Zone 1 Low galactic represented by GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL020_M1 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns 10 -nr 1.0 -ud 64 0 2 \
       -tag c1d1s1 -m GAL020 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s active_set_adabelief -top_k 0.4 \
       --name ptep1_c1d1s1_BD64_TD0_BS2_LiteBIRD_GAL020)
job_ids+=($jid)
# Zone 2 Medium galactic represented by GAL040 - GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL040_M1 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns 10 -nr 1.0 -ud 64 4 2 \
       -tag c1d1s1 -m GAL040 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s active_set_adabelief -top_k 0.4 \
       --name ptep1_c1d1s1_BD64_TD4_BS2_LiteBIRD_GAL040)
job_ids+=($jid)
# Zone 3 High galactic represented by GAL060 - GAL040
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL060_M1 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns 10 -nr 1.0 -ud 64 8 4 \
       -tag c1d1s1 -m GAL060 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s active_set_adabelief -top_k 0.4 \
       --name ptep1_c1d1s1_BD64_TD8_BS4_LiteBIRD_GAL060)
job_ids+=($jid)
# =============================================================================
# MultiRes MODELS (2)
# =============================================================================
# Zone 1 Low galactic represented by GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL020_M2 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns 10 -nr 1.0 -ud 64 2 2 \
       -tag c1d1s1 -m GAL020 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s active_set_adabelief -top_k 0.4 \
       --name ptep2_c1d1s1_BD64_TD2_BS2_LiteBIRD_GAL020)
job_ids+=($jid)
# Zone 2 Medium galactic represented by GAL040 - GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL040_M2 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns 10 -nr 1.0 -ud 64 8 2 \
       -tag c1d1s1 -m GAL040 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s active_set_adabelief -top_k 0.4 \
       --name ptep2_c1d1s1_BD64_TD8_BS2_LiteBIRD_GAL040)
job_ids+=($jid)
# Zone 3 High galactic represented by GAL060 - GAL040
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL060_M2 \
       $SLURM_SCRIPT $OUTPUT_DIR ptep-model  -n 64 -ns 10 -nr 1.0 -ud 64 16 4 \
       -tag c1d1s1 -m GAL060 -i LiteBIRD -cond -o $OUTPUT_DIR -mi 2000 -s active_set_adabelief -top_k 0.4 \
       --name ptep2_c1d1s1_BD64_TD16_BS4_LiteBIRD_GAL060)
job_ids+=($jid)

# =============================================================================
# CACHE_RUNS
# =============================================================================


deps=$(IFS=:; echo "${job_ids[*]}")

jid=$(sbatch --dependency=afterok:$deps \
        $SBATCH_ARGS \
       --job-name=CACHE_PTEP \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap -r ptep1 ptep2 -ird $OUTPUT_DIR \
      -mi 2000 -n 64 -i LiteBIRD -s optax_lbfgs -o SNAPSHOT)
