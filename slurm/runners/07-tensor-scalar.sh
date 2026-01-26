#!/bin/bash
# Tensor-to-scalar ratio runs (r=0.003, r=0.004)

# Collect all job IDs here
job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"
OUTPUT_DIR="RESULTS/RUNS/TENSOR_TO_SCALAR_34"

# Parameters matching the folder structure in RESULTS/RUNS/TENSOR_TO_SCALAR_34
# Based on existing folder: kmeans_cr3d1s1_BD10000_TD500_BS500_...
B_DUST=10000
T_DUST=500
B_SYNC=500

echo "=== Running Tensor-to-Scalar Runs ==="

# Loop over sky models: cr3d1s1 (r=0.003) and cr4d1s1 (r=0.004)
for TAG in cr3d1s1 cr4d1s1; do
    echo "Submitting jobs for TAG=$TAG"
    
    # Loop over masks
    for MASK in GAL020 GAL040 GAL060; do
        JOB_NAME="KM_${TAG}_${MASK}"
        NAME="kmeans_${TAG}_BD${B_DUST}_TD${T_DUST}_BS${B_SYNC}_${MASK}"
        
        jid=$(sbatch $BATCH_PARAMS --job-name=$JOB_NAME \
            $SLURM_SCRIPT $OUTPUT_DIR \
            kmeans-model -n 64 -ns 10 -nr 1.0 \
            -pc $B_DUST $T_DUST $B_SYNC \
            -tag $TAG -m $MASK -i LiteBIRD \
            -s active_set_adabelief -top_k 0.4 -mi 2000 \
            --name $NAME -o $OUTPUT_DIR)
        
        job_ids+=("$jid")
    done
done

# Dependencies for r_analysis
deps=$(IFS=:; echo "${job_ids[*]}")

echo "Submitting analysis job..."

# Analysis for cr3d1s1
sbatch --dependency=afterany:$deps \
    $BATCH_PARAMS \
    --job-name=ANA_TENSOR_3 \
    $SLURM_SCRIPT $OUTPUT_DIR \
    r_analysis snap -r "kmeans_cr3d1s1_BD${B_DUST}_TD${T_DUST}_BS${B_SYNC}" -ird $OUTPUT_DIR \
    -mi 2000 -s optax_lbfgs -n 64 -i LiteBIRD

# Analysis for cr4d1s1
sbatch --dependency=afterany:$deps \
    $BATCH_PARAMS \
    --job-name=ANA_TENSOR_4 \
    $SLURM_SCRIPT $OUTPUT_DIR \
    r_analysis snap -r "kmeans_cr4d1s1_BD${B_DUST}_TD${T_DUST}_BS${B_SYNC}" -ird $OUTPUT_DIR \
    -mi 2000 -s optax_lbfgs -n 64 -i LiteBIRD
