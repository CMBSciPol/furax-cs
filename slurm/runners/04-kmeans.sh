#!/bin/bash
# Corrected Bash loop over BS values

# Collect all kmeans job IDs here
job_ids=()
BATCH_PARAMS='--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100'
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"
# Define the 5 sets of parameters to run
# Format: "B_DUST T_DUST B_SYNC VARYING_PARAM"
# VARYING_PARAM key: 1=B_DUST, 2=T_DUST, 3=B_SYNC
CONFIGS=(
    "4000 10 0 3"    # Case 1: B_DUST=4000, T_DUST=10, Vary B_SYNC
    "4000 0 10 2"    # Case 2: B_DUST=4000, B_SYNC=10, Vary T_DUST
    "10000 500 0 3"  # Case 3: B_DUST=10000, T_DUST=500, Vary B_SYNC
    "10000 0 500 2"  # Case 4: B_DUST=10000, B_SYNC=500, Vary T_DUST
    "0 500 500 1"    # Case 5: T_DUST=500, B_SYNC=500, Vary B_DUST
    "10000 3500 0 3"  # Case 6: B_DUST=10000, T_DUST=3500, Vary B_SYNC
    "10000 0 300 2"  # Case 7: B_DUST=10000, B_SYNC=300, Vary T_DUST
)

# Ranges for varying parameters
# For Case 1 & 2 (low values)
RANGE_LOW="50 100 150 200 250 300 350 400 450"
# For Case 3, 4, 5 (high values)
RANGE_HIGH="50 100 200 300 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000"

for config in "${CONFIGS[@]}"; do
    read -r B_DUST_BASE T_DUST_BASE B_SYNC_BASE VARY_IDX <<< "$config"
    
    # Determine varying parameter and range
    if [ "$VARY_IDX" -eq 1 ]; then
        VARY_NAME="B_DUST"
        RANGE=$RANGE_HIGH
        OUTPUT_BASE="BDXXX_TD${T_DUST_BASE}_BS${B_SYNC_BASE}"
    elif [ "$VARY_IDX" -eq 2 ]; then
        VARY_NAME="T_DUST"
        if [ "$B_DUST_BASE" -eq 4000 ]; then RANGE=$RANGE_LOW; else RANGE=$RANGE_HIGH; fi
        OUTPUT_BASE="BD${B_DUST_BASE}_TDXXX_BS${B_SYNC_BASE}"
    elif [ "$VARY_IDX" -eq 3 ]; then
        VARY_NAME="B_SYNC"
        if [ "$B_DUST_BASE" -eq 4000 ]; then RANGE=$RANGE_LOW; else RANGE=$RANGE_HIGH; fi
        OUTPUT_BASE="BD${B_DUST_BASE}_TD${T_DUST_BASE}_BSXXX"
    fi

    OUTPUT_DIR="RESULTS/KMEANS/$OUTPUT_BASE"
    current_job_ids=()

    echo "=== Running Configuration: $OUTPUT_BASE (Varying $VARY_NAME) ==="

    for VAL in $RANGE; do
        # Set parameters based on what is varying
        B_DUST=$B_DUST_BASE
        T_DUST=$T_DUST_BASE
        B_SYNC=$B_SYNC_BASE

        if [ "$VARY_IDX" -eq 1 ]; then B_DUST=$VAL; fi
        if [ "$VARY_IDX" -eq 2 ]; then T_DUST=$VAL; fi
        if [ "$VARY_IDX" -eq 3 ]; then B_SYNC=$VAL; fi

        JOB_NAME="KM_${VARY_NAME}_${VAL}"

        # Submit jobs for 3 masks
        for MASK in GAL020 GAL040 GAL060; do
             NAME="kmeans_c1d1s1_BD${B_DUST}_TD${T_DUST}_BS${B_SYNC}_${MASK}"
             jid=$(sbatch $BATCH_PARAMS --job-name=${JOB_NAME}_${MASK} \
                $SLURM_SCRIPT $OUTPUT_DIR \
                kmeans-model -n 64 -ns 10 -nr 1.0 \
                -pc $B_DUST $T_DUST $B_SYNC \
                -tag c1d1s1 -m $MASK -i LiteBIRD \
                -s active_set_adabelief -top_k 0.4 -mi 2000 \
                --name $NAME -o $OUTPUT_DIR)
             current_job_ids+=("$jid")
        done
    done
    
    # Dependencies for r_analysis
    deps=$(IFS=:; echo "${current_job_ids[*]}")
    
    # Construct regex for r_analysis
    if [ "$VARY_IDX" -eq 1 ]; then REGEX="kmeans_c1d1s1_BD(\d+)_TD${T_DUST}_BS${B_SYNC}"; fi
    if [ "$VARY_IDX" -eq 2 ]; then REGEX="kmeans_c1d1s1_BD${B_DUST}_TD(\d+)_BS${B_SYNC}"; fi
    if [ "$VARY_IDX" -eq 3 ]; then REGEX="kmeans_c1d1s1_BD${B_DUST}_TD${T_DUST}_BS(\d+)"; fi

    sbatch --dependency=afterany:$deps \
        $BATCH_PARAMS \
        --job-name=ANA_${OUTPUT_BASE} \
        $SLURM_SCRIPT $OUTPUT_DIR \
        r_analysis snap -r "$REGEX" -ird $OUTPUT_DIR \
        -mi 2000 -s optax_lbfgs -n 64 -i LiteBIRD
done

