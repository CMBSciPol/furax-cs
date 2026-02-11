#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --time=05:00:00 --parsable"
OUTPUT_DIR="RESULTS/MINIMIZE/VALIDATE_FURAX"
mkdir -p $OUTPUT_DIR

job_ids=()
noise="1.0"
sky="c1d0s0"
tag=$sky
mask="ALL-GALACTIC"

# Common arguments for all runs
# We use the same parameters as in validate_k.sh for consistency
COMMON_ARGS="-n 64 -ns 20 -nr $noise -pc 140 140 140 -tag $tag -m $mask -i LiteBIRD -sp 1.54 20.0 -3.0 -mi 2000 -o $OUTPUT_DIR"

# =============================================================================
# FURAX Runs (Active Set, L-BFGS, SciPy TNC)
# =============================================================================

# 2. Optax L-BFGS (example: zoom + cond)
ls_lbfgs="zoom"
flag="-cond"
cname="cond"
run_name_lbfgs="kmeans_lbfgs_ls${ls_lbfgs}_${cname}"
job_name_lbfgs="FX_LB"

if [ ! -f "$OUTPUT_DIR/$run_name_lbfgs/best_params.npz" ]; then
    echo "Submitting $run_name_lbfgs"
    jid=$(sbatch $SBATCH_ARGS --job-name="$job_name_lbfgs" \
            $SLURM_SCRIPT $OUTPUT_DIR kmeans-model $COMMON_ARGS \
            -s optax_lbfgs -ls $ls_lbfgs $flag --name $run_name_lbfgs)
    job_ids+=("$jid")
else
    echo "Skipping $run_name_lbfgs (already done)"
fi

# 3. SciPy TNC
run_name_tnc="kmeans_scipy_tnc"
job_name_tnc="FX_TNC"

if [ ! -f "$OUTPUT_DIR/$run_name_tnc/best_params.npz" ]; then
    echo "Submitting $run_name_tnc"
    jid=$(sbatch $SBATCH_ARGS --job-name="$job_name_tnc" \
         $SLURM_SCRIPT $OUTPUT_DIR kmeans-model $COMMON_ARGS \
         -s scipy_tnc --name $run_name_tnc)
    job_ids+=("$jid")
else
    echo "Skipping $run_name_tnc (already done)"
fi

# 4. AdaBelief (Conditioned + Zoom)
# Note: Ensure "adabelief" is a valid solver name in your system
run_name_ada="kmeans_adabelief_lszoom_cond"
job_name_ada="FX_ADA"

if [ ! -f "$OUTPUT_DIR/$run_name_ada/best_params.npz" ]; then
    echo "Submitting $run_name_ada"
    jid=$(sbatch $SBATCH_ARGS --job-name="$job_name_ada" \
            $SLURM_SCRIPT $OUTPUT_DIR kmeans-model $COMMON_ARGS \
            -s adabelief -ls zoom -cond --name $run_name_ada)
    job_ids+=("$jid")
else
    echo "Skipping $run_name_ada (already done)"
fi

# 5. Active Set AdaBelief (Zoom + k values)
ks=("0" "1" "2" "4" "6" "8")
AS_ADA_PATS=""

for k in "${ks[@]}"; do
    run_name="kmeans_topk${k}_active_set_adabelief_lszoom"
    job_name="FX_AS_ADA_${k}"
    
    if [ ! -f "$OUTPUT_DIR/$run_name/best_params.npz" ]; then
        echo "Submitting $run_name"
        jid=$(sbatch $SBATCH_ARGS --job-name="$job_name" \
                $SLURM_SCRIPT $OUTPUT_DIR kmeans-model $COMMON_ARGS \
                -s ADABK$k --name $run_name)
        job_ids+=("$jid")
    else
        echo "Skipping $run_name (already done)"
    fi
    
    # Capture pattern for this run
    AS_ADA_PATS="$AS_ADA_PATS $run_name"
done

# Regex to capture active set runs as a group
# Matches: kmeans_topk(0.1)_active_set_adabelief_lszoom
AS_ADA_REG="kmeans_ADABK(\d+)_active_set_adabelief_lszoom"


# =============================================================================
# Analysis
# =============================================================================
deps=$(IFS=:; echo "${job_ids[*]}")
if [ -z "$deps" ]; then
    echo "No new jobs submitted. Skipping analysis."
else
    echo "Submitting analysis jobs depending on: $deps"

    # Regex to capture all Furax runs submitted above
    # Note: AS_ADA_REG captures the group for plotting/validation comparison
    RUN_PATS="$run_name_lbfgs $run_name_tnc $run_name_ada $AS_ADA_REG"

    # Validate
    sbatch --dependency=afterany:$deps \
           $SBATCH_ARGS \
           --job-name=FX_val_FURAX \
           $SLURM_SCRIPT $OUTPUT_DIR r_analysis validate \
           -r $RUN_PATS \
           -ird $OUTPUT_DIR \
           --scales 1e-4 1e-5

    # Snapshot
    snap_id=$(sbatch --dependency=afterany:$deps \
           $SBATCH_ARGS \
           --job-name=FX_snap_FURAX \
           $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap \
           -r $RUN_PATS \
           -ird $OUTPUT_DIR \
           -s optax_lbfgs \
           -o $OUTPUT_DIR/SNAPSHOT_FURAX \
           -mi 2000 \
           -n 64 \
           -i LiteBIRD)

    # Plot
    sbatch --dependency=afterany:$snap_id \
            $SBATCH_ARGS \
            --job-name=FX_plt_FURAX \
            $SLURM_SCRIPT $OUTPUT_DIR r_analysis plot \
            -r $RUN_PATS \
            -ird $OUTPUT_DIR \
            -a \
            --snapshot $OUTPUT_DIR/SNAPSHOT_FURAX
fi 