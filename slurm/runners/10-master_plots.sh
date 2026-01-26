#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
FORMAT="png"
FONT=14

# Define standard SBATCH arguments
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --time=04:00:00"

echo "Submitting Master Plots Generation Jobs (Format: $FORMAT, Font: $FONT)..."

# =============================================================================
# 1. MINIMIZE ANALYSIS
# =============================================================================
echo "Submitting MINIMIZE Analysis..."

DIR="RESULTS/MINIMIZE"
RUN_NAME="PAPER/PLOTS/MINIMIZE"
REGEX_MINIMIZE="kmeans_topk0.4_active_set_adabelief_lszoom fgbuster kmeans_lbfgs_lszoom_cond "

# 1. Snap
jid_min_snap=$(sbatch $SBATCH_ARGS --job-name=SNAP_MIN $SLURM_SCRIPT $RUN_NAME r_analysis snap \
    -r $REGEX_MINIMIZE \
    -ird $DIR/ \
    -o PAPER/SNAPPING/SNAP_MINIMIZE)
echo "  Submitted SNAP_MIN (ID: $jid_min_snap)"

# 2. Plot (Depends on Snap)
jid_min_plot=$(sbatch --dependency=afterany:$jid_min_snap $SBATCH_ARGS --job-name=PLOT_MIN $SLURM_SCRIPT $RUN_NAME r_analysis plot \
    -r $REGEX_MINIMIZE \
    -t "(This work) ADABK4" "FGBuster TNC" "(This work) L-BFGS" \
    -ird $DIR \
    --snapshot PAPER/SNAPPING/SNAP_MINIMIZE \
    --output-format $FORMAT --font-size $FONT \
    -pp -pt -ar -o PAPER/PLOTS/MINIMIZE)
echo "  Submitted PLOT_MIN (ID: $jid_min_plot)"

# 3. Python Validation Script (Depends on Plot)
jid_min_py=$(sbatch --dependency=afterany:$jid_min_plot $SBATCH_ARGS --job-name=PY_MIN $SLURM_SCRIPT $RUN_NAME python runners/minimize_validation.py \
    --output-format $FORMAT --font-size $FONT --plot-dir PAPER/PLOTS/MINIMIZE)
echo "  Submitted PY_MIN (ID: $jid_min_py)"

# 4. Validate (Parallel to Plot)
jid_min_val=$(sbatch $SBATCH_ARGS --job-name=VAL_MIN $SLURM_SCRIPT $RUN_NAME r_analysis validate \
    -r $REGEX_MINIMIZE \
    -t "(This work) ADABK4" "FGBuster TNC" "(This work) L-BFGS" \
    -ird $DIR \
    --scales 1e-5 --plot-type nll --aggregate --no-vmap --steps 100 \
    --output-format $FORMAT --font-size $FONT \
    -o PAPER/PLOTS/MINIMIZE/VALIDATION)
echo "  Submitted VAL_MIN (ID: $jid_min_val)"


# =============================================================================
# 2. KMEANS ANALYSIS
# =============================================================================
echo "Submitting KMEANS Analysis..."

DIR_KM="RESULTS/KMEANS"
RUN_NAME_KM="PAPER/PLOTS/KMEANS"
REGEX_ALL="kmeans_c1d1s1_BD(\d+)_TD(\d+)_BS(\d+)"

# Master Snapshot
jid_km_snap=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM_ALL $SLURM_SCRIPT $RUN_NAME_KM r_analysis snap \
    -r "$REGEX_ALL" \
    -ird $DIR_KM/ \
    -o PAPER/SNAPPING/SNAP_KMEANS \
    -mi 1000 -s optax_lbfgs)
echo "  Submitted SNAP_KM_ALL (ID: $jid_km_snap)"

# Variance Plot (Depends on Snap)
sbatch --dependency=afterany:$jid_km_snap $SBATCH_ARGS --job-name=PLOT_KM_VAR $SLURM_SCRIPT $RUN_NAME_KM r_analysis plot \
    -r "$REGEX_ALL" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arv -o PAPER/PLOTS/KMEANS/VARIANCES

# R Plots (Depend on Snap)
REGEX_BD10000_TD500_BSXXX="kmeans_c1d1s1_BD10000_TD500_BS(\d+)"
sbatch --dependency=afterany:$jid_km_snap $SBATCH_ARGS --job-name=PLOT_KM_R1 $SLURM_SCRIPT $RUN_NAME_KM r_analysis plot \
    -r "$REGEX_BD10000_TD500_BSXXX" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_TD500_VARY_BS

    
REGEX_BD10000_TDXXX_BS500="kmeans_c1d1s1_BD10000_TD(\d+)_BS500"
sbatch --dependency=afterany:$jid_km_snap $SBATCH_ARGS --job-name=PLOT_KM_R2 $SLURM_SCRIPT $RUN_NAME_KM r_analysis plot \
    -r "$REGEX_BD10000_TDXXX_BS500" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_BS500_VARY_TD

# R Plots (Depend on Snap)
REGEX_BD10000_TD3500_BSXXX="kmeans_c1d1s1_BD10000_TD3500_BS(\d+)"
sbatch --dependency=afterany:$jid_km_snap $SBATCH_ARGS --job-name=PLOT_KM_R3 $SLURM_SCRIPT $RUN_NAME_KM r_analysis plot \
    -r "$REGEX_BD10000_TD3500_BSXXX" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_TD3500_VARY_BS

    
REGEX_BD10000_TDXXX_BS300="kmeans_c1d1s1_BD10000_TD(\d+)_BS300"
sbatch --dependency=afterany:$jid_km_snap $SBATCH_ARGS --job-name=PLOT_KM_R4 $SLURM_SCRIPT $RUN_NAME_KM r_analysis plot \
    -r "$REGEX_BD10000_TDXXX_BS300" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_BS3w00_VARY_TD

REGEX_BDXXX_TD500_BS500="kmeans_c1d1s1_BD(\d+)_TD500_BS500"
sbatch --dependency=afterany:$jid_km_snap $SBATCH_ARGS --job-name=PLOT_KM_R5 $SLURM_SCRIPT $RUN_NAME_KM r_analysis plot \
    -r "$REGEX_BDXXX_TD500_BS500" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BS500_TD500_VARY_BD

# Best Runs Maps
sbatch --dependency=afterany:$jid_km_snap $SBATCH_ARGS --job-name=PLOT_KM_BEST $SLURM_SCRIPT $RUN_NAME_KM r_analysis plot \
    -r kmeans_BD10000_TD2000_BS300 kmeans_BD4000_TD250_BS10 \
    -t "Best_r" "Best_sigma" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -atm -asm -pp -pt -as -ar -o PAPER/PLOTS/KMEANS/R_PLOT/RES_MAPS


# =============================================================================
# 3. PTEP ANALYSIS
# =============================================================================
echo "Submitting PTEP Analysis..."

RUN_NAME_PTEP="PAPER/PLOTS/MULTIRES"
jid_ptep_snap=$(sbatch $SBATCH_ARGS --job-name=SNAP_PTEP $SLURM_SCRIPT $RUN_NAME_PTEP r_analysis snap \
    -r ptep1 ptep2 kmeans_BD10000_TD2000_BS300 \
    -ird RESULTS/KMEANS/ RESULTS/MULTIRES/ \
    -o PAPER/SNAPPING/SNAP_PTEP \
    -mi 1000 -s optax_lbfgs)
echo "  Submitted SNAP_PTEP (ID: $jid_ptep_snap)"

sbatch --dependency=afterany:$jid_ptep_snap $SBATCH_ARGS --job-name=PLOT_PTEP $SLURM_SCRIPT $RUN_NAME_PTEP r_analysis plot \
    -r ptep1 ptep2 kmeans_BD10000_TD2000_BS300 \
    -t "PTEP1" "PTEP2" "Best_KM" \
    -ird RESULTS/KMEANS/ RESULTS/MULTIRES/ \
    --snapshot PAPER/SNAPPING/SNAP_PTEP \
    --output-format $FORMAT --font-size $FONT \
    -ptm -psm -pp -pt -as -ar -o PAPER/PLOTS/MULTIRES/


# =============================================================================
# 4. TENSOR TO SCALAR ANALYSIS
# =============================================================================
echo "Submitting TENSOR TO SCALAR Analysis..."

RUN_NAME_CR="PAPER/PLOTS/TENSOR_TO_SCALAR"
DIR_CR="RESULTS/TENSOR_TO_SCALAR_34"

jid_cr_snap=$(sbatch $SBATCH_ARGS --job-name=SNAP_CR $SLURM_SCRIPT $RUN_NAME_CR r_analysis snap \
    -r cr3d1s1 cr4d1s1 \
    -ird $DIR_CR \
    -o PAPER/SNAPPING/SNAP_TENSOR_TO_SCALAR_34 \
    -mi 1000 -s optax_lbfgs)
echo "  Submitted SNAP_CR (ID: $jid_cr_snap)"

sbatch --dependency=afterany:$jid_cr_snap $SBATCH_ARGS --job-name=PLOT_CR $SLURM_SCRIPT $RUN_NAME_CR r_analysis plot \
    -r cr3d1s1 cr4d1s1 \
    -t "r=0.003" "r=0.004" \
    -ird $DIR_CR \
    --snapshot PAPER/SNAPPING/SNAP_TENSOR_TO_SCALAR_34 \
    --output-format $FORMAT --font-size $FONT \
    -ptm -psm -pp -pt -as -ar -o PAPER/PLOTS/TENSOR_TO_SCALAR_34/

echo "All Master Plots Generation Jobs Submitted."