#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
FORMAT="png"
FONT=14

echo "Starting Master Plots Generation (Local)..."

# =============================================================================
# 1. MINIMIZE ANALYSIS
# =============================================================================
echo "Processing MINIMIZE Analysis..."

DIR="RESULTS/MINIMIZE"
REGEX_MINIMIZE="kmeans_topk0.4_active_set_adabelief_lszoom fgbuster kmeans_lbfgs_lszoom_cond "

# 1. Snap
r_analysis snap \
    -r $REGEX_MINIMIZE \
    -ird $DIR/ \
    -o PAPER/SNAPPING/SNAP_MINIMIZE

# 2. Plot
r_analysis plot \
    -r $REGEX_MINIMIZE \
    -t "(This work) ADABK4" "FGBuster TNC" "(This work) L-BFGS" \
    -ird $DIR \
    --snapshot PAPER/SNAPPING/SNAP_MINIMIZE \
    --output-format $FORMAT --font-size $FONT \
    -pp -pt -ar -o PAPER/PLOTS/MINIMIZE

# 3. Python Validation Script
python3 slurm/runners/minimize_validation.py \
    --output-format $FORMAT --font-size $FONT --plot-dir PAPER/PLOTS/MINIMIZE

# 4. Validate
r_analysis validate \
    -r $REGEX_MINIMIZE \
    -ird $DIR \
    --scales 1e-5 --plot-type nll --aggregate --no-vmap --steps 100 \
    --output-format $FORMAT --font-size $FONT \
    -o PAPER/PLOTS/MINIMIZE/VALIDATION


# =============================================================================
# 2. KMEANS ANALYSIS
# =============================================================================
echo "Processing KMEANS Analysis..."

DIR_KM="RESULTS/KMEANS"
REGEX_ALL="kmeans_c1d1s1_BD(\d+)_TD(\d+)_BS(\d+)"

# Master Snapshot
r_analysis snap \
    -r "$REGEX_ALL" \
    -ird $DIR_KM/ \
    -o PAPER/SNAPPING/SNAP_KMEANS \
    -mi 1000 -s optax_lbfgs

# Variance Plot
r_analysis plot \
    -r "$REGEX_ALL" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arv -o PAPER/PLOTS/KMEANS/VARIANCES

# R Plots
REGEX_BD10000_TD500_BSXXX="kmeans_c1d1s1_BD10000_TD500_BS(\d+)"
r_analysis plot \
    -r "$REGEX_BD10000_TD500_BSXXX" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_TD500_VARY_BS

    
REGEX_BD10000_TDXXX_BS500="kmeans_c1d1s1_BD10000_TD(\d+)_BS500"
r_analysis plot \
    -r "$REGEX_BD10000_TDXXX_BS500" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_BS500_VARY_TD

# R Plots
REGEX_BD10000_TD3500_BSXXX="kmeans_c1d1s1_BD10000_TD3500_BS(\d+)"
r_analysis plot \
    -r "$REGEX_BD10000_TD3500_BSXXX" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_TD3500_VARY_BS

    
REGEX_BD10000_TDXXX_BS300="kmeans_c1d1s1_BD10000_TD(\d+)_BS300"
r_analysis plot \
    -r "$REGEX_BD10000_TDXXX_BS300" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BD10000_BS300_VARY_TD

REGEX_BDXXX_TD500_BS500="kmeans_c1d1s1_BD(\d+)_TD500_BS500"
r_analysis plot \
    -r "$REGEX_BDXXX_TD500_BS500" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -arc -o PAPER/PLOTS/KMEANS/R_PLOT/BS500_TD500_VARY_BD

# Best Runs Maps
r_analysis plot \
    -r kmeans_BD10000_TD2000_BS300 kmeans_BD4000_TD250_BS10 \
    -t "Best_r" "Best_sigma" \
    -ird $DIR_KM/ \
    --snapshot PAPER/SNAPPING/SNAP_KMEANS \
    --output-format $FORMAT --font-size $FONT \
    -ptm -psm -pp -pt -as -ar -o PAPER/PLOTS/KMEANS/R_PLOT/RES_MAPS


# =============================================================================
# 3. PTEP ANALYSIS
# =============================================================================
echo "Processing PTEP Analysis..."

r_analysis snap \
    -r ptep1 ptep2 kmeans_BD10000_TD2000_BS300 \
    -ird RESULTS/KMEANS/ RESULTS/MULTIRES/ \
    -o PAPER/SNAPPING/SNAP_PTEP \
    -mi 1000 -s optax_lbfgs

r_analysis plot \
    -r ptep1 ptep2 kmeans_BD10000_TD2000_BS300 \
    -t "PTEP1" "PTEP2" "Best_KM" \
    -ird RESULTS/KMEANS/ RESULTS/MULTIRES/ \
    --snapshot PAPER/SNAPPING/SNAP_PTEP \
    --output-format $FORMAT --font-size $FONT \
    -ptm -psm -pp -pt -as -ar -o PAPER/PLOTS/MULTIRES/


# =============================================================================
# 4. TENSOR TO SCALAR ANALYSIS
# =============================================================================
echo "Processing TENSOR TO SCALAR Analysis..."

DIR_CR="RESULTS/TENSOR_TO_SCALAR_34"

r_analysis snap \
    -r cr3d1s1 cr4d1s1 \
    -ird $DIR_CR \
    -o PAPER/SNAPPING/SNAP_TENSOR_TO_SCALAR_34 \
    -mi 1000 -s optax_lbfgs

r_analysis plot \
    -r cr3d1s1 cr4d1s1 \
    -t "r=0.003" "r=0.004" \
    -ird $DIR_CR \
    --snapshot PAPER/SNAPPING/SNAP_TENSOR_TO_SCALAR_34 \
    --output-format $FORMAT --font-size $FONT \
    -ptm -psm -pp -pt -as -ar -o PAPER/PLOTS/TENSOR_TO_SCALAR_34/

echo "All Plots Generated Locally."
