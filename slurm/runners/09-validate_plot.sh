#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# =============================================================================
# KMEANS SETS (Regex)
# =============================================================================

# Set 1: Varying BS
DIR="RESULTS/KMEANS/BD10000_TD500_BSXXX"
REGEX="kmeans_c1d1s1_BD10000_TD500_BS(\d+)"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM1 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM1 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/BD10000_TD500_BSXXX
sbatch $SBATCH_ARGS --job-name=VAL_KM1 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/BD10000_TD500_BSXXX

# Set 2: Varying TD
DIR="RESULTS/KMEANS/BD10000_TDXXX_BS500"
REGEX="kmeans_c1d1s1_BD10000_TD(\d+)_BS500"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM2 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM2 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/BD10000_TDXXX_BS500
sbatch $SBATCH_ARGS --job-name=VAL_KM2 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/BD10000_TDXXX_BS500

# Set 3: Varying BS Small
DIR="RESULTS/KMEANS/BD4000_TD10_BSXXX"
REGEX="kmeans_c1d1s1_BD4000_TD10_BS(\d+)"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM3 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM3 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/BD4000_TD10_BSXXX
sbatch $SBATCH_ARGS --job-name=VAL_KM3 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/BD4000_TD10_BSXXX

# Set 4: Varying TD Small
DIR="RESULTS/KMEANS/BD4000_TDXXX_BS10"
REGEX="kmeans_c1d1s1_BD4000_TD(\d+)_BS10"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM4 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM4 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/BD4000_TDXXX_BS10
sbatch $SBATCH_ARGS --job-name=VAL_KM4 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/BD4000_TDXXX_BS10

# Set 5: Varying BD
DIR="RESULTS/KMEANS/BDXXX_TD500_BS500"
REGEX="kmeans_c1d1s1_BD(\d+)_TD500_BS500"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM5 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM5 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/BDXXX_TD500_BS500
sbatch $SBATCH_ARGS --job-name=VAL_KM5 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/BDXXX_TD500_BS500


# Set 6: Varying BS second run
DIR="RESULTS/KMEANS/BD10000_TD3500_BSXXX"
REGEX="kmeans_c1d1s1_BD10000_TD3500_BS(\d+)"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM6 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM6 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/BD10000_TD3500_BSXXX
sbatch $SBATCH_ARGS --job-name=VAL_KM6 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/BD10000_TD3500_BSXXX

# Set 7: Varying TD second run
DIR="RESULTS/KMEANS/BD10000_TDXXX_BS300"
REGEX="kmeans_c1d1s1_BD10000_TD(\d+)_BS300"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM7 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM7 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/BD10000_TDXXX_BS300
sbatch $SBATCH_ARGS --job-name=VAL_KM7 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/BD10000_TDXXX_BS300


# All together
DIR="RESULTS/KMEANS/ALL_SETS"
REGEX="kmeans_c1d1s1_BD(\d+)_TD(\d+)_BS(\d+)"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM_ALL $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM_ALL $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/ALL_SETS

# =============================================================================
# MINIMIZE (VALIDATION)
# =============================================================================

# FGBuster
DIR="RESULTS/MINIMIZE/VALIDATE_FG"
REGEX="fgbuster"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_FG $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_FG $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/MINIMIZE/VALIDATE_FG
sbatch $SBATCH_ARGS --job-name=VAL_FG $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/MINIMIZE/VALIDATE_FG

# Furax
DIR="RESULTS/MINIMIZE/VALIDATE_FURAX"
REGEX="kmeans"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_FX $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_FX $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/MINIMIZE/VALIDATE_FURAX
sbatch $SBATCH_ARGS --job-name=VAL_FX $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/MINIMIZE/VALIDATE_FURAX


# =============================================================================
# MULTIRES (PTEP) - Chained
# =============================================================================
DIR="RESULTS/MULTIRES"

# Set 1: ptep1
jid1=$(sbatch $SBATCH_ARGS --job-name=SNAP_PTEP1 $SLURM_SCRIPT ANALYSIS r_analysis snap -r ptep1 -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=PLOT_PTEP1 $SLURM_SCRIPT ANALYSIS r_analysis plot -r ptep1 -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/MULTIRES
sbatch $SBATCH_ARGS --job-name=VAL_PTEP1 $SLURM_SCRIPT ANALYSIS r_analysis validate -r ptep1 -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/MULTIRES

# Set 2: ptep2 (Depends on jid1 to share/update snapshot)
jid2=$(sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=SNAP_PTEP2 $SLURM_SCRIPT ANALYSIS r_analysis snap -r ptep2 -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid2 $SBATCH_ARGS --job-name=PLOT_PTEP2 $SLURM_SCRIPT ANALYSIS r_analysis plot -r ptep2 -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/MULTIRES
sbatch $SBATCH_ARGS --job-name=VAL_PTEP2 $SLURM_SCRIPT ANALYSIS r_analysis validate -r ptep2 -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/MULTIRES


# =============================================================================
# TENSOR_TO_SCALAR - Chained
# =============================================================================
DIR="RESULTS/TENSOR_TO_SCALAR_34"

# Set 1: cr3d1s1
jid1=$(sbatch $SBATCH_ARGS --job-name=SNAP_CR3 $SLURM_SCRIPT ANALYSIS r_analysis snap -r kmeans_cr3d1s1 -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=PLOT_CR3 $SLURM_SCRIPT ANALYSIS r_analysis plot -r kmeans_cr3d1s1 -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/TENSOR_TO_SCALAR_34
sbatch $SBATCH_ARGS --job-name=VAL_CR3 $SLURM_SCRIPT ANALYSIS r_analysis validate -r kmeans_cr3d1s1 -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/TENSOR_TO_SCALAR_34

# Set 2: cr4d1s1 (Depends on jid1)
jid2=$(sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=SNAP_CR4 $SLURM_SCRIPT ANALYSIS r_analysis snap -r kmeans_cr4d1s1 -ird $DIR -o REPORT/SNAPPING -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid2 $SBATCH_ARGS --job-name=PLOT_CR4 $SLURM_SCRIPT ANALYSIS r_analysis plot -r kmeans_cr4d1s1 -ird $DIR -a --snapshot REPORT/SNAPPING -o REPORT/PLOTS/TENSOR_TO_SCALAR_34
sbatch $SBATCH_ARGS --job-name=VAL_CR4 $SLURM_SCRIPT ANALYSIS r_analysis validate -r kmeans_cr4d1s1 -ird $DIR --noise-ratio 1.0 --scales 1e-4 1e-5 --steps 10 -o REPORT/PLOTS/TENSOR_TO_SCALAR_34
