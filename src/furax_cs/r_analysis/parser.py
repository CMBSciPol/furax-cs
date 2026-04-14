import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the r_analysis tool using subcommands.

    Structure:
        Global/Common Args: -n, -i, -r, -ird, --no-tex
        Subcommands:
            1. snap:  Compute statistics and save to disk (no plotting).
            2. plot:  Generate plots (from raw results or snapshot).
            3. validate: Run NLL validation analysis.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with a 'subcommand' attribute indicating the mode.
    """
    # 1. Define Parent Parser (Common arguments shared by all subcommands)
    # add_help=False is crucial here to prevent conflict with the main parser's help
    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument("-n", "--nside", type=int, default=64, help="The nside of the map")
    common_parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
        help="Instrument to use",
    )
    common_parser.add_argument(
        "-r",
        "--runs",
        type=str,
        nargs="*",
        help="List of run name keywords to filter result folders",
        required=True,
    )
    common_parser.add_argument(
        "-ird",
        "--input-results-dir",
        type=str,
        nargs="*",
        help="Directory where results are stored",
        required=True,
    )
    common_parser.add_argument(
        "--no-tex", action="store_true", help="Disable LaTeX rendering for plot text"
    )
    common_parser.add_argument(
        "--sky",
        type=str,
        default="c1d0s0",
        help="Sky model tag to use for true parameters (default: c1d0s0)",
    )
    common_parser.add_argument(
        "-mi",
        "--max-iterations",
        type=int,
        default=1000,
        help="Max iterations for computing systematics",
    )
    common_parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="optax_lbfgs",
        help="Solver for optimization. Options: optax_lbfgs, optax_lbfgs, "
        "optimistix_bfgs_wolfe, optimistix_lbfgs_wolfe, optimistix_ncg_hs_wolfe, "
        "scipy_tnc, zoom (alias), backtrack (alias), adam",
    )

    # 2. Main Parser
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
EXAMPLES:
  1. Compute statistics (snapshot) for runs matching 'kmeans' and 'c1d1s1':
     r_analysis snap -n 64 -r "kmeans" "c1d1s1" -ird results/ -o snapshots/my_snap.npz

  2. Plot all spectra from a previously saved snapshot:
     r_analysis plot -n 64 -r ".*" -ird results/ --snapshot snapshots/my_snap.npz -as

  3. Plot specific metrics (R vs Clusters) directly from results:
     r_analysis plot -n 64 -r "kmeans_set1" -ird results/ -arc

  4. Run validation analysis on specific runs:
     r_analysis validate -n 64 -r "validation" -ird results/ --steps 10 --noise-ratio 0.1

ARGUMENT NOTES:
  -r / --runs:
     Accepts a list of regex patterns. Only folders in input-results-dir matching ALL
     patterns will be processed.
     Example: -r "kmeans" "mask020" matches "results/kmeans_mask020_..."
     but NOT "results/kmeans_mask040_...".
     you can also do for example "kmeans_BD(\d+)_TD500" to capture groups.
     This will take all folders matching the pattern and extract the group (e.g. BD number)
""",
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Mode of operation")

    # ==========================================
    # 1. SNAP SUBCOMMAND
    # ==========================================
    parser_snap = subparsers.add_parser(
        "snap",
        parents=[common_parser],
        help="Compute all statistics and save snapshot without plotting",
    )
    parser_snap.add_argument(
        "-o",
        "--output-parquet",
        type=str,
        required=True,
        help="Path to the output parquet file (e.g. snapshots/my_run.parquet)",
    )
    parser_snap.add_argument(
        "--noise-selection",
        type=str,
        default="min-value",
        help="Noise realization selection for plotting: 'min-value' (default), 'min-nll', or an integer index.",
    )
    parser_snap.add_argument(
        "--max-ns",
        type=int,
        default=None,
        help="Maximum number of noise realizations to use (default: all).",
    )
    parser_snap.add_argument(
        "--no-images",
        action="store_true",
        help="Skip rendering mollview PIL images into the parquet (faster snap)",
    )
    parser_snap.add_argument(
        "--combine",
        action="store_true",
        help=(
            "Merge all matched result dirs (from all -r patterns) into a single entry. "
            "Use --name to set the entry name (defaults to 'COMBINED')."
        ),
    )
    parser_snap.add_argument(
        "--name",
        type=str,
        nargs="*",
        default=None,
        help="Optional display names for each run group (one per matched pattern).",
    )
    parser_snap.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum number of entries per parquet file. Splits into numbered files if exceeded.",
    )

    # ==========================================
    # 2. BIN SUBCOMMAND
    # ==========================================
    parser_bin = subparsers.add_parser(
        "bin",
        parents=[common_parser],
        help="Bin spectral parameters and produce patches .npy files",
    )
    parser_bin.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for patches .npy files",
    )
    parser_bin.add_argument(
        "--bin-bd",
        type=int,
        default=None,
        help="Number of bins for beta_dust parameter",
    )
    parser_bin.add_argument(
        "--bin-td",
        type=int,
        default=None,
        help="Number of bins for temp_dust parameter",
    )
    parser_bin.add_argument(
        "--bin-bs",
        type=int,
        default=None,
        help="Number of bins for beta_synchrotron parameter",
    )
    parser_bin.add_argument(
        "--noise-selection",
        type=str,
        default="min-value",
        help="Noise realization selection for bin edge definition: 'min-value', 'min-nll', or index",
    )
    parser_bin.add_argument(
        "--fwhm",
        type=float,
        default=None,
        help="FWHM (degrees) for Gaussian HEALPix smoothing applied before binning. Omit for no smoothing.",
    )

    # ==========================================
    # 3. PLOT SUBCOMMAND
    # ==========================================
    # Minimal parent parser for plot (no -r/-ird/-n/--sky/-mi/-s needed)
    plot_common_parser = argparse.ArgumentParser(add_help=False)
    plot_common_parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
        help="Instrument to use",
    )
    plot_common_parser.add_argument(
        "--no-tex", action="store_true", help="Disable LaTeX rendering for plot text"
    )

    parser_plot = subparsers.add_parser(
        "plot", parents=[plot_common_parser], help="Generate plots from parquet snapshots"
    )

    # Input/Output for plotting
    parser_plot.add_argument(
        "--parquet-dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing .parquet snapshot files produced by 'snap'",
    )
    parser_plot.add_argument(
        "-r",
        "--runs",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional regex patterns to filter parquet files by keyword (stem). "
            "Only parquets whose filename stem matches any pattern are loaded. "
            "If omitted, all parquets in --parquet-dir are loaded."
        ),
    )
    parser_plot.add_argument(
        "-g",
        "--groups",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Regex patterns defining named groups. Each pattern forms a separate group. "
            "Multiple-file aggregate plots run once per group (saved in {output}/{group}/). "
            "Single-file aggregate plots (plot_r_vs_c, plot_v_vs_c, plot_nll_vs_c) "
            "show one line per group on a combined chart."
        ),
    )
    parser_plot.add_argument(
        "-gt",
        "--group-titles",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Human-readable title for each -g group. Used as subdir name and legend label. "
            "Defaults to the group pattern (or 'results' when -g is omitted)."
        ),
    )
    parser_plot.add_argument(
        "-t",
        "--title",
        type=str,
        nargs="*",
        default=None,
        help="Per-row title overrides (sequential across all groups). Used as curve labels in r-likelihood plots.",
    )
    parser_plot.add_argument(
        "-o", "--output", type=str, help="Output directory for plots", default="plots/"
    )
    parser_plot.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf", "show"],
        default="png",
        help="Output format: png, pdf, or show (inline)",
    )
    parser_plot.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Font size for plots",
    )
    parser_plot.add_argument(
        "--transparent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save figures with transparent background (default: True). Use --no-transparent to disable.",
    )
    parser_plot.add_argument(
        "--color",
        type=str,
        nargs="*",
        default=None,
        help="Custom color list for run lines (e.g. --color red blue green). Cycles if fewer than runs.",
    )
    parser_plot.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=None,
        help="x-axis limits for r-estimation plots (e.g. --xlim -0.002 0.01)",
    )
    parser_plot.add_argument(
        "--r-legend-anchor",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="bbox_to_anchor for the legend in r-estimation plots (e.g. --r-legend-anchor 1.0 1.0)",
    )
    parser_plot.add_argument(
        "--s-legend-anchor",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="bbox_to_anchor for the legend in BB spectra plots (e.g. --s-legend-anchor 1.0 1.0)",
    )
    parser_plot.add_argument(
        "--r-figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=None,
        help="Figure size for r-estimation plots in inches (e.g. --r-figsize 10 8)",
    )
    parser_plot.add_argument(
        "--s-figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=None,
        help="Figure size for BB spectra plots in inches (e.g. --s-figsize 10 8)",
    )
    parser_plot.add_argument(
        "--r-range",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=None,
        help="r fill-between range for BB spectra plot (e.g. --r-range 1e-3 4e-3)",
    )
    parser_plot.add_argument(
        "--r-plot",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=None,
        help="Two truth r values plotted as separate lines in BB spectra + vertical lines in r-likelihood (e.g. --r-plot 0 3e-3)",
    )

    parser_plot.add_argument(
        "--cl-obs-label",
        action="store_true",
        default=False,
        help=(
            "Use C_ell^obs label for the dashed line in BB spectra plots "
            "(for runs that include CMB signal). Default label is 'Total (C_ell^res)'."
        ),
    )
    parser_plot.add_argument(
        "--no-tot-residuals",
        action="store_true",
        default=False,
        help="Omit the total residuals line from aggregate BB spectra plots.",
    )

    # Visualization Toggles
    # Grouping them makes the --help output much cleaner
    vis_group = parser_plot.add_argument_group("Visualization Toggles")

    vis_group.add_argument("-a", "--plot-all", action="store_true", help="Plot all results")
    vis_group.add_argument(
        "-pi", "--plot-illustrations", action="store_true", help="Plot illustrations"
    )

    # Single Run Plots
    vis_group.add_argument(
        "-pp", "--plot-params", action="store_true", help="Plot spectral parameter maps"
    )
    vis_group.add_argument(
        "-pt", "--plot-patches", action="store_true", help="Plot patch assignments"
    )
    vis_group.add_argument(
        "-ps", "--plot-cl-spectra", action="store_true", help="Plot spectra (one by one)"
    )
    vis_group.add_argument(
        "-pc", "--plot-cmb-recon", action="store_true", help="Plot CMB recon (one by one)"
    )
    vis_group.add_argument(
        "-psm",
        "--plot-systematic-maps",
        action="store_true",
        help="Plot systematic residuals (single)",
    )
    vis_group.add_argument(
        "-ptm",
        "--plot-statistical-maps",
        action="store_true",
        help="Plot statistical residuals (single)",
    )
    vis_group.add_argument(
        "-pr", "--plot-r-estimation", action="store_true", help="Plot R estimation (single)"
    )
    vis_group.add_argument(
        "-ppr",
        "--plot-params-residuals",
        action="store_true",
        help="Plot individual param maps + residuals vs truth (one per run)",
    )

    # Aggregate/Multi-Run Plots
    vis_group.add_argument(
        "-as", "--plot-all-spectra", action="store_true", help="Plot all spectra"
    )
    vis_group.add_argument(
        "-ar", "--plot-all-r-estimation", action="store_true", help="Plot R estimation comparison"
    )
    vis_group.add_argument(
        "-ah",
        "--plot-all-histograms",
        action="store_true",
        help="Plot histograms of True params and all pixels/realizations",
    )

    # Correlations
    vis_group.add_argument(
        "-arc", "--plot-r-vs-c", action="store_true", help="Plot r vs number of clusters"
    )
    vis_group.add_argument(
        "-avc", "--plot-v-vs-c", action="store_true", help="Plot variance vs number of clusters"
    )
    vis_group.add_argument(
        "-anlc", "--plot-nll-vs-c", action="store_true", help="Plot NLL vs number of clusters"
    )
    vis_group.add_argument("-arv", "--plot-r-vs-v", action="store_true", help="Plot r vs variance")

    # ==========================================
    # 3. VALIDATE SUBCOMMAND
    # ==========================================
    parser_validate = subparsers.add_parser(
        "validate", parents=[common_parser], help="Run NLL validation"
    )
    parser_validate.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Steps for validation perturbation (default: 5).",
    )
    parser_validate.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1e-3, 1e-4],
        help="Scales for validation perturbation (default: [1e-3, 1e-4]).",
    )
    parser_validate.add_argument(
        "--noise-selection",
        type=str,
        default="min-value",
        help="Noise realization selection: 'min-value' (default), 'min-nll', or an integer index.",
    )
    parser_validate.add_argument(
        "-t", "--titles", type=str, nargs="*", help="List of titles for the plots", default=None
    )
    parser_validate.add_argument(
        "--plot-type",
        type=str,
        nargs="+",
        default=["nll-grad"],
        help="Plot type: 'nll-grad' (default), 'nll', 'grad', or 'grad-maps-{idx}' (e.g., grad-maps-0).",
    )
    parser_validate.add_argument(
        "--perturb-beta-dust",
        type=str,
        default="all",
        help="Perturbation spec for beta_dust: 'all' (default), '-1' (skip), '0:30' (slice), '0,1,2' (indices), 'max', 'min'.",
    )
    parser_validate.add_argument(
        "--perturb-beta-pl",
        type=str,
        default="all",
        help="Perturbation spec for beta_pl: 'all' (default), '-1' (skip), '0:30' (slice), '0,1,2' (indices), 'max', 'min'.",
    )
    parser_validate.add_argument(
        "--perturb-temp-dust",
        type=str,
        default="all",
        help="Perturbation spec for temp_dust: 'all' (default), '-1' (skip), '0:30' (slice), '0,1,2' (indices), 'max', 'min'.",
    )
    parser_validate.add_argument(
        "--aggregate",
        action="store_true",
        help="Combine all runs on same plot (except grad-maps).",
    )
    parser_validate.add_argument(
        "--no-vmap",
        action="store_true",
        help="Use for-loop instead of vmap (slower but uses less memory).",
    )
    parser_validate.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf", "show"],
        default="png",
        help="Output format: png, pdf, or show (inline)",
    )
    parser_validate.add_argument(
        "-o", "--output", type=str, help="Output directory for plots", default="plots/"
    )
    parser_validate.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Font size for plots",
    )

    # ==========================================
    # 4. ESTIMATE SUBCOMMAND
    # ==========================================
    parser_estimate = subparsers.add_parser(
        "estimate", help="Estimate tensor-to-scalar ratio r from spectra or maps"
    )

    # Input data
    parser_estimate.add_argument(
        "--cmb",
        type=str,
        required=True,
        help="Path to CMB data (.npy): 1D spectrum C_ell, or 2D map (2, npix) for QU, or (3, npix) for IQU",
    )
    parser_estimate.add_argument(
        "--cmb-hat",
        type=str,
        help="Optional path to reconstructed CMB maps (.npy), shape (n_realizations, 2, npix) or (n_realizations, 3, npix)",
    )
    parser_estimate.add_argument(
        "--syst",
        type=str,
        help="Optional path to systematic residual map (.npy), shape (2, npix) or (3, npix)",
    )

    # Parameters
    parser_estimate.add_argument(
        "--fsky",
        type=float,
        help="Sky fraction (required if input is spectrum, inferred if input is map)",
    )
    parser_estimate.add_argument(
        "--nside",
        type=int,
        help="HEALPix resolution (inferred from map if not provided)",
    )

    # Output
    parser_estimate.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional path to save results (.npz format)",
    )
    parser_estimate.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf", "show"],
        default="png",
        help="Output format for plot: png, pdf, or show (inline)",
    )

    return parser.parse_args()
