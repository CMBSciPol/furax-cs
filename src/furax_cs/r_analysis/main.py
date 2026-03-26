import re

import datasets
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from furax_cs.data.instruments import get_instrument

from ..logging_utils import (
    error,
    warning,
)
from .compute import get_compute_flags
from .parser import parse_args
from .plotting import (
    get_plot_flags,
    run_grouped_plot,
)
from .r_estimate import run_estimate
from .run_grep import run_grep
from .snapshot import run_snapshot
from .validate import run_validate

out_folder = "plots/"


def run_analysis() -> int | None:
    """Entry point for the r_analysis CLI driver.

    This function orchestrates the complete analysis pipeline using subcommands:
    - snap: Compute statistics and save snapshot (auto-caches W_D_FG)
    - plot: Generate plots from results or snapshots
    - validate: Run NLL validation analysis
    - estimate: Estimate tensor-to-scalar ratio r from spectra or maps

    The analysis flow is controlled by command-line arguments parsed via :func:`parse_args`.
    """
    args = parse_args()

    # Handle estimate subcommand separately (doesn't use common_parser)
    if args.subcommand == "estimate":
        return run_estimate(
            cmb_path=args.cmb,
            cmb_hat_path=args.cmb_hat,
            syst_path=args.syst,
            fsky=args.fsky,
            nside=args.nside,
            output_path=args.output,
            output_format=args.output_format,
        )

    # Handle bin subcommand (uses common_parser, needs run_grep)
    if args.subcommand == "bin":
        from .binning import run_bin

        bin_config: dict[str, int] = {}
        if args.bin_bd is not None:
            bin_config["beta_dust"] = args.bin_bd
        if args.bin_td is not None:
            bin_config["temp_dust"] = args.bin_td
        if args.bin_bs is not None:
            bin_config["beta_pl"] = args.bin_bs
        if not bin_config:
            error("At least one --bin-* argument is required for bin.")
            return 1

        matched_results = run_grep(
            result_folders=args.input_results_dir,
            run_specs=args.runs,
        )
        if len(matched_results) == 0:
            error("No results matched the provided run specifications. Exiting.")
            return 1

        # Collect all folders from all matched groups
        all_folders: list[str] = []
        for _kw, (folders, _indices, _root) in matched_results.items():
            all_folders.extend(folders)

        return run_bin(
            folders=all_folders,
            nside=args.nside,
            output_dir=args.output_dir,
            bin_config=bin_config,
            noise_selection=args.noise_selection,
        )

    # run_grep only needed for snap/validate subcommands
    matched_results = None
    nside = None
    instrument = None
    if args.subcommand in ("snap", "validate"):
        nside = args.nside
        instrument = get_instrument(args.instrument)
        if args.no_tex:
            plt.rcParams["text.usetex"] = False

        matched_results = run_grep(
            result_folders=args.input_results_dir,
            run_specs=args.runs,
        )
        if len(matched_results) == 0:
            error("No results matched the provided run specifications. Exiting.")
            return

    if args.subcommand == "snap":
        assert matched_results is not None, "matched_results should be set for snap subcommand"
        assert (
            nside is not None and instrument is not None
        ), "nside and instrument should be set for snap subcommand"
        flags = get_compute_flags(args, snapshot_mode=True)
        combine_kw = None
        names = args.name
        if args.combine:
            combine_kw = args.name[0] if args.name else "COMBINED"
            names = None  # name is used as combine_kw, not per-entry

        return run_snapshot(
            matched_results,
            nside,
            instrument,
            args.output_parquet,
            flags,
            args.max_iterations,
            args.solver,
            noise_selection=args.noise_selection,
            sky_tag=args.sky,
            skip_images=args.no_images,
            max_ns=args.max_ns,
            combine_kw=combine_kw,
            names=names,
            max_size=args.max_size,
        )

    if args.subcommand == "plot":
        from pathlib import Path

        if args.no_tex:
            plt.rcParams["text.usetex"] = False

        parquet_dirs = [Path(d) for d in args.parquet_dir]
        all_parquets = sorted(p for d in parquet_dirs for p in d.glob("*.parquet"))
        if not all_parquets:
            error(f"No parquet files found in {args.parquet_dir}")
            return

        data_files = [str(p) for p in all_parquets]

        # Use pyarrow unified schema to handle parquets with different columns
        # (e.g. older parquets missing fg_nocmb_q/u added for binning support)
        import pyarrow as pa
        import pyarrow.dataset as pads
        import pyarrow.parquet as pq

        pa_schemas = [pq.read_schema(f) for f in data_files]
        unified_schema = pa.unify_schemas(pa_schemas)
        pa_table = pads.dataset(data_files, format="parquet", schema=unified_schema).to_table()
        ds = datasets.Dataset(pa_table).with_format("numpy")

        if args.runs:
            patterns = args.runs
            ds = ds.filter(
                lambda x: any(re.search(pat, str(x.get("name", x["kw"]))) for pat in patterns)
            )

            # Reorder rows to match -r pattern order (so -t titles align correctly)
            def _pattern_order(row):
                name = str(row.get("name", row["kw"]))
                for i, pat in enumerate(patterns):
                    if re.search(pat, name):
                        return i
                return len(patterns)

            order = [_pattern_order(ds[i]) for i in range(len(ds))]
            ds = ds.select(sorted(range(len(ds)), key=lambda i: order[i]))

        indiv_flags, aggregate_flags = get_plot_flags(args)

        # Build groups: if -g not given, one implicit group covering all filtered rows
        if args.groups:
            groups = [
                (
                    pattern,
                    ds.filter(lambda x, p=pattern: bool(re.search(p, str(x.get("name", x["kw"]))))),
                )
                for pattern in args.groups
            ]
        else:
            groups = [(".*", ds)]

        return run_grouped_plot(
            groups,
            indiv_flags,
            aggregate_flags,
            args.output_format,
            args.font_size,
            output_dir=args.output,
            group_titles=args.group_titles,
            row_titles=args.title,
            colors=args.color,
            xlim=args.xlim,
            r_legend_anchor=args.r_legend_anchor,
            s_legend_anchor=args.s_legend_anchor,
            r_figsize=args.r_figsize,
            s_figsize=args.s_figsize,
            r_range=args.r_range,
            r_plot=args.r_plot,
            transparent=args.transparent,
        )

    if args.subcommand == "validate":
        # Handle titles: if regex expanded to different number of groups, use expanded names
        assert matched_results is not None, "matched_results should be set for validate subcommand"
        assert (
            nside is not None and instrument is not None
        ), "nside and instrument should be set for validate subcommand"
        titles = args.titles
        if not titles or len(titles) != len(matched_results):
            if titles and len(titles) != len(matched_results):
                warning(
                    f"Got {len(matched_results)} result groups but {len(titles)} titles. "
                    f"Using expanded pattern names as titles."
                )
            titles = list(matched_results.keys())
        return run_validate(
            matched_results,
            titles,
            nside,
            instrument,
            args.steps,
            args.scales,
            plot_type=args.plot_type,
            perturb_beta_dust=args.perturb_beta_dust,
            perturb_beta_pl=args.perturb_beta_pl,
            perturb_temp_dust=args.perturb_temp_dust,
            noise_selection=args.noise_selection,
            aggregate=args.aggregate,
            use_vmap=not args.no_vmap,
            output_format=args.output_format,
            font_size=args.font_size,
            output_dir=args.output,
        )


def main() -> None:
    """CLI entry point for the r_analysis tool.

    This is the primary entry point registered as the ``r_analysis`` console script
    in pyproject.toml. It simply invokes :func:`run_analysis`.
    """
    run_analysis()


if __name__ == "__main__":
    run_analysis()
