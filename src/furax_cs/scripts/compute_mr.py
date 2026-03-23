import argparse

import jax.numpy as jnp
from furax_cs import get_mask
from furax_cs.multires_clusters import multires_clusters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count multi-resolution clusters for a given mask and target nside.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  compute-mr --mask GAL040 --nside 256 --target-nside 8
  compute-mr --mask GAL020 --nside 128 --target-nside 4 8 4
  compute-mr --mask GAL060 --nside 64  --target-nside 0      # single global patch
""",
    )
    parser.add_argument("--mask", "-m", required=True, help="Mask name (e.g. GAL020, GAL040)")
    parser.add_argument("--nside", "-n", type=int, required=True, help="HEALPix nside of the map")
    parser.add_argument(
        "--target-nside",
        "-t",
        type=int,
        nargs="+",
        required=True,
        metavar="NSIDE",
        help=(
            "Target nside for multires patches. "
            "Give 1 value (used for all 3 params) or 3 values: beta_dust temp_dust beta_pl. "
            "Use 0 for a single global patch."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.target_nside) == 1:
        t = args.target_nside[0]
        target_ud_grade = {"beta_dust": t, "temp_dust": t, "beta_pl": t}
    elif len(args.target_nside) == 3:
        target_ud_grade = {
            "beta_dust": args.target_nside[0],
            "temp_dust": args.target_nside[1],
            "beta_pl": args.target_nside[2],
        }
    else:
        raise ValueError("--target-nside expects 1 or 3 values")

    mask = get_mask(args.mask, args.nside)
    (indices,) = jnp.where(mask == 1)
    patch_indices = multires_clusters(mask, indices, target_ud_grade, nside=args.nside)

    print(f"Mask: {args.mask}  |  nside: {args.nside}  |  valid pixels: {int(indices.size)}\n")

    for key, patches in patch_indices.items():
        param = key.replace("_patches", "")
        t_nside = target_ud_grade[param]
        n_clusters = int(jnp.unique(patches).size)

        if t_nside == 0:
            full_count = int(indices.size)  # single global patch
        else:
            full_count = (args.nside // t_nside) ** 2

        counts = jnp.bincount(patches, length=n_clusters)
        n_incomplete = int(jnp.sum(counts < full_count))

        print(f"[{param}]  target_nside={t_nside}")
        print(f"  Number of clusters   : {n_clusters}")
        print(f"  Full patch size      : {full_count} px")
        print(f"  Incomplete patches   : {n_incomplete}")
        print()
