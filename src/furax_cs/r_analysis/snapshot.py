import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..logging_utils import info, success, warning
from .compute import compute_all

SNAPSHOT_MANIFEST_NAME = "manifest.json"
SNAPSHOT_VERSION = 1


def _snapshot_filename_from_title(title: str) -> str:
    """Generate a stable filename slug for a snapshot entry."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    if not slug:
        slug = "entry"
    return f"{slug}_{digest}.pkl"


# Over kill helper?
def _tree_to_numpy(tree: Any) -> Any:
    """Convert JAX arrays to numpy arrays recursively."""

    def _convert_leaf(x):
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "__array__"):
            return np.asarray(x)
        return x

    return jax.tree.map(_convert_leaf, tree)


# Over kill helper?
def _tree_to_jax(tree: Any) -> Any:
    """Convert numpy arrays to JAX arrays recursively."""

    def _convert_leaf(x):
        if isinstance(x, np.ndarray):
            return jnp.asarray(x)
        return x

    return jax.tree.map(_convert_leaf, tree)


def load_snapshot(
    snapshot_dir: str,
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """Load cached analysis payloads from disk snapshots."""
    snapshot_path = Path(snapshot_dir)
    manifest_path = snapshot_path / SNAPSHOT_MANIFEST_NAME

    if not manifest_path.exists():
        return [], {"version": SNAPSHOT_VERSION, "entries": []}

    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    entries = []
    for item in manifest.get("entries", []):
        title = item.get("title")
        filename = item.get("file")
        if title is None or filename is None:
            continue
        payload_path = snapshot_path / filename
        if not payload_path.exists():
            warning(f"Snapshot payload missing for '{title}' at {payload_path}")
            continue
        with payload_path.open("rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict):
            converted_payload = {}
            for key in ("cmb", "cl", "r", "residual", "plotting_data"):
                value = payload.get(key)
                if value is not None:
                    converted_payload[key] = _tree_to_jax(value)
                else:
                    converted_payload[key] = value
            payload = converted_payload
        entries.append((title, payload))

    return entries, manifest


def save_snapshot_entry(
    snapshot_dir: Union[str, Path],
    manifest: dict[str, Any],
    title: str,
    payload: Any,
) -> dict[str, Any]:
    """Persist a single snapshot payload and update the manifest."""
    snapshot_path = Path(snapshot_dir)
    snapshot_path.mkdir(parents=True, exist_ok=True)

    entries = manifest.setdefault("entries", [])
    lookup = {item["title"]: item for item in entries if "title" in item}

    existing_entry = lookup.get(title)
    filename = None
    if existing_entry is not None:
        filename = existing_entry.get("file")
    if not filename:
        filename = _snapshot_filename_from_title(title)

    payload_path = snapshot_path / filename
    with payload_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if existing_entry is not None:
        existing_entry["file"] = filename
    else:
        entries.append({"title": title, "file": filename})

    manifest["version"] = SNAPSHOT_VERSION
    manifest["entries"] = entries
    return manifest


def write_snapshot_manifest(snapshot_dir: Union[str, Path], manifest: dict[str, Any]) -> None:
    """Write the manifest JSON for stored snapshot entries."""
    snapshot_path = Path(snapshot_dir)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_path / SNAPSHOT_MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)


def load_and_filter_snapshot(
    snapshot_path: Union[str, Path] | None, matched_results: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load snapshot and filter matched_results.

    Snapshots are indexed by kw (run spec), not title. This is more stable
    since titles are just for display and may change.

    Parameters
    ----------
    snapshot_path : str or Path
        Path to snapshot directory.
    matched_results : dict
        Format: {kw: (folders, indices)} from run_grep()

    Returns
    -------
    existing_results : dict
        {kw: (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data)}
        Results already in snapshot, keyed by kw
    to_compute : dict
        {kw: (folders, indices)} - subset of matched_results not in snapshot
    """
    # Load existing snapshot (keyed by kw)
    if snapshot_path and Path(snapshot_path).exists():
        entries, _ = load_snapshot(snapshot_path)
        snapshot_store = {kw: payload for kw, payload in entries}
    else:
        snapshot_store = {}

    existing_results = {}
    to_compute = {}

    for kw, (folders, indices, root) in matched_results.items():
        if kw in snapshot_store:
            # Already computed
            payload = snapshot_store[kw]
            existing_results[kw] = {
                "cmb": payload["cmb"],
                "cl": payload["cl"],
                "r": payload["r"],
                "residual": payload["residual"],
                "plotting_data": payload["plotting_data"],
            }
        else:
            # Need to compute
            to_compute[kw] = (folders, indices, root)

    return existing_results, to_compute


def serialize_snapshot_payload(result: tuple[Any, ...]) -> dict[str, Any]:
    """Serialize snapshot payload to numpy arrays for storage.

    Parameters
    ----------
    result : tuple
        (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data)
    Returns
    -------
    payload : dict
        Dictionary with numpy arrays for storage.
    """
    cmb, cl, r, residual, plotting = result
    payload = {
        "cmb": _tree_to_numpy(cmb),
        "cl": _tree_to_numpy(cl),
        "r": _tree_to_numpy(r),
        "residual": _tree_to_numpy(residual),
        "plotting_data": _tree_to_numpy(plotting),
    }
    return payload


def save_snapshot(snapshot_path: Union[str, Path], results: dict[str, Any]) -> None:
    """Save computed results to snapshot directory.

    Results are indexed by kw (run spec), not title.

    Parameters
    ----------
    snapshot_path : str or Path
        Path to snapshot directory.
    results : dict
        {kw: (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data)}
        from compute_all(), keyed by kw
    """
    snapshot_path_obj = Path(snapshot_path)

    # Load existing manifest or create new
    manifest_path = snapshot_path_obj / SNAPSHOT_MANIFEST_NAME
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"version": SNAPSHOT_VERSION, "entries": []}

    # Save each result (keyed by kw)
    for kw, res in results.items():
        payload = serialize_snapshot_payload(res)
        # save_snapshot_entry uses kw as the "title" field in manifest
        manifest = save_snapshot_entry(snapshot_path_obj, manifest, kw, payload)

    write_snapshot_manifest(snapshot_path_obj, manifest)


def run_snapshot(
    matched_results: dict[str, Any],
    nside: int,
    instrument: Any,
    snapshot_path: str,
    flags: dict[str, bool],
    max_iter: int,
    solver_name: str,
    noise_selection: str,
) -> int:
    """Entry point for 'snap' subcommand.

    Computes statistics for matched runs and saves to snapshot directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with output_snapshot path.
    matched_results : dict
        Format: {kw: (folders, indices)} from run_grep()
    nside : int
        HEALPix resolution parameter.
    instrument : FGBusterInstrument
        Instrument configuration object.
    """

    existing, to_compute = load_and_filter_snapshot(snapshot_path, matched_results)

    if existing:
        info(f"Found {len(existing)} existing entries in snapshot")

    if to_compute:
        info(f"Computing {len(to_compute)} missing entries...")
        computed = compute_all(
            to_compute,
            nside,
            instrument,
            flags,
            max_iter,
            solver_name,
            noise_selection=noise_selection,
        )
        save_snapshot(snapshot_path, computed)
        success(f"Saved {len(computed)} new entries to snapshot at {snapshot_path}")
    else:
        info("All results already in snapshot, nothing to compute")

    return 0
