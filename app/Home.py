"""Streamlit dashboard for browsing CMB component separation results."""

from __future__ import annotations

import re

import datasets
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET_REPO = "ASKabalan/furax-cs-results"
HF_CONFIGS = ("c1d0s0", "c1d1s1")

PLOTLY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

MAX_RUNS = 4

# Matches section_42 kw (with GAL) and section_41 kw (without GAL)
KW_PATTERN = re.compile(r"BD(?P<BD>\d+)_TD(?P<TD>\d+)_BS(?P<BS>\d+)(?:_GAL(?P<GAL>\d+))?")

# Image keys and their labels
PATCH_KEYS = [
    ("img_patches_beta_dust", "β_dust"),
    ("img_patches_temp_dust", "T_dust"),
    ("img_patches_beta_pl", "β_sync"),
]
PARAM_KEYS = [
    ("img_params_beta_dust", "β_dust"),
    ("img_params_temp_dust", "T_dust"),
    ("img_params_beta_pl", "β_sync"),
]

# Square plot size
PLOT_SIZE = 600

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading dataset from HuggingFace…")
def load_hf_dataset() -> datasets.Dataset:
    parts = []
    for cfg in HF_CONFIGS:
        ds = datasets.load_dataset(HF_DATASET_REPO, name=cfg, split="train")
        parts.append(ds)
    return datasets.concatenate_datasets(parts)


@st.cache_data(show_spinner="Building metadata index…")
def build_metadata(_ds: datasets.Dataset) -> list[dict]:
    """Extract lightweight metadata from every row for sidebar filtering."""
    rows = []
    for i in range(len(_ds)):
        row = _ds[i]
        kw = str(row["kw"])
        m = KW_PATTERN.search(kw)
        parsed = m.groupdict() if m else {}
        r_best = row.get("r_best")
        if r_best is not None:
            r_best = float(r_best)
            if r_best != r_best:  # NaN
                r_best = None
        sigma_neg = row.get("sigma_r_neg")
        if sigma_neg is not None:
            sigma_neg = float(sigma_neg)
            if sigma_neg != sigma_neg:
                sigma_neg = None
        sigma_pos = row.get("sigma_r_pos")
        if sigma_pos is not None:
            sigma_pos = float(sigma_pos)
            if sigma_pos != sigma_pos:
                sigma_pos = None
        fsky = row.get("fsky")
        if fsky is not None:
            fsky = float(fsky)
            if fsky != fsky:
                fsky = None
        rows.append(
            {
                "idx": i,
                "kw": kw,
                "sky_tag": str(row["sky_tag"]),
                "name": str(row.get("name", kw)),
                "BD": parsed.get("BD", ""),
                "TD": parsed.get("TD", ""),
                "BS": parsed.get("BS", ""),
                "GAL": parsed.get("GAL") or "",  # None → ""
                "r_best": r_best,
                "sigma_r_neg": sigma_neg,
                "sigma_r_pos": sigma_pos,
                "fsky": fsky,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arr(row: dict, key: str):
    v = row.get(key)
    if v is None:
        return None
    arr = np.asarray(v)
    return arr if arr.ndim > 0 else None


def _float_or_none(row: dict, key: str):
    v = row.get(key)
    if v is None:
        return None
    f = float(v)
    return None if f != f else f


def unique_sorted(values: list[str]) -> list[str]:
    """Return sorted unique non-empty strings, numerically if possible."""
    uniq = sorted(set(v for v in values if v))
    try:
        return sorted(uniq, key=int)
    except ValueError:
        return uniq


def cascading_options(meta: list[dict], filters: dict[str, str | None]) -> list[dict]:
    """Filter metadata rows by current filter selections (cascade)."""
    result = meta
    for key, val in filters.items():
        if val is not None:
            result = [r for r in result if r[key] == val]
    return result


def _reset_if_stale(key: str, valid_options: list[str]):
    """Delete session_state entry if its value is no longer in the valid options."""
    if key in st.session_state and st.session_state[key] not in valid_options:
        del st.session_state[key]


# ---------------------------------------------------------------------------
# Plotly layout helpers
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(size=16),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="grey",
        borderwidth=1,
        font=dict(size=13),
    ),
    margin=dict(l=60, r=20, t=30, b=60),
    height=PLOT_SIZE,
)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def build_r_likelihood_plot(runs: list[dict]) -> go.Figure:
    """Replicate plot_all_r_estimation as a Plotly figure."""
    fig = go.Figure()
    for i, run in enumerate(runs):
        row = run["row"]
        r_grid = _arr(row, "r_grid")
        L_vals = _arr(row, "L_vals")
        r_best = _float_or_none(row, "r_best")
        sigma_neg = _float_or_none(row, "sigma_r_neg")
        sigma_pos = _float_or_none(row, "sigma_r_pos")

        if r_grid is None or L_vals is None or r_best is None:
            continue

        color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        likelihood = L_vals / L_vals.max()
        name = run["meta"]["name"]

        label = f"{name} r̂ = {r_best:.2e}"
        if sigma_pos is not None and sigma_neg is not None:
            label += f" +{sigma_pos:.1e}/−{sigma_neg:.1e}"

        fig.add_trace(
            go.Scatter(
                x=r_grid,
                y=likelihood,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                legendgroup=name,
            )
        )

        if sigma_neg is not None and sigma_pos is not None:
            mask = (r_grid > r_best - sigma_neg) & (r_grid < r_best + sigma_pos)
            r_band = r_grid[mask]
            l_band = likelihood[mask]
            if len(r_band) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([r_band, r_band[::-1]]),
                        y=np.concatenate([l_band, np.zeros_like(l_band)]),
                        fill="toself",
                        fillcolor=color,
                        opacity=0.2,
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=name,
                        hoverinfo="skip",
                    )
                )

        fig.add_vline(x=r_best, line_dash="dash", line_color=color, opacity=0.7)

    fig.update_layout(
        xaxis_title="r",
        yaxis_title="Relative Likelihood",
        xaxis_range=[-0.002, 0.005],
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", griddash="dot")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", griddash="dot")
    return fig


def build_cl_residuals_plot(runs: list[dict]) -> go.Figure:
    """Replicate plot_all_cl_residuals as a Plotly figure."""
    fig = go.Figure()

    first_row = runs[0]["row"]
    cl_bb_r1 = _arr(first_row, "cl_bb_r1")
    ell_range = _arr(first_row, "ell_range")
    cl_bb_lens = _arr(first_row, "cl_bb_lens")

    if ell_range is None or cl_bb_r1 is None:
        return fig

    # Grey band: r in [1e-3, 4e-3]
    r_lo, r_hi = 1e-3, 4e-3
    y_lo = r_lo * cl_bb_r1
    y_hi = r_hi * cl_bb_r1

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([ell_range, ell_range[::-1]]),
            y=np.concatenate([y_hi, y_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(128,128,128,0.35)",
            line=dict(width=0),
            name="C_ℓ^BB, r∈[10⁻³, 4·10⁻³]",
            hoverinfo="skip",
        )
    )

    if cl_bb_lens is not None:
        fig.add_trace(
            go.Scatter(
                x=ell_range,
                y=cl_bb_lens,
                mode="lines",
                name="C_ℓ^BB lens",
                line=dict(color="grey", width=2),
            )
        )

    for i, run in enumerate(runs):
        row = run["row"]
        color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        name = run["meta"]["name"]

        cl_total = _arr(row, "cl_total_res")
        cl_syst = _arr(row, "cl_syst_res")
        cl_stat = _arr(row, "cl_stat_res")

        if cl_total is not None:
            fig.add_trace(
                go.Scatter(
                    x=ell_range,
                    y=cl_total,
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=f"{name} total<br>ℓ=%{{x}}<br>Cℓ=%{{y:.2e}}",
                )
            )
        if cl_syst is not None:
            fig.add_trace(
                go.Scatter(
                    x=ell_range,
                    y=cl_syst,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=f"{name} syst<br>ℓ=%{{x}}<br>Cℓ=%{{y:.2e}}",
                )
            )
        if cl_stat is not None:
            fig.add_trace(
                go.Scatter(
                    x=ell_range,
                    y=cl_stat,
                    mode="lines",
                    line=dict(color=color, dash="dot", width=1.5),
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=f"{name} stat<br>ℓ=%{{x}}<br>Cℓ=%{{y:.2e}}",
                )
            )

    # Style legend entries
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Total (C_ℓ^res)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="black"),
            name="Systematic (C_ℓ^syst)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="black", dash="dot"),
            name="Statistical (C_ℓ^stat)",
        )
    )
    for i, run in enumerate(runs):
        color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=2),
                name=run["meta"]["name"],
            )
        )

    fig.update_layout(
        xaxis_title="Multipole ℓ",
        yaxis_title="C_ℓ^BB [μK²]",
        xaxis_type="log",
        yaxis_type="log",
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", griddash="dash")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", griddash="dash")
    return fig


# ---------------------------------------------------------------------------
# Sidebar: run selection
# ---------------------------------------------------------------------------


def run_selector(
    slot: int,
    meta: list[dict],
) -> dict | None:
    """Render cascading filters in an expander and return selected metadata row."""
    with st.sidebar.expander(f"Run {slot + 1}", expanded=(slot == 0)):
        enabled = st.checkbox("Enable", value=(slot == 0), key=f"enable_{slot}")
        if not enabled:
            return None

        # Sky tag filter
        sky_tags = unique_sorted([r["sky_tag"] for r in meta])
        _reset_if_stale(f"sky_{slot}", sky_tags)
        sky_tag = st.selectbox("Sky tag", sky_tags, key=f"sky_{slot}")
        filtered = cascading_options(meta, {"sky_tag": sky_tag})

        # GAL filter — show option only when rows have GAL values
        gal_opts = unique_sorted([r["GAL"] for r in filtered])
        filters: dict[str, str | None] = {"sky_tag": sky_tag}
        if gal_opts:
            _reset_if_stale(f"gal_{slot}", gal_opts)
            gal = st.selectbox("GAL (mask)", gal_opts, key=f"gal_{slot}")
            filters["GAL"] = gal
        else:
            # Clean up stale GAL key when switching to a sky_tag without GAL
            if f"gal_{slot}" in st.session_state:
                del st.session_state[f"gal_{slot}"]
            st.selectbox(
                "GAL (mask)",
                ["ALL-GALACTIC"],
                disabled=True,
                key=f"gal_disabled_{slot}",
            )

        filtered = cascading_options(meta, filters)

        # BD filter
        bd_opts = unique_sorted([r["BD"] for r in filtered])
        _reset_if_stale(f"bd_{slot}", bd_opts)
        bd = st.selectbox("BD (K_β_dust)", bd_opts, key=f"bd_{slot}")
        filters["BD"] = bd
        filtered = cascading_options(meta, filters)

        # TD filter
        td_opts = unique_sorted([r["TD"] for r in filtered])
        _reset_if_stale(f"td_{slot}", td_opts)
        td = st.selectbox("TD (K_T_dust)", td_opts, key=f"td_{slot}")
        filters["TD"] = td
        filtered = cascading_options(meta, filters)

        # BS filter
        bs_opts = unique_sorted([r["BS"] for r in filtered])
        _reset_if_stale(f"bs_{slot}", bs_opts)
        bs = st.selectbox("BS (K_β_sync)", bs_opts, key=f"bs_{slot}")
        filters["BS"] = bs
        filtered = cascading_options(meta, filters)

        if not filtered:
            st.warning("No matching run")
            return None

        chosen = filtered[0]
        st.caption(f"**kw:** `{chosen['kw']}`")
        if chosen["r_best"] is not None:
            r = chosen["r_best"]
            sp = chosen["sigma_r_pos"]
            sn = chosen["sigma_r_neg"]
            sigma_str = ""
            if sp is not None and sn is not None:
                sigma_str = f" (+{sp:.1e}/−{sn:.1e})"
            st.caption(f"**r̂** = {r:.2e}{sigma_str}")
        if chosen["fsky"] is not None:
            st.caption(f"**fsky** = {chosen['fsky']:.3f}")
        return chosen


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------


def _render_image_rows(
    runs: list[dict],
    keys: list[tuple[str, str]],
    header: str,
):
    """Render image rows vertically: one row per parameter, sub-columns per run."""
    st.markdown(f"#### {header}")
    has_any = False
    for key, label in keys:
        imgs = []
        for run in runs:
            img = run["row_raw"].get(key)
            if img is not None:
                imgs.append((run["meta"]["name"], img))
        if imgs:
            has_any = True
            st.markdown(f"**{label}**")
            cols = st.columns(len(imgs))
            for col, (rname, img) in zip(cols, imgs):
                col.image(img, caption=rname, use_container_width=True)
    if not has_any:
        st.caption(f"No {header.lower()} images available.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="CMB CompSep Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("CMB Component Separation Results")

    st.markdown(
        """
This app lets you explore results from the
[FURAX-CS](https://github.com/CMBSciPol/furax-cs) GPU-accelerated CMB
parametric component separation pipeline.

**Data** — Two sky configurations are available: `c1d0s0` (CMB-only dust, no
synchrotron) and `c1d1s1` (CMB + dust + synchrotron).  Each row stores the
recovered power spectra, *r*-likelihood, patch/parameter maps, and residual
diagnostics for one clustering configuration.

**How to use** — Select up to **4 runs** in the sidebar by choosing a sky tag,
galactic mask, and patch counts (BD, TD, BS).  The main area shows patch &
parameter maps on the left and the *r*-likelihood and Cℓ residual plots on the
right, enabling side-by-side comparison across configurations.
        """
    )

    # --- Load data ---
    ds = load_hf_dataset()
    meta = build_metadata(ds)

    if not meta:
        st.error("Dataset is empty.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("Run Selection")

    # --- Select up to 4 runs ---
    selected: list[dict] = []
    for slot in range(MAX_RUNS):
        chosen_meta = run_selector(slot, meta)
        if chosen_meta is not None:
            ds_np = ds.with_format("numpy")
            row = ds_np[chosen_meta["idx"]]
            row_raw = ds[chosen_meta["idx"]]
            selected.append({"meta": chosen_meta, "row": row, "row_raw": row_raw})

    if not selected:
        st.info("Enable at least one run in the sidebar to view results.")
        return

    # --- Main area: images left, plots right ---
    col_img, col_plots = st.columns([1, 1])

    # --- Left column: images stacked vertically ---
    with col_img:
        _render_image_rows(selected, PATCH_KEYS, "Patches")
        st.markdown("---")
        _render_image_rows(selected, PARAM_KEYS, "Parameters")

    # --- Right column: square plots stacked ---
    with col_plots:
        st.markdown("#### r Likelihood")
        fig_r = build_r_likelihood_plot(selected)
        if fig_r.data:
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.caption("No r estimation data available.")

        st.markdown("---")

        st.markdown("#### Cℓ Residuals")
        fig_cl = build_cl_residuals_plot(selected)
        if len(fig_cl.data) > 2:
            st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.caption("No Cℓ residual data available.")


if __name__ == "__main__":
    main()
