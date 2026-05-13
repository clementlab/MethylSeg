"""Plotting and utility helpers shared across the public methylseg workflow."""

from enum import Enum
from itertools import permutations

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
from numba import njit


from .helper_classes import MethylationStates, SampleInfo


def get_biological_state_colors(cmap_name: str = "viridis"):
    """
    Return a fixed color mapping for the biological methylation states
    keyed by their canonical enum values (LOW=0, PMD=1, INTERMEDIATE=2, HIGH=3).
    """
    state_values = [state.value for state in MethylationStates]
    base_cmap = plt.get_cmap(cmap_name, len(state_values))
    state_colors_rgba = {
        state_value: base_cmap(idx) for idx, state_value in enumerate(state_values)
    }
    state_colors_hex = {
        state_value: mcolors.to_hex(color)
        for state_value, color in state_colors_rgba.items()
    }
    cmap = mcolors.ListedColormap(
        [state_colors_rgba[state_value] for state_value in state_values]
    )
    boundaries = np.arange(min(state_values) - 0.5, max(state_values) + 1.5, 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, state_colors_rgba, state_colors_hex


def get_present_biological_states(labels) -> list[int]:
    labels_numeric = MethylationStates.convert_to_numeric(labels)
    valid_state_values = {state.value for state in MethylationStates}
    return [
        int(state_value)
        for state_value in sorted(np.unique(labels_numeric))
        if int(state_value) in valid_state_values
    ]


def normalize_state_label(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, MethylationStates):
        return value.name
    if isinstance(value, Enum):
        return str(value.name)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped in MethylationStates.__members__:
            return stripped
        try:
            return MethylationStates(int(stripped)).name
        except (TypeError, ValueError):
            return stripped
    if isinstance(value, (int, np.integer)):
        try:
            return MethylationStates(int(value)).name
        except ValueError:
            return str(int(value))
    return str(value)


def annotate_plot_df_with_regions(
    df_plot: pd.DataFrame,
    regions_df: pd.DataFrame,
    *,
    chrom_col: str,
    pos_col: str,
    color_pmd_only: bool,
    region_label_col: str = "state",
) -> tuple[pd.DataFrame, str, dict[str, str], dict[str, list[str]], str, str]:
    plot_df = df_plot.copy()
    outside_region_color = "#7E7E7E"
    plot_df["__region_color__"] = "non-PMD" if color_pmd_only else "Outside regions"

    if regions_df is None or regions_df.empty:
        if color_pmd_only:
            plot_df["__region_color__"] = "non-PMD"
            return (
                plot_df,
                "__region_color__",
                {"PMD": "#d62728", "non-PMD": "#1f77b4"},
                {"__region_color__": ["PMD", "non-PMD"]},
                "PMD status",
                "PMD status",
            )
        return (
            plot_df,
            "__region_color__",
            {"Outside regions": outside_region_color},
            {"__region_color__": ["Outside regions"]},
            "Region state",
            "Region state",
        )

    required_cols = {"CpG_chrm", "start", "end", region_label_col}
    missing_cols = required_cols - set(regions_df.columns)
    if missing_cols:
        raise ValueError(
            "regions_df is missing required columns for coloring: "
            f"{sorted(missing_cols)}"
        )

    _, _, _, state_colors_hex = get_biological_state_colors()
    state_color_map = {
        state.name: state_colors_hex[state.value] for state in MethylationStates
    }
    region_df = regions_df.copy()
    region_df["CpG_chrm"] = region_df["CpG_chrm"].astype(str)
    region_df["start"] = pd.to_numeric(region_df["start"], errors="raise").astype(int)
    region_df["end"] = pd.to_numeric(region_df["end"], errors="raise").astype(int)
    region_df[region_label_col] = region_df[region_label_col].apply(
        normalize_state_label
    )
    region_df = region_df.sort_values(["CpG_chrm", "start", "end"]).reset_index(
        drop=True
    )

    for chrom, chrom_regions in region_df.groupby("CpG_chrm", sort=False):
        chrom_mask = plot_df[chrom_col].astype(str) == str(chrom)
        if not chrom_mask.any():
            continue
        chrom_indices = plot_df.index[chrom_mask].to_numpy()
        chrom_positions = plot_df.loc[chrom_mask, pos_col].to_numpy(dtype=np.int64)

        for region in chrom_regions.itertuples(index=False):
            region_mask = (chrom_positions >= int(region.start)) & (
                chrom_positions < int(region.end)
            )
            if not region_mask.any():
                continue
            if color_pmd_only:
                color_label = (
                    "PMD"
                    if normalize_state_label(getattr(region, region_label_col)) == "PMD"
                    else "non-PMD"
                )
            else:
                color_label = normalize_state_label(getattr(region, region_label_col))
                if color_label is None:
                    color_label = "Region"
            plot_df.loc[chrom_indices[region_mask], "__region_color__"] = color_label

    if color_pmd_only:
        return (
            plot_df,
            "__region_color__",
            {"PMD": "#d62728", "non-PMD": "#1f77b4"},
            {"__region_color__": ["PMD", "non-PMD"]},
            "PMD status",
            "PMD status",
        )

    present_labels = plot_df["__region_color__"].dropna().astype(str).unique().tolist()
    ordered_labels = [
        state.name for state in MethylationStates if state.name in present_labels
    ]
    if "Outside regions" in present_labels:
        ordered_labels.append("Outside regions")
    for label in present_labels:
        if label not in ordered_labels:
            ordered_labels.append(label)

    color_map = {
        label: state_color_map.get(label, "#9e9e9e") for label in ordered_labels
    }
    color_map["Outside regions"] = outside_region_color
    return (
        plot_df,
        "__region_color__",
        color_map,
        {"__region_color__": ordered_labels},
        "Region state",
        "Region state",
    )


def plot_interactive_beta_scatter(
    *,
    df_plot: pd.DataFrame,
    sample_info: SampleInfo | None,
    sample_info_removed: pd.DataFrame | None,
    chrom: str | None,
    out_dir: str | None,
    label_col: str,
    x_col: str = "CpG_beg",
    y_col: str = "beta",
    label_title: str | None = None,
    show_plot: bool = True,
    max_points: int = 120_000,
    color_pmd_only: bool = False,
    color_regions_df: pd.DataFrame | None = None,
    region_label_col: str = "state",
) -> object | None:
    df_plot = df_plot.copy()
    df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
    removed_plot = None

    if sample_info_removed is not None:
        removed_plot = sample_info_removed.copy()
        removed_plot = removed_plot.loc[:, ~removed_plot.columns.duplicated()]

    if chrom is not None and "CpG_chrm" in df_plot.columns:
        df_plot = df_plot[df_plot["CpG_chrm"] == chrom]
        if removed_plot is not None and "CpG_chrm" in removed_plot.columns:
            removed_plot = removed_plot[removed_plot["CpG_chrm"] == chrom]

    if df_plot.empty:
        print("[INFO] No data to plot.")
        return None

    df_plot = df_plot.sort_values(x_col).reset_index(drop=True)

    if removed_plot is not None and not removed_plot.empty:
        required_removed_cols = {"CpG_chrm", x_col, "beta"}
        missing_removed_cols = required_removed_cols - set(removed_plot.columns)
        if missing_removed_cols:
            raise ValueError(
                "sample_info_removed is missing required columns: "
                f"{sorted(missing_removed_cols)}"
            )
        removed_plot = removed_plot.sort_values(x_col).reset_index(drop=True)

    retained_n = len(df_plot)
    removed_n = 0 if removed_plot is None else len(removed_plot)
    total_n = retained_n + removed_n
    downsampled = total_n > max_points

    if downsampled:
        rng = np.random.default_rng(42)

        if removed_n == 0:
            retained_keep = max_points
            removed_keep = 0
        else:
            retained_keep = int(round(max_points * retained_n / total_n))
            retained_keep = max(1, min(retained_keep, retained_n))
            removed_keep = max_points - retained_keep
            removed_keep = min(removed_keep, removed_n)

            leftover = max_points - (retained_keep + removed_keep)
            if leftover > 0:
                retained_room = retained_n - retained_keep
                retained_add = min(leftover, retained_room)
                retained_keep += retained_add
                leftover -= retained_add

            if leftover > 0:
                removed_room = removed_n - removed_keep
                removed_add = min(leftover, removed_room)
                removed_keep += removed_add

        if retained_n > retained_keep:
            keep_idx = np.sort(
                rng.choice(retained_n, size=retained_keep, replace=False)
            )
            df_plot = df_plot.iloc[keep_idx].reset_index(drop=True)

        if removed_plot is not None and removed_n > removed_keep:
            keep_idx = np.sort(rng.choice(removed_n, size=removed_keep, replace=False))
            removed_plot = removed_plot.iloc[keep_idx].reset_index(drop=True)

    if color_regions_df is not None:
        (
            df_plot,
            plot_color_col,
            color_map,
            category_orders,
            color_label,
            legend_title,
        ) = annotate_plot_df_with_regions(
            df_plot=df_plot,
            regions_df=color_regions_df,
            chrom_col="CpG_chrm",
            pos_col=x_col,
            color_pmd_only=color_pmd_only,
            region_label_col=region_label_col,
        )
    else:
        if label_col not in df_plot.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in plotting DataFrame."
            )
        if isinstance(df_plot[label_col].iloc[0], Enum):
            df_plot[label_col] = df_plot[label_col].apply(lambda x: x.value)
        df_plot[label_col] = df_plot[label_col].astype(int)

        if color_pmd_only:
            plot_color_col = f"{label_col}_pmd_status"
            df_plot[plot_color_col] = np.where(
                df_plot[label_col] == MethylationStates.PMD.value,
                "PMD",
                "non-PMD",
            )
            color_map = {"PMD": "#d62728", "non-PMD": "#1f77b4"}
            category_orders = {plot_color_col: ["PMD", "non-PMD"]}
            color_label = "PMD status"
            legend_title = "PMD status"
        else:
            df_plot[label_col] = df_plot[label_col].astype(str)
            plot_color_col = label_col
            _, _, _, state_colors_hex = get_biological_state_colors()
            present_state_values = get_present_biological_states(
                df_plot[label_col].astype(int).to_numpy()
            )
            color_map = {
                str(state_value): state_colors_hex[state_value]
                for state_value in present_state_values
            }
            category_orders = {
                plot_color_col: [
                    str(state_value) for state_value in present_state_values
                ]
            }
            color_label = label_title if label_title is not None else label_col
            legend_title = "State"

    title_parts = []
    if sample_info is not None:
        title_parts.append(str(sample_info.sample_id))
    if chrom is not None:
        title_parts.append(str(chrom))
    title_prefix = " ".join(title_parts) if title_parts else "Sample"
    plot_title = label_title if label_title is not None else label_col

    scatter_kwargs = {
        "data_frame": df_plot,
        "x": x_col,
        "y": y_col,
        "color": plot_color_col,
        "color_discrete_map": color_map,
        "labels": {
            x_col: "Genomic Position",
            y_col: "Methylation (beta)",
            plot_color_col: color_label,
        },
        "title": (
            f"{title_prefix}: Methylation Beta by {plot_title} "
            f"({'downsampled' if downsampled else 'full'})"
        ),
    }
    if category_orders is not None:
        scatter_kwargs["category_orders"] = category_orders

    fig = px.scatter(**scatter_kwargs)
    fig.update_traces(marker=dict(size=4, opacity=0.8))

    if removed_plot is not None and not removed_plot.empty:
        fig.add_trace(
            go.Scattergl(
                x=removed_plot[x_col],
                y=removed_plot[y_col],
                mode="markers",
                name="Removed CpGs",
                marker=dict(size=4, color="#d3d3d3", opacity=0.35),
                hovertemplate=(
                    "status: removed<br>"
                    "pos: %{x}<br>"
                    "beta: %{y:.3f}<extra></extra>"
                ),
            )
        )
        fig.data = (fig.data[-1],) + fig.data[:-1]

    fig.update_layout(
        legend_title_text=legend_title,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    if color_regions_df is None and not color_pmd_only:
        state_names = {str(s.value): s.name for s in MethylationStates}
        fig.for_each_trace(lambda t: t.update(name=state_names.get(t.name, t.name)))

    if show_plot:
        fig.show(renderer="notebook")

    if out_dir is not None:
        suffix = "_pmd_only" if color_pmd_only else ""
        if color_regions_df is not None:
            suffix += "_region_coloring"
        fig.write_html(f"{out_dir}/interactive_beta_by_{label_col}{suffix}.html")

    return fig


@njit
def build_emission_matrix_numba(
    positions,
    betas,
    window_sizes,
    int_low_cutoff,
    int_high_cutoff,
    high_cutoff,
):
    n = len(betas)
    n_windows = len(window_sizes)

    # Precompute masks
    low_mask = betas < int_low_cutoff
    int_mask = (betas >= int_low_cutoff) & (betas <= int_high_cutoff)
    high_mask = betas > high_cutoff

    beta_cumsum = np.cumsum(betas)
    beta_sq_cumsum = np.cumsum(betas * betas)
    low_cumsum = np.cumsum(low_mask.astype(np.int64))
    int_cumsum = np.cumsum(int_mask.astype(np.int64))
    high_cumsum = np.cumsum(high_mask.astype(np.int64))

    # 1 beta + 6 features per window
    n_features = 1 + 6 * n_windows
    X = np.zeros((n, n_features))

    # First column = raw beta
    X[:, 0] = betas

    feature_col = 1

    for w in range(n_windows):
        window_size = window_sizes[w]

        avg = np.zeros(n)
        std = np.zeros(n)
        high_pct = np.zeros(n)
        int_pct = np.zeros(n)
        low_pct = np.zeros(n)
        n_cpg = np.zeros(n)

        left = 0
        right = 0

        for i in range(n):

            center = positions[i]
            w_start = center - window_size // 2
            w_end = center + window_size // 2

            while left < n and positions[left] < w_start:
                left += 1

            while right + 1 < n and positions[right + 1] <= w_end:
                right += 1

            count = right - left + 1
            if count <= 0:
                continue

            if left == 0:
                sum_beta = beta_cumsum[right]
                sum_sq = beta_sq_cumsum[right]
                sum_low = low_cumsum[right]
                sum_int = int_cumsum[right]
                sum_high = high_cumsum[right]
            else:
                sum_beta = beta_cumsum[right] - beta_cumsum[left - 1]
                sum_sq = beta_sq_cumsum[right] - beta_sq_cumsum[left - 1]
                sum_low = low_cumsum[right] - low_cumsum[left - 1]
                sum_int = int_cumsum[right] - int_cumsum[left - 1]
                sum_high = high_cumsum[right] - high_cumsum[left - 1]

            mean = sum_beta / count
            var = (sum_sq / count) - mean * mean
            if var < 0.0:
                var = 0.0

            avg[i] = mean
            std[i] = np.sqrt(var)
            high_pct[i] = sum_high / count
            int_pct[i] = sum_int / count
            low_pct[i] = sum_low / count
            n_cpg[i] = count

        X[:, feature_col] = avg
        X[:, feature_col + 1] = std
        X[:, feature_col + 2] = high_pct
        X[:, feature_col + 3] = int_pct
        X[:, feature_col + 4] = low_pct
        X[:, feature_col + 5] = n_cpg

        feature_col += 6

    return X


def get_cluster_colors(n_states: int, cmap_name: str = "viridis"):
    """
    Return a discrete colormap, norm, and per-state hex colors
    such that state k always uses the k-th color.

    States are assumed to be integers 0..n_states-1.
    """
    # Discrete colormap with n_states entries
    cmap = plt.get_cmap(cmap_name, n_states)

    # Norm so that integer k maps to the k-th color
    boundaries = np.arange(-0.5, n_states + 0.5, 1)
    norm = mcolors.BoundaryNorm(boundaries, n_states)
    # Colors in numeric state order, as hex (for Plotly) and RGBA (for Matplotlib)
    state_colors_rgba = [cmap(k) for k in range(n_states)]
    state_colors_hex = [mcolors.to_hex(c) for c in state_colors_rgba]

    return cmap, norm, state_colors_rgba, state_colors_hex


def absorb_small_clusters(
    raw_labels: np.ndarray,
    emission_df: pd.DataFrame,
    min_frac: float = 0.001,
) -> np.ndarray:
    """
    Absorb clusters smaller than min_frac of total CpGs
    into nearest larger cluster (by mean beta).
    """

    labels = np.asarray(raw_labels).copy()
    unique = np.unique(labels)

    total = len(labels)
    cluster_sizes = {c: np.sum(labels == c) for c in unique}

    # Identify large clusters
    large_clusters = [c for c in unique if cluster_sizes[c] / total >= min_frac]

    # If all clusters are large, return unchanged
    if len(large_clusters) == len(unique):
        return labels

    beta_vals = emission_df["beta"].to_numpy()

    # Compute mean beta for each cluster
    cluster_means = {c: beta_vals[labels == c].mean() for c in unique}

    # Absorb small clusters
    for c in unique:
        if c in large_clusters:
            continue

        # Find nearest large cluster in beta space
        small_mean = cluster_means[c]

        nearest = min(
            large_clusters,
            key=lambda lc: abs(cluster_means[lc] - small_mean),
        )

        labels[labels == c] = nearest

    return labels


def get_regional_window_labels(window_specs) -> List[str]:
    sorted_window_specs = sorted(window_specs, key=lambda item: item[0])
    if len(sorted_window_specs) <= 2:
        return [label for _, label in sorted_window_specs]
    return [label for _, label in sorted_window_specs[-2:]]


def relabel_by_mean_emission(
    raw_labels: np.ndarray,
    emission_df: pd.DataFrame,
    state_cutoffs: Optional[Dict[str, object]] = None,
    int_low_cutoff: float = 0.2,
    int_high_cutoff: float = 0.7,
    window_specs: List[Tuple[int, str]] = [(40_000, "40kb"), (450_000, "450kb")],
) -> np.ndarray:
    labels = np.asarray(absorb_small_clusters(raw_labels, emission_df))
    clusters = np.unique(labels)

    if len(clusters) == 0:
        return np.asarray(labels, dtype=object)

    beta_min = (
        int_low_cutoff
        if state_cutoffs is None
        else state_cutoffs.get("beta_low_max", int_low_cutoff)
    )
    beta_max = (
        int_high_cutoff
        if state_cutoffs is None
        else state_cutoffs.get("beta_high_min", int_high_cutoff)
    )

    regional_window_labels = get_regional_window_labels(window_specs)

    def regional_mean(cluster, suffix):
        mask = labels == cluster
        return float(
            np.mean(
                [
                    emission_df[f"{w}_{suffix}"].to_numpy()[mask].mean()
                    for w in regional_window_labels
                ]
            )
        )

    def cluster_mean(cluster, col):
        mask = labels == cluster
        return float(emission_df[col].to_numpy()[mask].mean())

    # -----------------------------
    # Compute stats
    # -----------------------------
    stats = {
        c: {
            "beta": cluster_mean(c, "beta"),
            "intermediate": regional_mean(c, "int_pct"),
            "high": regional_mean(c, "high_pct"),
            "low": regional_mean(c, "low_pct"),
        }
        for c in clusters
    }

    beta_mid = (beta_min + beta_max) / 2.0
    beta_span = max(beta_max - beta_min, 1e-6)

    def beta_mid_score(beta: float) -> float:
        return max(0.0, 1.0 - (abs(beta - beta_mid) / beta_span))

    def low_score(cluster) -> float:
        s = stats[cluster]
        return (
            (2.0 * s["low"])
            + (1.0 - s["beta"])
            - (0.5 * s["intermediate"])
            - (0.75 * s["high"])
        )

    def high_score(cluster) -> float:
        s = stats[cluster]
        return (
            (2.0 * s["high"])
            + s["beta"]
            - (0.5 * s["intermediate"])
            - (0.75 * s["low"])
        )

    def pmd_score(cluster) -> float:
        s = stats[cluster]
        return (
            (3.0 * s["intermediate"])
            + (1.0 * s["low"])
            - (1.5 * s["high"])
            + beta_mid_score(s["beta"])
        )

    def intermediate_score(cluster) -> float:
        s = stats[cluster]
        return (
            (2.0 * s["intermediate"])
            + (1.0 * s["high"])
            - (1.0 * s["low"])
            + beta_mid_score(s["beta"])
        )

    def state_score(cluster, state):
        if state == MethylationStates.LOW:
            return low_score(cluster)
        if state == MethylationStates.PMD:
            return pmd_score(cluster)
        if state == MethylationStates.INTERMEDIATE:
            return intermediate_score(cluster)
        if state == MethylationStates.HIGH:
            return high_score(cluster)
        raise ValueError(f"Unknown state: {state}")

    candidate_states = [
        MethylationStates.LOW,
        MethylationStates.PMD,
        MethylationStates.INTERMEDIATE,
        MethylationStates.HIGH,
    ]

    # -----------------------------
    # Assign the most meaningful label(s) uniquely
    # -----------------------------
    best_assignment = None
    best_key = None

    for assignment in permutations(candidate_states, len(clusters)):
        score_vector = tuple(state_score(c, s) for c, s in zip(clusters, assignment))
        total_score = float(np.sum(score_vector))
        key = (total_score, score_vector)

        if best_key is None or key > best_key:
            best_key = key
            best_assignment = assignment

    mapping = {c: s for c, s in zip(clusters, best_assignment)}

    # -----------------------------
    # Apply mapping
    # -----------------------------
    new_labels = np.empty(labels.shape, dtype=object)
    for c, state in mapping.items():
        new_labels[labels == c] = state

    return new_labels
