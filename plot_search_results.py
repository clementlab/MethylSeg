from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

matplotlib.use("Agg")

from matplotlib import pyplot as plt


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_SEARCH_RESULTS = CODE_DIR / "test_hyperparameter_runs" / "search_results.csv"
DEFAULT_OUTPUT_DIR_NAME = "comparison_plots"
HMM_TYPE_PALETTE = {
    "ct": "#457b9d",
    "sticky": "#e76f51",
}
DISPLAY_NAMES = {
    "pmr_jaccard_bp": "PMR Jaccard (bp)",
    "pmr_probe_f1": "PMR Probe F1",
    "pmr_probe_recall": "PMR Probe Recall",
    "pmr_probe_precision": "PMR Probe Precision",
    "pmr_bp_coverage": "PMR bp Coverage",
    "weighted_state_f1": "Weighted State F1",
    "pmr_region_count": "PMR Region Count",
    "pmr_mean_region_bp": "Mean PMR Region Length (bp)",
    "pmr_median_region_bp": "Median PMR Region Length (bp)",
    "pmr_short_region_count": "Short PMR Region Count",
    "pmr_short_region_fraction": "Short PMR Region Fraction",
    "window_count": "Window Count",
    "holding_time_guess": "Holding Time Guess",
    "rank": "Rank",
}
HIGHER_IS_BETTER = {
    "pmr_jaccard_bp",
    "pmr_probe_f1",
    "pmr_probe_recall",
    "pmr_probe_precision",
    "pmr_bp_coverage",
    "weighted_state_f1",
    "pmr_mean_region_bp",
    "pmr_median_region_bp",
}
LOWER_IS_BETTER = {
    "pmr_short_region_fraction",
    "pmr_short_region_count",
    "pmr_region_count",
    "rank",
}


def _display_name(metric: str) -> str:
    return DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())


def _load_results(search_results_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(search_results_path)
    for column in results.columns:
        if column in {"candidate_id", "hmm_type", "window_labels", "window_sizes_bp"}:
            continue
        converted = pd.to_numeric(results[column], errors="coerce")
        if converted.notna().any() or pd.api.types.is_numeric_dtype(results[column]):
            results[column] = converted

    successful = results.copy()
    if "status" in successful.columns:
        successful = successful[successful["status"] == "success"].copy()
    if successful.empty:
        raise ValueError(f"No successful candidates found in {search_results_path}.")

    if "rank" in successful.columns and successful["rank"].notna().any():
        successful = successful.sort_values(["rank", "candidate_id"], na_position="last")
    else:
        sort_columns: list[str] = []
        ascending: list[bool] = []
        for metric in ("pmr_jaccard_bp", "pmr_probe_f1"):
            if metric in successful.columns:
                sort_columns.append(metric)
                ascending.append(False)
        if "candidate_id" in successful.columns:
            sort_columns.append("candidate_id")
            ascending.append(True)
        if sort_columns:
            successful = successful.sort_values(sort_columns, ascending=ascending)

    successful = successful.reset_index(drop=True)
    successful["hmm_type_label"] = successful["hmm_type"].map(
        {"ct": "cthmm", "sticky": "sticky"}
    ).fillna(successful["hmm_type"])
    return results, successful


def _available_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [
        metric
        for metric in metrics
        if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().any()
    ]


def _minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    min_value = valid.min()
    max_value = valid.max()
    if math.isclose(float(min_value), float(max_value)):
        scaled = pd.Series(1.0, index=valid.index, dtype=float)
    else:
        scaled = (valid - min_value) / (max_value - min_value)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[valid.index] = scaled
    return out


def _normalized_heatmap_frame(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    normalized = pd.DataFrame(index=df["candidate_id"])
    for metric in metrics:
        scaled = _minmax_scale(df[metric])
        if metric in LOWER_IS_BETTER:
            scaled = 1.0 - scaled
        normalized[_display_name(metric)] = scaled.to_numpy()
    return normalized


def _annotate_top_candidates(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    top_k: int,
) -> None:
    if top_k <= 0 or "rank" not in df.columns:
        return
    top = df.nsmallest(min(top_k, len(df)), "rank")
    for _, row in top.iterrows():
        x_val = row.get(x_metric)
        y_val = row.get(y_metric)
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        ax.annotate(
            row["candidate_id"],
            (x_val, y_val),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )


def _write_summary_tables(
    all_results: pd.DataFrame,
    successful: pd.DataFrame,
    out_dir: Path,
    top_n: int,
) -> dict[str, str]:
    table_paths: dict[str, str] = {}

    top_columns = [
        column
        for column in (
            "rank",
            "candidate_id",
            "hmm_type",
            "holding_time_guess",
            "window_count",
            "window_labels",
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_probe_recall",
            "pmr_bp_coverage",
            "pmr_short_region_fraction",
            "pmr_mean_region_bp",
            "pmr_median_region_bp",
            "weighted_state_f1",
        )
        if column in successful.columns
    ]
    top_path = out_dir / "top_candidates_metrics.csv"
    successful.head(top_n).loc[:, top_columns].to_csv(top_path, index=False)
    table_paths["top_candidates_metrics"] = str(top_path)

    summary_metrics = _available_metrics(
        successful,
        [
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_probe_recall",
            "pmr_bp_coverage",
            "pmr_short_region_fraction",
            "pmr_mean_region_bp",
            "pmr_median_region_bp",
            "weighted_state_f1",
        ],
    )
    if summary_metrics and "hmm_type" in successful.columns:
        summary_df = (
            successful.groupby("hmm_type")[summary_metrics]
            .agg(["count", "mean", "median", "max", "min"])
            .sort_index()
        )
        summary_path = out_dir / "metric_summary_by_hmm_type.csv"
        summary_df.to_csv(summary_path)
        table_paths["metric_summary_by_hmm_type"] = str(summary_path)

    correlation_metrics = _available_metrics(
        successful,
        [
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_probe_recall",
            "pmr_bp_coverage",
            "pmr_short_region_fraction",
            "pmr_mean_region_bp",
            "pmr_median_region_bp",
            "pmr_region_count",
            "window_count",
            "holding_time_guess",
            "weighted_state_f1",
        ],
    )
    if len(correlation_metrics) >= 2:
        corr_path = out_dir / "metric_correlations.csv"
        successful[correlation_metrics].corr(numeric_only=True).to_csv(corr_path)
        table_paths["metric_correlations"] = str(corr_path)

    if "status" in all_results.columns:
        status_counts = (
            all_results["status"]
            .value_counts(dropna=False)
            .rename_axis("status")
            .reset_index(name="count")
        )
    else:
        status_counts = pd.DataFrame({"status": ["success"], "count": [len(all_results)]})
    status_path = out_dir / "status_counts.csv"
    status_counts.to_csv(status_path, index=False)
    table_paths["status_counts"] = str(status_path)
    return table_paths


def _plot_top_candidate_metric_grid(
    successful: pd.DataFrame,
    out_dir: Path,
    top_n: int,
) -> Path | None:
    metrics = _available_metrics(
        successful,
        [
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_probe_recall",
            "pmr_bp_coverage",
            "pmr_short_region_fraction",
            "pmr_mean_region_bp",
        ],
    )
    if not metrics:
        return None

    top_df = successful.head(top_n).copy()
    n_cols = 2
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, max(5, n_rows * max(3.5, top_n * 0.22))),
        squeeze=False,
    )
    color_values = top_df["hmm_type"].map(HMM_TYPE_PALETTE).fillna("#6c757d")

    for ax, metric in zip(axes.flat, metrics):
        ax.barh(top_df["candidate_id"], top_df[metric], color=color_values)
        ax.set_title(_display_name(metric))
        ax.set_xlabel(_display_name(metric))
        ax.set_ylabel("Candidate")
        if metric not in {
            "pmr_mean_region_bp",
            "pmr_median_region_bp",
            "pmr_region_count",
        }:
            ax.set_xlim(left=0)
            if metric not in {"pmr_short_region_fraction"}:
                ax.set_xlim(0, 1)
        ax.invert_yaxis()

    for ax in axes.flat[len(metrics):]:
        ax.remove()

    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=label)
        for label, color in (("cthmm", HMM_TYPE_PALETTE["ct"]), ("sticky", HMM_TYPE_PALETTE["sticky"]))
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Top {min(top_n, len(top_df))} Candidates Across Key Metrics", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "top_candidates_metric_grid.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_top_candidate_heatmap(
    successful: pd.DataFrame,
    out_dir: Path,
    top_n: int,
) -> Path | None:
    metrics = _available_metrics(
        successful,
        [
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_probe_recall",
            "pmr_bp_coverage",
            "pmr_short_region_fraction",
            "pmr_mean_region_bp",
            "pmr_median_region_bp",
            "weighted_state_f1",
        ],
    )
    if len(metrics) < 2:
        return None

    top_df = successful.head(top_n).copy()
    heatmap_df = _normalized_heatmap_frame(top_df, metrics)
    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.1), max(5, len(top_df) * 0.35)))
    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Column-wise normalized score"},
        ax=ax,
    )
    ax.set_title(f"Top {min(top_n, len(top_df))} Candidates: Normalized Metric Heatmap")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Candidate")
    fig.tight_layout()
    out_path = out_dir / "top_candidates_metric_heatmap.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_metric_scatter_panels(
    successful: pd.DataFrame,
    out_dir: Path,
    annotate_top_k: int,
) -> Path | None:
    pairs = [
        ("pmr_jaccard_bp", "pmr_probe_f1"),
        ("pmr_probe_recall", "pmr_bp_coverage"),
        ("pmr_short_region_fraction", "pmr_probe_f1"),
        ("pmr_mean_region_bp", "pmr_jaccard_bp"),
    ]
    available_pairs = [
        (x_metric, y_metric)
        for x_metric, y_metric in pairs
        if x_metric in successful.columns and y_metric in successful.columns
    ]
    if not available_pairs:
        return None

    n_cols = 2
    n_rows = math.ceil(len(available_pairs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5.5 * n_rows), squeeze=False)

    for ax, (x_metric, y_metric) in zip(axes.flat, available_pairs):
        sns.scatterplot(
            data=successful,
            x=x_metric,
            y=y_metric,
            hue="hmm_type_label",
            style="hmm_type_label",
            palette={"cthmm": HMM_TYPE_PALETTE["ct"], "sticky": HMM_TYPE_PALETTE["sticky"]},
            s=80,
            alpha=0.85,
            ax=ax,
        )
        if x_metric in {"pmr_mean_region_bp", "pmr_median_region_bp", "holding_time_guess"}:
            positive = pd.to_numeric(successful[x_metric], errors="coerce")
            if (positive > 0).any():
                ax.set_xscale("log")
        ax.set_xlabel(_display_name(x_metric))
        ax.set_ylabel(_display_name(y_metric))
        ax.set_title(f"{_display_name(y_metric)} vs {_display_name(x_metric)}")
        _annotate_top_candidates(ax, successful, x_metric, y_metric, annotate_top_k)

    for ax in axes.flat[len(available_pairs):]:
        ax.remove()

    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[: len(available_pairs)]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    out_path = out_dir / "metric_scatter_panels.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_hmm_type_distributions(
    successful: pd.DataFrame,
    out_dir: Path,
) -> Path | None:
    metrics = _available_metrics(
        successful,
        [
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_bp_coverage",
            "pmr_short_region_fraction",
        ],
    )
    if not metrics or "hmm_type_label" not in successful.columns:
        return None

    n_cols = 2
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), squeeze=False)
    palette = {"cthmm": HMM_TYPE_PALETTE["ct"], "sticky": HMM_TYPE_PALETTE["sticky"]}

    for ax, metric in zip(axes.flat, metrics):
        sns.boxplot(
            data=successful,
            x="hmm_type_label",
            y=metric,
            hue="hmm_type_label",
            palette=palette,
            ax=ax,
            showfliers=False,
            legend=False,
        )
        sns.stripplot(
            data=successful,
            x="hmm_type_label",
            y=metric,
            hue="hmm_type_label",
            palette=palette,
            ax=ax,
            alpha=0.6,
            size=5,
            jitter=0.18,
            linewidth=0,
            legend=False,
        )
        ax.set_xlabel("HMM Type")
        ax.set_ylabel(_display_name(metric))
        ax.set_title(f"{_display_name(metric)} by HMM Type")
        if metric not in {"pmr_short_region_fraction"}:
            ax.set_ylim(bottom=0)
            if metric not in {"pmr_mean_region_bp", "pmr_median_region_bp"}:
                ax.set_ylim(0, 1)

    for ax in axes.flat[len(metrics):]:
        ax.remove()

    fig.tight_layout()
    out_path = out_dir / "metric_distributions_by_hmm_type.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_window_count_effects(
    successful: pd.DataFrame,
    out_dir: Path,
) -> Path | None:
    if "window_count" not in successful.columns:
        return None

    metrics = _available_metrics(
        successful,
        [
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_short_region_fraction",
        ],
    )
    if not metrics:
        return None

    n_cols = 1
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.8 * n_rows), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        sns.boxplot(
            data=successful,
            x="window_count",
            y=metric,
            color="#a8dadc",
            ax=ax,
            showfliers=False,
        )
        sns.stripplot(
            data=successful,
            x="window_count",
            y=metric,
            hue="hmm_type_label",
            palette={"cthmm": HMM_TYPE_PALETTE["ct"], "sticky": HMM_TYPE_PALETTE["sticky"]},
            ax=ax,
            alpha=0.75,
            size=5,
            jitter=0.15,
        )
        ax.set_xlabel("Window Count")
        ax.set_ylabel(_display_name(metric))
        ax.set_title(f"{_display_name(metric)} by Window Count")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label)
        for label, color in (("cthmm", HMM_TYPE_PALETTE["ct"]), ("sticky", HMM_TYPE_PALETTE["sticky"]))
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    out_path = out_dir / "window_count_effects.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_ct_holding_time_effects(
    successful: pd.DataFrame,
    out_dir: Path,
    annotate_top_k: int,
) -> Path | None:
    if "hmm_type" not in successful.columns or "holding_time_guess" not in successful.columns:
        return None
    ct_df = successful[successful["hmm_type"] == "ct"].copy()
    ct_df["holding_time_guess"] = pd.to_numeric(ct_df["holding_time_guess"], errors="coerce")
    ct_df = ct_df[ct_df["holding_time_guess"].notna() & (ct_df["holding_time_guess"] > 0)]
    if ct_df.empty:
        return None

    metrics = _available_metrics(
        ct_df,
        [
            "pmr_jaccard_bp",
            "pmr_probe_f1",
            "pmr_short_region_fraction",
        ],
    )
    if not metrics:
        return None

    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 4.5 * len(metrics)), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        sns.scatterplot(
            data=ct_df,
            x="holding_time_guess",
            y=metric,
            hue="window_count" if "window_count" in ct_df.columns else None,
            palette="viridis",
            s=90,
            alpha=0.85,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Holding Time Guess")
        ax.set_ylabel(_display_name(metric))
        ax.set_title(f"{_display_name(metric)} Across CTHMM Holding Times")
        _annotate_top_candidates(ax, ct_df, "holding_time_guess", metric, annotate_top_k)

    fig.tight_layout()
    out_path = out_dir / "ct_holding_time_effects.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_search_result_plots(
    search_results_path: str | Path,
    output_dir: str | Path | None = None,
    top_n: int = 15,
    annotate_top_k: int = 5,
) -> dict[str, str]:
    search_results_path = Path(search_results_path).resolve()
    if not search_results_path.exists():
        raise FileNotFoundError(f"Could not find search_results.csv: {search_results_path}")

    if output_dir is None:
        out_dir = search_results_path.parent / DEFAULT_OUTPUT_DIR_NAME
    else:
        out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    all_results, successful = _load_results(search_results_path)

    plot_paths: dict[str, str] = {}
    plot_builders = {
        "top_candidates_metric_grid": lambda: _plot_top_candidate_metric_grid(
            successful=successful,
            out_dir=out_dir,
            top_n=top_n,
        ),
        "top_candidates_metric_heatmap": lambda: _plot_top_candidate_heatmap(
            successful=successful,
            out_dir=out_dir,
            top_n=top_n,
        ),
        "metric_scatter_panels": lambda: _plot_metric_scatter_panels(
            successful=successful,
            out_dir=out_dir,
            annotate_top_k=annotate_top_k,
        ),
        "metric_distributions_by_hmm_type": lambda: _plot_hmm_type_distributions(
            successful=successful,
            out_dir=out_dir,
        ),
        "window_count_effects": lambda: _plot_window_count_effects(
            successful=successful,
            out_dir=out_dir,
        ),
        "ct_holding_time_effects": lambda: _plot_ct_holding_time_effects(
            successful=successful,
            out_dir=out_dir,
            annotate_top_k=annotate_top_k,
        ),
    }

    for name, builder in plot_builders.items():
        out_path = builder()
        if out_path is not None:
            plot_paths[name] = str(out_path)

    table_paths = _write_summary_tables(
        all_results=all_results,
        successful=successful,
        out_dir=out_dir,
        top_n=top_n,
    )

    manifest = {
        "search_results_path": str(search_results_path),
        "output_dir": str(out_dir),
        "n_total_candidates": int(len(all_results)),
        "n_successful_candidates": int(len(successful)),
        "top_n": int(min(top_n, len(successful))),
        "annotate_top_k": int(annotate_top_k),
        "plots": plot_paths,
        "tables": table_paths,
    }
    manifest_path = out_dir / "plot_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    return {
        "output_dir": str(out_dir),
        "plot_manifest": str(manifest_path),
        **plot_paths,
        **table_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from a methyl-seg hyperparameter search_results.csv file.",
    )
    parser.add_argument(
        "--search-results",
        default=str(DEFAULT_SEARCH_RESULTS),
        help="Path to the search_results.csv file to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots and summary tables. Defaults to <search_results_dir>/comparison_plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top-ranked candidates to emphasize in candidate-level plots.",
    )
    parser.add_argument(
        "--annotate-top-k",
        type=int,
        default=5,
        help="Number of top-ranked candidates to annotate in scatter plots.",
    )
    args = parser.parse_args()

    outputs = generate_search_result_plots(
        search_results_path=args.search_results,
        output_dir=args.output_dir,
        top_n=args.top_n,
        annotate_top_k=args.annotate_top_k,
    )
    print(yaml.safe_dump(outputs, sort_keys=False))


if __name__ == "__main__":
    main()
