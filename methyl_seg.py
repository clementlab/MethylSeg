import os
import textwrap
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cthmm
import joblib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import umap
import yaml
from hmmlearn import hmm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numba import njit
from panel import GridSpec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore", message="divide by zero encountered in log", module="cthmm"
)


CODE_DIR = Path(__file__).resolve().parent
FILES = CODE_DIR / "reference_files"
CANONICAL_AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))

# TODO: Think about what should be saved and what should be returned
# TODO: Remove legacy code that is not used anymore
# TODO: split into multiple files to clean up codebase
# TODO: Rename PMR to PMD throughout codebase for clarity and consistency with literature
# TODO: Add usage text and docstrings to all public functions and classes
# TODO: clean up plotting functions to make them easier for public use
# TODO: Add a function that runs full pipeline from raw input to figures and saves the outputs including running the cleaning step
# TODO: Add a function to run on all chromosomes and generate all summary files like methyltool compatator

@dataclass
class KMeansMethylationModel:
    kmeans: KMeans
    scaler: StandardScaler
    imputer: Optional[SimpleImputer]
    pca: Optional[PCA]
    feature_cols: List[str]
    n_states: int
    cluster_space: str = "pca"
    n_pca: Optional[int] = 5


class MethylStateAssignmentMethod(Enum):
    DEFINITION = "definition"
    KMEANS = "kmeans"
    AUTO = "auto"


class HMMObservationMode(Enum):
    DISCRETE_STATES = "discrete_states"
    GAUSSIAN_EMISSIONS = "gaussian_emissions"
    PCA_EMISSIONS = "pca_emissions"


@dataclass
class SampleInfo:
    """
    Simple dataclass to hold sample metadata and methylation data together.

    `sample_id` is a unique identifier for the sample (e.g., TCGA barcode).
    `meth_data` is expected to be a DataFrame with columns ['CpG_chrm', 'CpG_beg', 'CpG_end', 'beta'].
        - `CpG_chrm`: chromosome name (e.g., 'chr1')
        - `CpG_beg`: 0-based start position of the CpG
        - `CpG_end`: 0-based end position of the CpG (typically CpG is 1 base long)
        - `beta`: methylation beta value (0-1)
    """

    sample_id: str
    meth_data: pd.DataFrame

    @classmethod
    def __from_tsv__(cls, sample_name, file_name, sep="\t"):
        meth_data = pd.read_csv(file_name, sep=sep)

        return cls(sample_id=sample_name, meth_data=meth_data)

    def __to_tsv__(self, out_dir, sep="\t"):
        out_file = os.path.join(out_dir, f"{self.sample_id}.tsv")
        self.meth_data.to_csv(out_file, sep=sep, index=False)

    def __post_init__(self):
        required_cols = {"CpG_chrm", "CpG_beg", "CpG_end", "beta"}
        if not required_cols.issubset(self.meth_data.columns):
            missing = required_cols - set(self.meth_data.columns)
            raise ValueError(
                f"meth_data is missing required columns: {missing}. "
                "Expected columns: 'CpG_chrm', 'CpG_beg', 'CpG_end', 'beta'."
            )

        if not np.issubdtype(self.meth_data["beta"].dtype, np.number):
            raise ValueError("Column 'beta' must be numeric.")

        if (self.meth_data["beta"] < 0).any() or (self.meth_data["beta"] > 1).any():
            raise ValueError("Column 'beta' must have values between 0 and 1.")


class MethylDataPrep:
    REQUIRED_COLUMNS = ["CpG_chrm", "CpG_beg", "CpG_end", "beta"]
    INPUT_ROW_INDEX_COL = "__input_row_index__"
    LOW_COVERAGE_LIKE_BETA_VALUES = frozenset(
        {0.0, 0.25, 0.33, 0.5, 0.66, 0.67, 0.75, 1.0}
    )
    COMMON_ALIASES = {
        "CpG_chrm": ["CpG_chrm", "chrom", "chr", "chromosome"],
        "CpG_beg": ["CpG_beg", "start", "pos", "position"],
        "CpG_end": ["CpG_end", "end", "stop"],
        "beta": ["beta", "meth_beta", "methylation", "meth_percent"],
    }
    HEADER_ALIASES = {
        "CpG_chrm": {"cpg_chrm", "chrom", "chr", "chromosome"},
        "CpG_beg": {"cpg_beg", "start", "pos", "position"},
        "CpG_end": {"cpg_end", "end", "stop"},
        "beta": {"beta", "meth_beta", "methylation", "meth_percent"},
        "meth": {"meth", "methylated", "methylated_reads"},
        "coverage": {"coverage", "cov", "depth", "total_reads"},
        "probe": {"probe", "probe_id", "cpg", "cpg_id"},
    }

    def __init__(
        self,
        meth_file,
        sample_id,
        resolution="auto",
        min_coverage=5,
        remove_low_coverage_like_cpgs=False,
    ):
        """
        Parameters
        ----------
        meth_file : str or Path
            Path to methylation data file.
        sample_id : str
            Unique identifier for the sample.
        resolution : str, default="auto"
            Format of the methylation data. Valid options are:
            - "auto": Automatically detect format (450k or WGBS)
            - "wgbs": Whole-genome bisulfite sequencing format
            - "450k": Illumina 450k array format
        min_coverage : int, default=5
            Minimum coverage threshold for WGBS data.
        remove_low_coverage_like_cpgs : bool, default=False
            If True, remove CpGs with beta values commonly produced by very
            low coverage counts, such as 0.0, 0.25, 0.33, 0.5, 0.66/0.67,
            0.75, and 1.0.
        """
        self.meth_file = Path(meth_file)
        self.sample_id = sample_id
        self.resolution = resolution
        self.min_coverage = min_coverage
        self.remove_low_coverage_like_cpgs = remove_low_coverage_like_cpgs

    def _looks_like_header_row(self, row_values) -> bool:
        normalized = [str(v).strip().lower() for v in row_values]
        if len(normalized) < 4:
            return False

        beta_header = ["CpG_chrm", "CpG_beg", "CpG_end", "beta"]
        if all(
            normalized[i] in self.HEADER_ALIASES[canonical]
            for i, canonical in enumerate(beta_header)
        ):
            return True

        if len(normalized) >= 5:
            wgbs_header = ["CpG_chrm", "CpG_beg", "CpG_end", "meth", "coverage"]
            if all(
                normalized[i] in self.HEADER_ALIASES[canonical]
                for i, canonical in enumerate(wgbs_header)
            ):
                return True

            probe_header = ["CpG_chrm", "CpG_beg", "CpG_end", "beta", "probe"]
            if all(
                normalized[i] in self.HEADER_ALIASES[canonical]
                for i, canonical in enumerate(probe_header)
            ):
                return True

        return False

    def _promote_header_row(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not all(isinstance(col, int) for col in df.columns):
            return df
        if not self._looks_like_header_row(df.iloc[0].tolist()):
            return df

        df = df.copy()
        df.columns = df.iloc[0].tolist()
        return df.iloc[1:].reset_index(drop=True)

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._promote_header_row(df)

        if all(isinstance(col, int) for col in df.columns):
            if len(df.columns) == 4:
                df = df.copy()
                df.columns = ["CpG_chrm", "CpG_beg", "CpG_end", "beta"]
                return df

            if len(df.columns) == 5:
                df = df.copy()
                if self.resolution == "wgbs":
                    df.columns = ["CpG_chrm", "CpG_beg", "CpG_end", "meth", "coverage"]
                elif self.resolution == "450k":
                    df.columns = ["CpG_chrm", "CpG_beg", "CpG_end", "beta", "probe"]
                elif self.resolution == "auto":
                    col5_numeric = pd.to_numeric(df.iloc[:, 4], errors="coerce")
                    if col5_numeric.notna().mean() > 0.9:
                        df.columns = [
                            "CpG_chrm",
                            "CpG_beg",
                            "CpG_end",
                            "meth",
                            "coverage",
                        ]
                    else:
                        df.columns = [
                            "CpG_chrm",
                            "CpG_beg",
                            "CpG_end",
                            "beta",
                            "probe",
                        ]
                return df

        rename_map = {}
        for canonical, aliases in self.COMMON_ALIASES.items():
            for alias in aliases:
                if alias in df.columns:
                    rename_map[alias] = canonical
                    break
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _attach_input_row_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.INPUT_ROW_INDEX_COL in df.columns:
            return df
        df = df.copy()
        df[self.INPUT_ROW_INDEX_COL] = np.arange(len(df), dtype=np.int64)
        return df

    def _format_removed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        removed_df = df.copy()
        columns = [col for col in removed_df.columns if col != self.INPUT_ROW_INDEX_COL]
        if removed_df.empty:
            empty_df = pd.DataFrame(columns=columns)
            empty_df.index = pd.Index([], name="input_row_index", dtype=np.int64)
            return empty_df

        if self.INPUT_ROW_INDEX_COL not in removed_df.columns:
            raise ValueError(
                f"Missing {self.INPUT_ROW_INDEX_COL} while formatting removed rows."
            )

        removed_df = removed_df.set_index(self.INPUT_ROW_INDEX_COL, drop=True)
        removed_df.index = removed_df.index.astype(np.int64)
        removed_df.index.name = "input_row_index"
        return removed_df.sort_index()

    def _finalize_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self._attach_input_row_index(self._normalize_column_names(df))
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Could not prepare methylation data for {self.sample_id}. "
                f"Missing canonical columns: {missing}"
            )

        filtered_columns = self.REQUIRED_COLUMNS + [self.INPUT_ROW_INDEX_COL]
        df = df.copy()
        df["CpG_beg"] = pd.to_numeric(df["CpG_beg"], errors="raise").astype(np.int64)
        df["CpG_end"] = pd.to_numeric(df["CpG_end"], errors="raise").astype(np.int64)
        df["beta"] = pd.to_numeric(df["beta"], errors="raise").astype(np.float64)

        removed_frames = []
        if self.remove_low_coverage_like_cpgs:
            low_coverage_like_beta = df["beta"].isin(self.LOW_COVERAGE_LIKE_BETA_VALUES)
            if low_coverage_like_beta.any():
                removed_frames.append(df.loc[low_coverage_like_beta].copy())
            df = df.loc[~low_coverage_like_beta].copy()

        removed_df = (
            pd.concat(removed_frames, axis=0, sort=False)
            if removed_frames
            else df.iloc[0:0].copy()
        )
        filtered_df = df.loc[:, filtered_columns].copy()
        filtered_df = filtered_df.drop(columns=[self.INPUT_ROW_INDEX_COL]).reset_index(
            drop=True
        )
        return filtered_df, removed_df

    def _load_wgbs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(
            self.meth_file,
            sep="\t",
            header=None,
            low_memory=False,
        )
        df = self._promote_header_row(df)
        if df.shape[1] < 5:
            raise ValueError(
                f"Expected at least 5 columns for WGBS input: {self.meth_file}"
            )
        if all(isinstance(col, int) for col in df.columns):
            df.columns = ["CpG_chrm", "CpG_beg", "CpG_end", "meth", "coverage"]
        else:
            df = self._normalize_column_names(df)
        df = self._attach_input_row_index(df)
        df["meth"] = pd.to_numeric(df["meth"], errors="raise")
        df["coverage"] = pd.to_numeric(df["coverage"], errors="raise")
        df["beta"] = df["meth"] / df["coverage"]
        coverage_mask = df["coverage"] >= self.min_coverage
        coverage_removed_df = df.loc[~coverage_mask].copy()
        filtered_df, low_coverage_removed_df = self._finalize_dataframe(
            df.loc[coverage_mask].copy()
        )
        removed_frames = [
            removed_df
            for removed_df in (coverage_removed_df, low_coverage_removed_df)
            if not removed_df.empty
        ]
        removed_df = (
            pd.concat(removed_frames, axis=0, sort=False)
            if removed_frames
            else df.iloc[0:0].copy()
        )
        return filtered_df, self._format_removed_dataframe(removed_df)

    def _load_450k(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(
            self.meth_file,
            sep="\t",
            header=None,
            low_memory=False,
        )
        if df.empty:
            return pd.DataFrame(
                columns=self.REQUIRED_COLUMNS
            ), self._format_removed_dataframe(
                pd.DataFrame(columns=self.REQUIRED_COLUMNS + [self.INPUT_ROW_INDEX_COL])
            )

        df = self._promote_header_row(df)

        if df.shape[1] < 4:
            raise ValueError(
                f"Expected at least 4 columns for 450k input: {self.meth_file}"
            )

        df = self._normalize_column_names(df)
        filtered_df, removed_df = self._finalize_dataframe(df)
        return filtered_df, self._format_removed_dataframe(removed_df)

    def _load_auto(self) -> tuple[pd.DataFrame, pd.DataFrame]:

        try:
            return self._load_450k()
        except Exception:
            pass

        return self._load_wgbs()

    def prepare_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.resolution == "wgbs":
            filtered_df, removed_df = self._load_wgbs()
        elif self.resolution == "450k":
            filtered_df, removed_df = self._load_450k()
        elif self.resolution == "auto":
            filtered_df, removed_df = self._load_auto()
        else:
            raise ValueError(f"Unsupported methylation resolution: {self.resolution}")
        return filtered_df, removed_df

    def prepare(self) -> tuple[SampleInfo, pd.DataFrame]:
        filtered_df, removed_df = self.prepare_dataframe()
        return SampleInfo(sample_id=self.sample_id, meth_data=filtered_df), removed_df

    def write_prepared_tsv(self, out_file, sep="\t") -> Path:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        filtered_df, _ = self.prepare_dataframe()
        filtered_df.to_csv(out_file, sep=sep, index=False)
        return out_file


class MethylationStates(Enum):
    LOW = 0
    PMR = 1
    INTERMEDIATE = 2
    HIGH = 3

    def __str__(self):
        return self.name

    def __lt__(self, other):
        if isinstance(other, MethylationStates):
            return self.value < other.value
        return NotImplemented

    @staticmethod
    def convert_to_numeric(arr):
        arr = np.asarray(arr)
        if isinstance(arr[0], Enum):
            return np.array([a.value for a in arr], dtype=int)
        return arr.astype(int)


def get_biological_state_colors(cmap_name: str = "viridis"):
    """
    Return a fixed color mapping for the biological methylation states
    keyed by their canonical enum values (LOW=0, PMR=1, INTERMEDIATE=2, HIGH=3).
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


# TODO: move interactive beta plotting helpers to a shared utils module.
def _normalize_state_label(value) -> str | None:
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


def _annotate_plot_df_with_regions(
    df_plot: pd.DataFrame,
    regions_df: pd.DataFrame,
    *,
    chrom_col: str,
    pos_col: str,
    color_pmr_only: bool,
    region_label_col: str = "state",
) -> tuple[pd.DataFrame, str, dict[str, str], dict[str, list[str]], str, str]:
    plot_df = df_plot.copy()
    outside_region_color = "#7E7E7E"
    plot_df["__region_color__"] = "non-PMR" if color_pmr_only else "Outside regions"

    if regions_df is None or regions_df.empty:
        if color_pmr_only:
            plot_df["__region_color__"] = "non-PMR"
            return (
                plot_df,
                "__region_color__",
                {"PMR": "#d62728", "non-PMR": "#1f77b4"},
                {"__region_color__": ["PMR", "non-PMR"]},
                "PMR status",
                "PMR status",
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
        _normalize_state_label
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
            if color_pmr_only:
                color_label = (
                    "PMR"
                    if _normalize_state_label(getattr(region, region_label_col))
                    == "PMR"
                    else "non-PMR"
                )
            else:
                color_label = _normalize_state_label(getattr(region, region_label_col))
                if color_label is None:
                    color_label = "Region"
            plot_df.loc[chrom_indices[region_mask], "__region_color__"] = color_label

    if color_pmr_only:
        return (
            plot_df,
            "__region_color__",
            {"PMR": "#d62728", "non-PMR": "#1f77b4"},
            {"__region_color__": ["PMR", "non-PMR"]},
            "PMR status",
            "PMR status",
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


def _plot_interactive_beta_scatter(
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
    color_pmr_only: bool = False,
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
            keep_idx = np.sort(rng.choice(retained_n, size=retained_keep, replace=False))
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
        ) = _annotate_plot_df_with_regions(
            df_plot=df_plot,
            regions_df=color_regions_df,
            chrom_col="CpG_chrm",
            pos_col=x_col,
            color_pmr_only=color_pmr_only,
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

        if color_pmr_only:
            plot_color_col = f"{label_col}_pmr_status"
            df_plot[plot_color_col] = np.where(
                df_plot[label_col] == MethylationStates.PMR.value,
                "PMR",
                "non-PMR",
            )
            color_map = {"PMR": "#d62728", "non-PMR": "#1f77b4"}
            category_orders = {plot_color_col: ["PMR", "non-PMR"]}
            color_label = "PMR status"
            legend_title = "PMR status"
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

    if color_regions_df is None and not color_pmr_only:
        state_names = {str(s.value): s.name for s in MethylationStates}
        fig.for_each_trace(lambda t: t.update(name=state_names.get(t.name, t.name)))

    if show_plot:
        fig.show(renderer="notebook")

    if out_dir is not None:
        suffix = "_pmr_only" if color_pmr_only else ""
        if color_regions_df is not None:
            suffix += "_region_coloring"
        fig.write_html(f"{out_dir}/interactive_beta_by_{label_col}{suffix}.html")

    return fig


@njit
def _build_emission_matrix_numba(
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


class MethylSegConfig:
    """
    Lightweight helper that knows how to build a serializable dictionary
    from a MethylSegPathway instance and how to write/read YAML.
    """

    def __init__(self, config_dict: dict):
        self.config = config_dict

    @classmethod
    def from_instance(
        cls, inst: "MethylSegPathway", out_dir: str | None = None
    ) -> "MethylSegConfig":
        """
        Build a serializable config dict from a MethylSegPathway instance.

        - DataFrames are saved to feather files in inst.out_dir (or out_dir if provided).
        - scikit-learn objects (kmeans, pca, scaler) are joblib-dumped (optional).
        """
        base_dir = Path(out_dir or inst.out_dir or ".")
        base_dir.mkdir(parents=True, exist_ok=True)

        cfg = {}

        # pathway-level params
        cfg["pathway"] = {
            "data_path": getattr(inst, "data_path", None),
            "meth_ref_path": getattr(inst, "meth_ref_path", None),
            "samples_info_path": getattr(inst, "samples_info_path", None),
            "out_dir": str(base_dir),
            "train_sample": getattr(inst, "train_sample", None),
            "train_chroms": getattr(inst, "train_chroms", None),
            "max_cpg_per_chrom": getattr(inst, "max_cpg_per_chrom", None),
            "random_state": getattr(inst, "random_state", None),
        }

        # Assigner params
        cfg["state_assigner"] = {
            "window_specs": getattr(inst.assigner, "window_specs", None),
            "n_states": getattr(inst.assigner, "n_states", None),
            "int_low_cutoff": getattr(inst.assigner, "int_low_cutoff", None),
            "int_high_cutoff": getattr(inst.assigner, "int_high_cutoff", None),
            "high_cutoff": getattr(inst.assigner, "high_cutoff", None),
        }

        # --- Save rule-based state cutoffs if present ---
        if hasattr(inst.analyzer, "state_cutoffs"):
            cfg["state_cutoffs"] = {
                "cutoffs": inst.analyzer.state_cutoffs,
                "set_manually": bool(
                    getattr(inst.analyzer, "cutoffs_set_manually", False)
                ),
            }

        # --- Save array-like hmm_params to files and build a serializable copy ---
        hmm_params = getattr(inst, "hmm_params", {}) or {}
        # We'll write arrays to out_dir/hmm_params/
        hmm_param_dir = base_dir / "hmm_params"
        hmm_param_dir.mkdir(exist_ok=True, parents=True)

        serializable_hmm_params = {}
        for k, v in hmm_params.items():
            # numpy arrays
            if isinstance(v, np.ndarray):
                filename = f"{k}.npy"
                p = hmm_param_dir / filename
                # ensure float dtype if numeric (cthmm may require float)
                try:
                    np.save(p, v)
                except Exception:
                    # try to coerce to numpy array then save
                    np.save(p, np.asarray(v))
                serializable_hmm_params[k] = {"__npy_path__": str(p)}
            # long lists/tuples — treat similar to array
            elif isinstance(v, (list, tuple)):
                # heuristics: if short (len <= 20) keep inline; else save
                if len(v) <= 20:
                    serializable_hmm_params[k] = v
                else:
                    p = hmm_param_dir / f"{k}.npy"
                    np.save(p, np.asarray(v))
                    serializable_hmm_params[k] = {"__npy_path__": str(p)}
            else:
                # scalar, string, dict, etc. — assume yaml serializable
                serializable_hmm_params[k] = v

        cfg["hmm"] = {"type": getattr(inst, "hmm_type", None)}
        hmm_observation_mode = getattr(inst, "hmm_observation_mode", None)
        if isinstance(hmm_observation_mode, HMMObservationMode):
            hmm_observation_mode = hmm_observation_mode.value
        cfg["hmm"]["observation_mode"] = hmm_observation_mode
        cfg["hmm"]["params"] = serializable_hmm_params

        # Analyzer/segmenter basic settings
        state_assignment_method = getattr(
            inst.segmentor,
            "state_assignment_method",
            getattr(inst, "state_assignment_method", None),
        )
        if isinstance(state_assignment_method, MethylStateAssignmentMethod):
            state_assignment_method = state_assignment_method.value
        segmenter_hmm_observation_mode = getattr(
            inst.segmentor,
            "hmm_observation_mode",
            getattr(inst, "hmm_observation_mode", None),
        )
        if isinstance(segmenter_hmm_observation_mode, HMMObservationMode):
            segmenter_hmm_observation_mode = segmenter_hmm_observation_mode.value
        cfg["segmenter"] = {
            "state_assignment_method": state_assignment_method,
            "hmm_observation_mode": segmenter_hmm_observation_mode,
            "out_dir": str(base_dir),
        }

        # Paths for dataframes and models to be saved
        saved = {}

        # Save key DataFrames (if present) as feather and record their relative paths
        df_items = {
            "train_meth": getattr(inst.assigner, "train_meth", None),
            "training_summary_df": getattr(inst.assigner, "training_summary_df", None),
            "train_emission_df": (
                getattr(inst.assigner, "train_emmission_df", None)
                if hasattr(inst.assigner, "train_emmission_df")
                else getattr(inst.assigner, "train_emission_df", None)
            ),
            "train_joint": getattr(inst.analyzer, "train_joint", None),
            "regions_df": getattr(inst.segmentor, "regions_df", None),
        }

        for name, df in df_items.items():
            if df is None:
                continue
            path = base_dir / f"{name}.feather"
            # pandas may raise on non-DataFrame - skip safely
            try:
                df.reset_index(drop=True).to_feather(path)
                saved[name] = str(path)
            except Exception:
                # not a dataframe or failed to write; skip
                continue

        cfg["saved_tables"] = saved

        # Save model artifacts (kmeans, pca, scaler) using joblib if present
        models_saved = {}
        model_dir = base_dir / "models"
        model_dir.mkdir(exist_ok=True)
        if hasattr(inst.assigner, "model") and inst.assigner.model is not None:
            model = inst.assigner.model
            # Save the sklearn components separately
            try:
                if getattr(model, "kmeans", None) is not None:
                    joblib.dump(model.kmeans, model_dir / "kmeans.joblib")
                    models_saved["kmeans"] = str(model_dir / "kmeans.joblib")
                if getattr(model, "pca", None) is not None:
                    joblib.dump(model.pca, model_dir / "pca.joblib")
                    models_saved["pca"] = str(model_dir / "pca.joblib")
                if getattr(model, "scaler", None) is not None:
                    joblib.dump(model.scaler, model_dir / "scaler.joblib")
                    models_saved["scaler"] = str(model_dir / "scaler.joblib")
                if getattr(model, "imputer", None) is not None:
                    joblib.dump(model.imputer, model_dir / "imputer.joblib")
                    models_saved["imputer"] = str(model_dir / "imputer.joblib")
                # save feature_cols + n_states
                models_saved["feature_cols"] = list(model.feature_cols)
                models_saved["n_states"] = int(model.n_states)
            except Exception:
                # if model objects are not serializable, skip with a warning in the YAML
                models_saved["warning"] = "failed to joblib.dump some model parts"
        cfg["models"] = models_saved

        return cls(cfg)

    def to_yaml(self, yaml_path: str):
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(self.config, fh, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MethylSegConfig":
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls(data)


class MethylStateAssigner:

    def __init__(
        self,
        window_specs: List[Tuple[int, str]] = [
            (500_000, "500kb"),
            (1_000_000, "1Mb"),
        ],
        n_states: int = 4,
        int_low_cutoff: float = 0.2,
        int_high_cutoff: float = 0.7,
        high_cutoff: float = 0.8,
        out_dir=".",
        random_state: Optional[int] = 42,
        cluster_space: str = "pca",
        n_pca: Optional[int] = 5,
    ):
        """
        window_specs : list of (window_size, label).
        n_states : int
            Number of states for the HMM.
        """
        self.window_specs = window_specs
        self.n_states = n_states
        self.int_low_cutoff = int_low_cutoff
        self.int_high_cutoff = int_high_cutoff
        self.high_cutoff = high_cutoff
        self.out_dir = out_dir
        self.random_state = random_state
        self.cluster_space = self._validate_cluster_space(cluster_space)
        self.n_pca = n_pca

    @staticmethod
    def _validate_cluster_space(cluster_space: str) -> str:
        normalized_cluster_space = str(cluster_space).lower()
        if normalized_cluster_space not in {"pca", "raw"}:
            raise ValueError(
                "cluster_space must be either 'pca' or 'raw'. "
                f"Received: {cluster_space!r}"
            )
        return normalized_cluster_space

    def _get_regional_window_labels(self) -> List[str]:
        sorted_window_specs = sorted(self.window_specs, key=lambda item: item[0])
        if len(sorted_window_specs) <= 2:
            return [label for _, label in sorted_window_specs]
        return [label for _, label in sorted_window_specs[-2:]]

    def generate_multi_window_summary_centered(
        self,
        meth_data: pd.DataFrame,
        chrom: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute per-CpG multi-window summary statistics using windows centered on each CpG.

        Parameters
        ----------
        meth_data : DataFrame
        Must contain columns ['CpG_chrm', 'CpG_beg', 'CpG_end', 'beta'].
        chrom : optional chromosome filter.

        Returns
        -------
        DataFrame : one row per CpG with a 'summaries' dict per row.
        """
        df = meth_data.copy()

        if chrom is not None:
            df = df[df["CpG_chrm"] == chrom]

        df["CpG_mid"] = ((df["CpG_beg"].values + df["CpG_end"].values) // 2).astype(
            np.int64
        )
        df["summaries"] = [{} for _ in range(len(df))]

        def prefix_interval_sum(cumsum_array, start_idx, end_idx):
            if start_idx > end_idx:
                return 0.0
            if start_idx == 0:
                return float(cumsum_array[end_idx])
            return float(cumsum_array[end_idx] - cumsum_array[start_idx - 1])

        for chrom_name, chrom_df in df.groupby("CpG_chrm", sort=False):
            global_row_indices = chrom_df.index.to_numpy()
            positions = chrom_df["CpG_mid"].to_numpy()
            betas = chrom_df["beta"].to_numpy().astype(float)
            n_cpgs = len(chrom_df)
            if n_cpgs == 0:
                continue

            beta_cumsum = np.cumsum(betas)
            beta_squared_cumsum = np.cumsum(betas**2)

            int_cpgs = (
                (betas >= self.int_low_cutoff) & (betas <= self.int_high_cutoff)
            ).astype(np.int64)
            high_cpgs = (betas > self.high_cutoff).astype(np.int64)
            low_cpgs = (betas < self.int_low_cutoff).astype(np.int64)

            intermediate_cumsum = np.cumsum(int_cpgs)
            high_cumsum = np.cumsum(high_cpgs)
            low_cumsum = np.cumsum(low_cpgs)

            for window_size, label in self.window_specs:
                window_start_idx = 0
                window_end_idx = 0

                for cpg_idx in range(n_cpgs):
                    cpg_center = positions[cpg_idx]
                    window_start_pos = cpg_center - window_size // 2
                    window_end_pos = cpg_center + window_size // 2

                    while (
                        window_start_idx < n_cpgs
                        and positions[window_start_idx] < window_start_pos
                    ):
                        window_start_idx += 1

                    while (
                        window_end_idx + 1 < n_cpgs
                        and positions[window_end_idx + 1] <= window_end_pos
                    ):
                        window_end_idx += 1

                    cpg_count = window_end_idx - window_start_idx + 1
                    if cpg_count <= 0:
                        continue

                    sum_beta = prefix_interval_sum(
                        beta_cumsum, window_start_idx, window_end_idx
                    )
                    sum_beta_squared = prefix_interval_sum(
                        beta_squared_cumsum, window_start_idx, window_end_idx
                    )

                    mean_beta = sum_beta / cpg_count
                    variance = (sum_beta_squared / cpg_count) - (mean_beta**2)
                    variance = max(variance, 0.0)
                    stddev_beta = float(np.sqrt(variance))

                    intermediate_count = prefix_interval_sum(
                        intermediate_cumsum, window_start_idx, window_end_idx
                    )
                    high_count = prefix_interval_sum(
                        high_cumsum, window_start_idx, window_end_idx
                    )
                    low_count = prefix_interval_sum(
                        low_cumsum, window_start_idx, window_end_idx
                    )

                    intermediate_fraction = intermediate_count / cpg_count
                    high_fraction = high_count / cpg_count
                    low_fraction = low_count / cpg_count

                    median_beta = float(
                        np.median(betas[window_start_idx : window_end_idx + 1])
                    )

                    summary = {
                        "window_info": {
                            "window_size": window_size,
                            "window_start": int(window_start_pos),
                            "window_end": int(window_end_pos),
                            "CpG_count": int(cpg_count),
                        },
                        "avg_meth": float(mean_beta),
                        "median_meth": median_beta,
                        "std": float(stddev_beta),
                        "high_pct": float(high_fraction),
                        "int_pct": float(intermediate_fraction),
                        "low_pct": float(low_fraction),
                    }

                    global_row = global_row_indices[cpg_idx]
                    df.at[global_row, "summaries"][label] = summary

        df = df.drop(columns=["CpG_mid"])
        return df

    def create_emission_df(
        self,
        summary_stats: pd.DataFrame,
        windows_to_use: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert the `summaries` column into a flat emission feature matrix.

        Parameters
        ----------
        windows_to_use : list of window labels to include (subset of summary keys).
            If None, use all available windows.

        Returns
        -------
        DataFrame with columns:
        'beta' +
        '{label}_avg_meth', '{label}_std', '{label}_high_pct', '{label}_int_pct',
        '{label}_low_pct',
        '{label}_n_cpg'
        """
        if "summaries" not in summary_stats.columns:
            raise ValueError("summary_stats must have a 'summaries' column.")

        first_non_empty = None
        for idx, val in summary_stats["summaries"].items():
            if isinstance(val, dict) and len(val) > 0:
                first_non_empty = val
                break

        if first_non_empty is None:
            # helpful debugging info
            n_rows = len(summary_stats)
            raise ValueError(
                "create_emission_df: 'summaries' column contains no non-empty entries. "
                f"summary_stats has {n_rows} rows. "
                "This likely means load_sample_methylation returned zero CpGs for this sample/chrom. "
                "Check that the sample exists and the methylation file contains entries for the requested chromosome."
            )
        all_labels = list(first_non_empty.keys())
        if windows_to_use is None:
            windows = all_labels
        else:
            missing = [w for w in windows_to_use if w not in all_labels]
            if missing:
                raise ValueError(f"Requested windows not found in summaries: {missing}")
            windows = windows_to_use

        emission_data: Dict[int, Dict[str, float]] = {}

        for idx, row in summary_stats.iterrows():
            emission_data[idx] = {}
            emission_data[idx]["beta"] = float(row["beta"])
            summaries = row["summaries"]
            for label in windows:
                window_summary = summaries[label]
                emission_data[idx][f"{label}_avg_meth"] = window_summary["avg_meth"]
                emission_data[idx][f"{label}_std"] = window_summary["std"]
                emission_data[idx][f"{label}_high_pct"] = window_summary["high_pct"]
                emission_data[idx][f"{label}_int_pct"] = window_summary["int_pct"]
                emission_data[idx][f"{label}_low_pct"] = window_summary["low_pct"]
                emission_data[idx][f"{label}_n_cpg"] = float(
                    window_summary["window_info"]["CpG_count"]
                )

        emission_df = pd.DataFrame.from_dict(emission_data, orient="index")
        emission_df.index = summary_stats.index
        return emission_df

    def _append_derived_emission_features(
        self,
        emission_df: pd.DataFrame,
    ) -> pd.DataFrame:
        emission_df = emission_df.copy()
        sorted_window_specs = sorted(self.window_specs, key=lambda item: item[0])
        active_window_labels = [
            label
            for _, label in sorted_window_specs
            if f"{label}_avg_meth" in emission_df.columns
        ]

        if not active_window_labels:
            emission_df["beta_vs_largest_window_avg_meth_abs_diff"] = 0.0
            emission_df["smallest_vs_largest_window_avg_meth_abs_diff"] = 0.0
            return emission_df

        beta_values = pd.to_numeric(emission_df["beta"], errors="raise")
        largest_window_label = active_window_labels[-1]
        largest_window_avg = pd.to_numeric(
            emission_df[f"{largest_window_label}_avg_meth"],
            errors="raise",
        )
        emission_df["beta_vs_largest_window_avg_meth_abs_diff"] = (
            beta_values - largest_window_avg
        ).abs()

        if len(active_window_labels) == 1:
            emission_df["smallest_vs_largest_window_avg_meth_abs_diff"] = 0.0
        else:
            smallest_window_label = active_window_labels[0]
            smallest_window_avg = pd.to_numeric(
                emission_df[f"{smallest_window_label}_avg_meth"],
                errors="raise",
            )
            emission_df["smallest_vs_largest_window_avg_meth_abs_diff"] = (
                smallest_window_avg - largest_window_avg
            ).abs()

        return emission_df

    def build_emission_matrix(
        self,
        positions,
        betas,
        window_specs,
        int_low_cutoff,
        int_high_cutoff,
        high_cutoff,
    ):

        window_sizes = np.array([w[0] for w in window_specs], dtype=np.int64)

        X = _build_emission_matrix_numba(
            positions,
            betas,
            window_sizes,
            int_low_cutoff,
            int_high_cutoff,
            high_cutoff,
        )

        # Feature names (Python side)
        feature_names = ["beta"]
        for _, label in window_specs:
            feature_names.extend(
                [
                    f"{label}_avg_meth",
                    f"{label}_std",
                    f"{label}_high_pct",
                    f"{label}_int_pct",
                    f"{label}_low_pct",
                    f"{label}_n_cpg",
                ]
            )

        return X, feature_names

    def absorb_small_clusters(
        self,
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

    # TODO : move to utils to share with MethylStateAnalyzer and MethylSegmenter
    def relabel_by_mean_emission(
        self,
        raw_labels: np.ndarray,
        emission_df: pd.DataFrame,
        state_cutoffs: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        labels = np.asarray(self.absorb_small_clusters(raw_labels, emission_df))
        clusters = np.unique(labels)

        if len(clusters) == 0:
            return np.asarray(labels, dtype=object)

        beta_min = (
            self.int_low_cutoff
            if state_cutoffs is None
            else state_cutoffs.get("beta_low_max", self.int_low_cutoff)
        )
        beta_max = (
            self.int_high_cutoff
            if state_cutoffs is None
            else state_cutoffs.get("beta_high_min", self.int_high_cutoff)
        )

        regional_window_labels = self._get_regional_window_labels()

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

        def pmr_score(cluster) -> float:
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
            if state == MethylationStates.PMR:
                return pmr_score(cluster) 
            if state == MethylationStates.INTERMEDIATE:
                return intermediate_score(cluster)
            if state == MethylationStates.HIGH:
                return high_score(cluster)
            raise ValueError(f"Unknown state: {state}")

        candidate_states = [
            MethylationStates.LOW,
            MethylationStates.PMR,
            MethylationStates.INTERMEDIATE,
            MethylationStates.HIGH,
        ]

        # -----------------------------
        # Assign the most meaningful label(s) uniquely
        # -----------------------------
        best_assignment = None
        best_key = None

        for assignment in permutations(candidate_states, len(clusters)):
            score_vector = tuple(
                state_score(c, s) for c, s in zip(clusters, assignment)
            )
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

    def _transform_emission_features(
        self,
        feature_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_matrix = feature_matrix.copy()
        count_cols = [col for col in feature_matrix.columns if col.endswith("_n_cpg")]
        for col in count_cols:
            col_values = pd.to_numeric(feature_matrix[col], errors="raise")
            if (col_values.dropna() < 0).any():
                raise ValueError(
                    f"Emission count feature '{col}' contains negative values, "
                    "so log1p preprocessing cannot be applied."
                )
            feature_matrix[col] = np.log1p(col_values.astype(np.float64))
        return feature_matrix

    def _preprocess_emission_features(
        self,
        emission_df: pd.DataFrame,
        feature_cols: List[str],
        fit: bool = False,
    ):
        feature_matrix = emission_df[feature_cols].copy()
        feature_matrix = self._transform_emission_features(feature_matrix)

        all_nan_cols = feature_matrix.columns[feature_matrix.isna().all()].tolist()
        if fit and all_nan_cols:
            raise ValueError(
                "Emission features are entirely missing for columns: "
                f"{all_nan_cols}. Median imputation cannot fit these features."
            )

        feature_values = feature_matrix.to_numpy(dtype=np.float64, copy=True)
        if np.isinf(feature_values).any():
            raise ValueError(
                "Emission features contain infinite values before imputation."
            )
        has_missing = np.isnan(feature_values).any()

        if fit:
            if has_missing:
                imputer = SimpleImputer(strategy="median")
                imputed_values = imputer.fit_transform(feature_values)
            else:
                imputer = None
                imputed_values = feature_values
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(imputed_values)
            if not np.isfinite(scaled_values).all():
                raise ValueError(
                    "Emission preprocessing produced non-finite values during fit."
                )
            return scaled_values, imputer, scaler

        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        if getattr(self.model, "scaler", None) is None:
            raise ValueError(
                "The trained KMeans model is missing a scaler. Refit the model "
                "with the updated preprocessing pipeline."
            )

        if getattr(self.model, "imputer", None) is None:
            if has_missing:
                raise ValueError(
                    "Emission features contain missing values, but the trained "
                    "KMeans model was fit without an imputer."
                )
            imputed_values = feature_values
        elif has_missing:
            imputed_values = self.model.imputer.transform(feature_values)
        else:
            imputed_values = feature_values
        if not np.isfinite(imputed_values).all():
            raise ValueError(
                "Emission preprocessing produced non-finite values after imputation."
            )
        scaled_values = self.model.scaler.transform(imputed_values)
        if not np.isfinite(scaled_values).all():
            raise ValueError(
                "Emission preprocessing produced non-finite values after scaling."
            )
        return scaled_values

    def _resolve_feature_cols(
        self,
        emission_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> List[str]:
        if feature_cols is None:
            feature_cols = [
                col
                for col in emission_df.columns.tolist()
                if not col.endswith("_n_cpg")
            ]

        if not feature_cols:
            raise ValueError(
                "No emission feature columns were selected. Pass feature_cols "
                "explicitly if you want to include only count-based features."
            )

        return feature_cols

    def fit_kmeans_on_emissions(
        self,
        emission_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[KMeansMethylationModel, Optional[np.ndarray], np.ndarray]:
        """
        Fit KMeans on emission features using the assigner's configured cluster space.
        Also returns PCA scores for PCA-backed clustering and relabeled assignments.
        """
        feature_cols = self._resolve_feature_cols(
            emission_df=emission_df,
            feature_cols=feature_cols,
        )

        X_scaled, imputer, scaler = self._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=feature_cols,
            fit=True,
        )

        if self.cluster_space == "pca":
            if self.n_pca is None or self.n_pca <= 0:
                raise ValueError(
                    "n_pca must be a positive integer when cluster_space='pca'."
                )
            n_components = min(self.n_pca, X_scaled.shape[0], X_scaled.shape[1])
            if n_components <= 0:
                raise ValueError(
                    "Cannot fit PCA for clustering because the emission matrix is empty."
                )
            pca = PCA(n_components=n_components, random_state=self.random_state)
            kmeans_input = pca.fit_transform(X_scaled)
            pca_scores = kmeans_input
        else:
            pca = None
            kmeans_input = X_scaled
            pca_scores = None

        kmeans = KMeans(
            n_clusters=self.n_states, n_init=10, random_state=self.random_state
        )
        raw_labels = kmeans.fit_predict(kmeans_input)
        relabeled = self.relabel_by_mean_emission(raw_labels, emission_df)
        model = KMeansMethylationModel(
            kmeans=kmeans,
            scaler=scaler,
            imputer=imputer,
            pca=pca,
            feature_cols=feature_cols,
            n_states=self.n_states,
            cluster_space=self.cluster_space,
            n_pca=self.n_pca,
        )
        return model, pca_scores, relabeled

    def apply_kmeans_to_emissions(
        self,
        emission_df: pd.DataFrame,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a previously trained KMeansMethylationModel to a new emission_df.

        Returns (pca_scores, raw_distances, raw_labels, relabeled_labels).
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        X_scaled = self._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )

        if self.model.pca is not None:
            pca_scores = self.model.pca.transform(X_scaled)
            kmeans_input = pca_scores
        else:
            pca_scores = None
            kmeans_input = X_scaled

        raw_distances = self.model.kmeans.transform(kmeans_input)

        raw_labels = self.model.kmeans.predict(kmeans_input)
        relabeled = self.relabel_by_mean_emission(raw_labels, emission_df)
        return pca_scores, raw_distances, raw_labels, relabeled

    def _get_kmeans_metric_input(self, emission_df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        X_scaled = self._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )
        metric_cluster_space = getattr(self.model, "cluster_space", self.cluster_space)
        if metric_cluster_space == "pca":
            if getattr(self.model, "pca", None) is None:
                raise ValueError(
                    "The trained KMeans model is configured for PCA clustering but "
                    "is missing a PCA model."
                )
            return self.model.pca.transform(X_scaled)
        return X_scaled

    def calculate_kmeans_cluster_metrics(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """
        Calculate clustering-quality metrics in the same feature space used for KMeans.
        """
        metrics = {
            "silhouette_score": None,
            "davies_bouldin_score": None,
        }

        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        labels_array = np.asarray(labels)
        if labels_array.ndim == 0:
            labels_array = labels_array.reshape(1)
        if labels_array.size == 0:
            return metrics

        labels_numeric = self._normalize_plot_labels(
            labels=labels_array,
            expected_length=len(emission_df),
        )
        metric_input = self._get_kmeans_metric_input(emission_df)

        unique_labels = np.unique(labels_numeric)
        n_samples = metric_input.shape[0]
        n_clusters = len(unique_labels)
        if n_samples < 2 or n_clusters < 2 or n_clusters >= n_samples:
            return metrics

        try:
            metrics["silhouette_score"] = float(
                silhouette_score(metric_input, labels_numeric)
            )
        except ValueError:
            pass

        try:
            metrics["davies_bouldin_score"] = float(
                davies_bouldin_score(metric_input, labels_numeric)
            )
        except ValueError:
            pass

        return metrics

    def _fit_plot_pca(
        self,
        emission_df: pd.DataFrame,
        n_pca_plot: int,
    ) -> Tuple[PCA, np.ndarray, List[str], bool]:
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        if n_pca_plot not in (2, 3):
            raise ValueError("n_pca_plot must be either 2 or 3.")

        X_scaled = self._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )
        feature_names = list(self.model.feature_cols)

        if self.model.pca is not None:
            pca = self.model.pca
            if pca.n_components_ < n_pca_plot:
                raise ValueError(
                    "The trained PCA model does not contain enough components for "
                    f"a {n_pca_plot}D PCA plot."
                )
            return pca, pca.transform(X_scaled), feature_names, True

        max_plot_components = min(X_scaled.shape[0], X_scaled.shape[1])
        if max_plot_components < n_pca_plot:
            raise ValueError(
                "Not enough samples/features are available to fit a temporary PCA "
                f"with {n_pca_plot} components."
            )

        pca = PCA(n_components=n_pca_plot, random_state=self.random_state)
        plot_scores = pca.fit_transform(X_scaled)
        return pca, plot_scores, feature_names, False

    def plot_umap_clusters(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        chrom: Optional[str] = None,
        sample_name: Optional[str] = None,
        use_pca: bool = False,
        use_parrallel: bool = True,
    ):
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        random_state = None if use_parrallel else self.random_state
        if use_parrallel:
            print(
                "UMAP parralellisation cannot work with random seed, setting random_state to None for UMAP."
            )

        X_scaled = self._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )

        if use_pca:
            if self.model.pca is None:
                raise ValueError(
                    "model.pca is None: PCA-backed UMAP requires fitting with n_pca > 0."
                )
            umap_input = self.model.pca.transform(X_scaled)
            title_suffix = "PCA features"
        else:
            umap_input = X_scaled
            title_suffix = "raw features"

        embedding = umap.UMAP(n_components=2, random_state=random_state).fit_transform(
            umap_input
        )

        cmap, norm, state_colors_rgba, _ = get_biological_state_colors()
        labels_numeric = MethylationStates.convert_to_numeric(labels)

        state_names = {state.value: state.name for state in MethylationStates}
        present_states = get_present_biological_states(labels_numeric)

        title_parts = []
        if sample_name is not None:
            title_parts.append(str(sample_name))
        if chrom is not None:
            title_parts.append(str(chrom))
        title_prefix = " ".join(title_parts) if title_parts else "Sample"

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels_numeric,
            cmap=cmap,
            norm=norm,
            s=8,
        )
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title(f"{title_prefix}: UMAP + KMeans States ({title_suffix})")
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=state_colors_rgba[state_value],
                markeredgecolor=state_colors_rgba[state_value],
                markersize=7,
                label=state_names.get(state_value, str(state_value)),
            )
            for state_value in present_states
        ]
        if legend_handles:
            plt.legend(handles=legend_handles, title="State", loc="best")
        plt.tight_layout()
        plt.show()

    def plot_kmeans_clusters(
        self,
        meth_data: pd.DataFrame,
        labels: np.ndarray,
        chrom: Optional[str] = None,
        sample_name: Optional[str] = None,
        feature_cols_for_table: Optional[List[str]] = None,
        interactive: bool = False,
    ):
        # Convert Enum labels to integers if needed
        if isinstance(labels.flat[0], Enum):
            labels_numeric = np.array([lbl.value for lbl in labels])
        else:
            labels_numeric = labels

        n_states = self.n_states
        cmap, norm, state_colors_rgba, state_colors_hex = get_biological_state_colors()
        present_states = get_present_biological_states(labels_numeric)
        x_pos = meth_data["CpG_beg"].to_numpy()
        y_beta = meth_data["beta"].to_numpy()
        title_parts = []
        if sample_name is not None:
            title_parts.append(str(sample_name))
        if chrom is not None:
            title_parts.append(str(chrom))
        title_prefix = " ".join(title_parts) if title_parts else "Sample"
        if not interactive:

            if len(meth_data) != len(labels):
                raise ValueError("meth_data and labels must be the same length.")

            show_table = (
                feature_cols_for_table is not None and len(feature_cols_for_table) > 0
            )

            if show_table:
                fig = plt.figure(figsize=(18, 6))
                gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)
                ax_scatter = fig.add_subplot(gs[0])
                ax_table = fig.add_subplot(gs[1])
            else:
                fig = plt.figure(figsize=(12, 5))
                ax_scatter = fig.add_subplot(111)
                ax_table = None

            x_pos = meth_data["CpG_beg"].to_numpy()
            y_beta = meth_data["beta"].to_numpy()

            sc = ax_scatter.scatter(
                x_pos,
                y_beta,
                c=labels_numeric,
                cmap=cmap,
                norm=norm,
                s=10,
            )
            ax_scatter.set_xlabel("Genomic Position")
            ax_scatter.set_ylabel("Methylation (beta)")
            ax_scatter.set_title(f"{title_prefix}: Methylation Beta by KMeans Cluster")
            cbar = plt.colorbar(sc, ax=ax_scatter, ticks=present_states, label="State")
            cbar.set_ticklabels(
                [MethylationStates(state_value).name for state_value in present_states]
            )

            if show_table:
                ax_table.axis("off")
                rows = [
                    feature_cols_for_table[i : i + 2]
                    for i in range(0, len(feature_cols_for_table), 2)
                ]
                for r in rows:
                    if len(r) < 2:
                        r.append("")
                table = ax_table.table(
                    cellText=rows,
                    colLabels=["Feature", "Feature"],
                    loc="center",
                    cellLoc="left",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.3)
                ax_table.set_title("Features Used", pad=10)

            plt.tight_layout()
            plt.show()

        if interactive:
            # Always plot using readable biological state names so sparse
            # numeric labels like {0, 2, 3} map correctly to colors.
            labels_str = np.array(
                [MethylationStates(int(lbl)).name for lbl in labels_numeric]
            )

            # Ordered unique label names
            state_names = [
                MethylationStates(state_value).name for state_value in present_states
            ]

            color_map = {
                state_name: state_colors_hex[MethylationStates[state_name].value]
                for state_name in state_names
            }

            fig_interactive = px.scatter(
                x=x_pos,
                y=y_beta,
                color=labels_str,  # Enum names now :)
                color_discrete_map=color_map,
                category_orders={"color": state_names},
                labels={
                    "x": "Genomic Position",
                    "y": "Methylation (beta)",
                    "color": "State",
                },
                title=f"{title_prefix}: Methylation States (Interactive)",
            )
            fig_interactive.update_traces(marker=dict(size=4))
            fig_interactive.show(renderer="notebook")

    def plot_pca_clusters(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        n_pca_plot: int = 2,  # 2 or 3
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,  # True -> old hexbin behavior
        interactive: bool = False,  # 3D Plotly option
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        sample_name: str | None = None,
        chrom: str | None = None,
    ):
        """
        PCA embedding + loadings, using consistent colors per state.

        Set ``include_kmeans_metrics=False`` to skip the expensive clustering
        quality metric calculation and annotation.
        """
        return self._plot_pca_clusters_impl(
            emission_df=emission_df,
            labels=labels,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
            pca_hexbin=pca_hexbin,
            interactive=interactive,
            include_kmeans_metrics=include_kmeans_metrics,
            include_biplot=include_biplot,
            label_title=label_title,
            sample_name=sample_name,
            chrom=chrom,
            highlight_mask=None,
        )

    def _wrap_pca_loading_feature_name(
        self,
        feature_name: str,
        width: int = 24,
    ) -> str:
        if not feature_name:
            return ""
        return textwrap.fill(
            str(feature_name),
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )

    def _build_pca_loadings_table(
        self,
        pca: PCA,
        feature_names: List[str],
        n_pca_plot: int,
        top_n_loadings: int,
    ) -> Tuple[List[List[str]], List[str]]:
        n_pca = pca.n_components_
        signed_loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_pca)],
            index=feature_names,
        )
        abs_loadings = signed_loadings.abs()

        top_pc1 = abs_loadings["PC1"].sort_values(ascending=False).head(top_n_loadings)
        top_pc2 = abs_loadings["PC2"].sort_values(ascending=False).head(top_n_loadings)
        if n_pca_plot == 3 and "PC3" in abs_loadings.columns:
            top_pc3 = (
                abs_loadings["PC3"].sort_values(ascending=False).head(top_n_loadings)
            )
        else:
            top_pc3 = None

        table_data = []
        for i in range(top_n_loadings):
            row = [
                (
                    self._wrap_pca_loading_feature_name(top_pc1.index[i])
                    if i < len(top_pc1)
                    else ""
                ),
                (
                    f"{signed_loadings.loc[top_pc1.index[i], 'PC1']:.3f}"
                    if i < len(top_pc1)
                    else ""
                ),
                (
                    self._wrap_pca_loading_feature_name(top_pc2.index[i])
                    if i < len(top_pc2)
                    else ""
                ),
                (
                    f"{signed_loadings.loc[top_pc2.index[i], 'PC2']:.3f}"
                    if i < len(top_pc2)
                    else ""
                ),
            ]
            if n_pca_plot == 3 and top_pc3 is not None:
                row.extend(
                    [
                        (
                            self._wrap_pca_loading_feature_name(top_pc3.index[i])
                            if i < len(top_pc3)
                            else ""
                        ),
                        (
                            f"{signed_loadings.loc[top_pc3.index[i], 'PC3']:.3f}"
                            if i < len(top_pc3)
                            else ""
                        ),
                    ]
                )
            table_data.append(row)

        if n_pca_plot == 3 and top_pc3 is not None:
            col_labels = [
                "PC1 Feature",
                "Abs Loading",
                "PC2 Feature",
                "Abs Loading",
                "PC3 Feature",
                "Abs Loading",
            ]
        else:
            col_labels = ["PC1 Feature", "Abs Loading", "PC2 Feature", "Abs Loading"]

        return table_data, col_labels

    def _add_pca_highlight_overlay(
        self,
        ax,
        plot_scores: np.ndarray,
        highlight_mask: np.ndarray,
    ) -> Line2D | None:
        if not np.any(highlight_mask):
            return None

        ax.scatter(
            plot_scores[highlight_mask, 0],
            plot_scores[highlight_mask, 1],
            marker="^",
            c="red",
            edgecolors="black",
            linewidths=0.6,
            s=40,
            zorder=5,
        )
        return Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=8,
            label="Highlighted region",
        )

    def _format_kmeans_cluster_metrics(
        self,
        metrics: Dict[str, Optional[float]],
    ) -> str:
        def render_metric(metric_key: str) -> str:
            metric_value = metrics.get(metric_key)
            if metric_value is None or not np.isfinite(metric_value):
                return "n/a"
            return f"{metric_value:.3f}"

        return "\n".join(
            [
                f"Silhouette: {render_metric('silhouette_score')}",
                f"Davies-Bouldin: {render_metric('davies_bouldin_score')}",
            ]
        )

    def _add_pca_metrics_annotation(
        self,
        ax,
        metrics_text: Optional[str],
    ) -> None:
        if not metrics_text:
            return
        ax.text(
            0.98,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )

    def _add_pca_metrics_annotation_3d(
        self,
        ax,
        metrics_text: Optional[str],
    ) -> None:
        if not metrics_text:
            return
        ax.text2D(
            0.98,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )

    def _add_pca_metrics_annotation_plotly(
        self,
        fig,
        metrics_text: Optional[str],
    ) -> None:
        if not metrics_text:
            return
        fig.update_layout(
            annotations=[
                {
                    "x": 0.98,
                    "y": 0.02,
                    "xref": "paper",
                    "yref": "paper",
                    "text": metrics_text.replace("\n", "<br>"),
                    "showarrow": False,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "align": "right",
                    "bgcolor": "rgba(255, 255, 255, 0.85)",
                    "bordercolor": "black",
                    "borderwidth": 1,
                }
            ]
        )

    def _normalize_plot_labels(
        self,
        labels,
        expected_length: int,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1:
            labels_array = labels_array.reshape(-1)

        if len(labels_array) != expected_length:
            raise ValueError(
                "labels and emission_df must be the same length. "
                f"Received {len(labels_array)} labels for {expected_length} emission rows. "
                "If you are plotting HMM labels, make sure they come from the same "
                "meth_data/emissions_df pair rather than from a separately sampled "
                "training table."
            )

        if len(labels_array) == 0:
            return np.array([], dtype=int)

        first_value = labels_array[0]
        if isinstance(first_value, Enum):
            return np.array([label.value for label in labels_array], dtype=int)

        if isinstance(first_value, str):
            label_map = {state.name.lower(): state.value for state in MethylationStates}
            normalized = []
            for label in labels_array:
                label_key = str(label).strip().lower()
                if label_key not in label_map:
                    raise ValueError(
                        "String labels must match methylation state names: "
                        f"{sorted(label_map)}. Received: {label!r}"
                    )
                normalized.append(label_map[label_key])
            return np.array(normalized, dtype=int)

        return labels_array.astype(int)

    def _add_pca_biplot_overlay(
        self,
        ax,
        pca: PCA,
        plot_scores: np.ndarray,
        feature_names: List[str],
        top_n_loadings: int,
    ) -> None:
        if plot_scores.shape[1] < 2 or pca.components_.shape[0] < 2:
            raise ValueError("Biplot overlay requires at least two PCA components.")

        loadings = pd.DataFrame(
            pca.components_.T[:, :2],
            columns=["PC1", "PC2"],
            index=feature_names,
        )
        loading_magnitude = np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2)
        top_features = loading_magnitude.sort_values(ascending=False).head(
            top_n_loadings
        )
        if top_features.empty:
            return

        score_scale = np.max(np.abs(plot_scores[:, :2]), axis=0)
        score_scale = np.where(score_scale == 0, 1.0, score_scale)
        loading_scale = np.max(
            np.abs(loadings.loc[top_features.index, ["PC1", "PC2"]].to_numpy()),
            axis=0,
        )
        loading_scale = np.where(loading_scale == 0, 1.0, loading_scale)
        arrow_scale = 0.8 * np.min(score_scale / loading_scale)
        head_width = max(0.02 * np.max(score_scale), 1e-6)

        for feature_name in top_features.index:
            x_loading = float(loadings.loc[feature_name, "PC1"]) * arrow_scale
            y_loading = float(loadings.loc[feature_name, "PC2"]) * arrow_scale
            ax.arrow(
                0,
                0,
                x_loading,
                y_loading,
                color="black",
                width=0.0,
                head_width=head_width,
                length_includes_head=True,
                alpha=0.8,
                zorder=6,
            )
            ax.text(
                x_loading * 1.08,
                y_loading * 1.08,
                self._wrap_pca_loading_feature_name(feature_name, width=18),
                fontsize=8,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=7,
            )

    def _plot_pca_clusters_impl(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        n_pca_plot: int = 2,
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,
        interactive: bool = False,
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        sample_name: str | None = None,
        chrom: str | None = None,
        highlight_mask: Optional[np.ndarray] = None,
    ):
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        pca, plot_scores, feature_names, _ = self._fit_plot_pca(
            emission_df=emission_df,
            n_pca_plot=n_pca_plot,
        )
        cmap, norm, state_colors_rgba, state_colors_hex = get_biological_state_colors()

        labels_numeric = self._normalize_plot_labels(
            labels=labels,
            expected_length=plot_scores.shape[0],
        )
        present_states = get_present_biological_states(labels_numeric)
        if highlight_mask is not None:
            highlight_mask = np.asarray(highlight_mask, dtype=bool)
            if highlight_mask.ndim != 1 or len(highlight_mask) != plot_scores.shape[0]:
                raise ValueError(
                    "highlight_mask must be a 1D boolean array aligned to emission_df."
                )
            if n_pca_plot != 2 or interactive:
                raise ValueError(
                    "Region highlighting is currently supported only for 2-D "
                    "non-interactive PCA plots."
                )

        n_pca = pca.n_components_
        explained_variance = pca.explained_variance_ratio_
        pc_axis_labels = [
            f"PC{i+1} ({explained_variance[i] * 100:.1f}%)" for i in range(n_pca)
        ]
        metrics_text = None
        if include_kmeans_metrics:
            metrics_text = self._format_kmeans_cluster_metrics(
                self.calculate_kmeans_cluster_metrics(
                    emission_df=emission_df,
                    labels=labels,
                )
            )
        table_data, col_labels = self._build_pca_loadings_table(
            pca=pca,
            feature_names=feature_names,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
        )

        fig = plt.figure(figsize=(22 if n_pca_plot == 3 else 20, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.35])

        # Title
        title_parts = []
        if sample_name is not None:
            title_parts.append(str(sample_name))
        if chrom is not None:
            title_parts.append(str(chrom))
        title_prefix = " ".join(title_parts) if title_parts else "Sample"
        title_basis = "PCA"

        # --- PCA embedding ---
        if n_pca_plot == 2:
            ax0 = fig.add_subplot(gs[0])
            highlight_handle = None

            if pca_hexbin:
                gridsize = 60
                cluster_cmaps = []
                for state_value in present_states:
                    base_color = state_colors_rgba[state_value]
                    cluster_cmaps.append(
                        mcolors.LinearSegmentedColormap.from_list(
                            f"cluster_{state_value}", [(1, 1, 1, 0.0), base_color]
                        )
                    )

                for cmap_idx, state_value in enumerate(present_states):
                    mask = labels_numeric == state_value
                    if not np.any(mask):
                        continue
                    ax0.hexbin(
                        plot_scores[mask, 0],
                        plot_scores[mask, 1],
                        gridsize=gridsize,
                        bins="log",
                        mincnt=1,
                        cmap=cluster_cmaps[cmap_idx],
                    )

                ax0.set_xlabel(pc_axis_labels[0])
                ax0.set_ylabel(pc_axis_labels[1])
                ax0.set_title(
                    f"{title_prefix}: PCA + {label_title}s (2-D Hexbin, {title_basis})"
                )
                self._add_pca_metrics_annotation(ax0, metrics_text)
                if highlight_mask is not None:
                    highlight_handle = self._add_pca_highlight_overlay(
                        ax=ax0,
                        plot_scores=plot_scores,
                        highlight_mask=highlight_mask,
                    )

                from matplotlib.patches import Patch

                handles = [
                    Patch(
                        color=state_colors_rgba[state_value],
                        label=MethylationStates(state_value).name,
                    )
                    for state_value in present_states
                ]
                if highlight_handle is not None:
                    handles.append(highlight_handle)
                ax0.legend(handles=handles, title=label_title, loc="best")

            else:
                sc0 = ax0.scatter(
                    plot_scores[:, 0],
                    plot_scores[:, 1],
                    c=labels_numeric,
                    cmap=cmap,
                    norm=norm,
                    s=8,
                )
                ax0.set_xlabel(pc_axis_labels[0])
                ax0.set_ylabel(pc_axis_labels[1])
                ax0.set_title(
                    f"{title_prefix}: PCA + {label_title}s (2-D, {title_basis})"
                )
                self._add_pca_metrics_annotation(ax0, metrics_text)
                cbar = plt.colorbar(
                    sc0,
                    ax=ax0,
                    ticks=present_states,
                    label=label_title,
                )
                cbar.set_ticklabels(
                    [
                        MethylationStates(state_value).name
                        for state_value in present_states
                    ]
                )
                if highlight_mask is not None:
                    highlight_handle = self._add_pca_highlight_overlay(
                        ax=ax0,
                        plot_scores=plot_scores,
                        highlight_mask=highlight_mask,
                    )
                    if highlight_handle is not None:
                        ax0.legend(
                            handles=[highlight_handle],
                            title="Overlay",
                            loc="best",
                        )

            if include_biplot:
                self._add_pca_biplot_overlay(
                    ax=ax0,
                    pca=pca,
                    plot_scores=plot_scores,
                    feature_names=feature_names,
                    top_n_loadings=top_n_loadings,
                )

        else:
            # 3-D PCA
            if include_biplot:
                raise ValueError(
                    "include_biplot is currently supported only for 2-D PCA plots."
                )
            if interactive and plot_scores.shape[1] >= 3:
                labels_str = np.array(
                    [MethylationStates(int(lbl)).name for lbl in labels_numeric]
                )
                fig_plotly = px.scatter_3d(
                    x=plot_scores[:, 0],
                    y=plot_scores[:, 1],
                    z=plot_scores[:, 2],
                    color=labels_str,
                    color_discrete_map={
                        MethylationStates(state_value).name: state_colors_hex[
                            state_value
                        ]
                        for state_value in present_states
                    },
                    category_orders={
                        "color": [
                            MethylationStates(state_value).name
                            for state_value in present_states
                        ]
                    },
                    labels={
                        "x": pc_axis_labels[0],
                        "y": pc_axis_labels[1],
                        "z": pc_axis_labels[2],
                        "color": label_title,
                    },
                    title=(
                        f"{title_prefix}: PCA + {label_title}s "
                        f"(3-D Interactive, {title_basis})"
                    ),
                )
                self._add_pca_metrics_annotation_plotly(fig_plotly, metrics_text)
                fig_plotly.update_traces(marker=dict(size=3))
                fig_plotly.show(renderer="notebook")
            else:
                ax0 = fig.add_subplot(gs[0], projection="3d")
                sc0 = ax0.scatter(
                    plot_scores[:, 0],
                    plot_scores[:, 1],
                    plot_scores[:, 2],
                    c=labels_numeric,
                    cmap=cmap,
                    norm=norm,
                    s=8,
                )
                ax0.set_xlabel(pc_axis_labels[0])
                ax0.set_ylabel(pc_axis_labels[1])
                ax0.set_zlabel(pc_axis_labels[2])
                ax0.set_title(
                    f"{title_prefix}: PCA + {label_title}s (3-D, {title_basis})"
                )
                self._add_pca_metrics_annotation_3d(ax0, metrics_text)
                cbar = fig.colorbar(sc0, ax=ax0, ticks=present_states, shrink=0.6)
                cbar.set_label(label_title)
                cbar.set_ticklabels(
                    [
                        MethylationStates(state_value).name
                        for state_value in present_states
                    ]
                )

        # --- Loadings table ---
        ax1 = fig.add_subplot(gs[1])
        ax1.axis("off")
        table = ax1.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)
        try:
            table.auto_set_column_width(col=list(range(len(col_labels))))
        except AttributeError:
            pass
        feature_col_indices = list(range(0, len(col_labels), 2))
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_ha("center")
                continue
            if col_idx in feature_col_indices:
                cell.get_text().set_ha("left")
            else:
                cell.get_text().set_ha("center")
        ax1.set_title(f"Top {top_n_loadings} Absolute Loadings", pad=10)

        plt.tight_layout()
        plt.show()

    def plot_pca_clusters_with_region(
        self,
        meth_data: pd.DataFrame,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        region_start: int,
        region_end: int,
        region_chrom: Optional[str] = None,
        n_pca_plot: int = 2,
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,
        interactive: bool = False,
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        sample_name: str | None = None,
        chrom: str | None = None,
    ):
        required_cols = {"CpG_chrm", "CpG_beg", "CpG_end"}
        missing_cols = sorted(required_cols - set(meth_data.columns))
        if missing_cols:
            raise ValueError(
                "meth_data must contain genomic coordinate columns for region "
                f"highlighting. Missing: {missing_cols}"
            )

        if len(meth_data) != len(emission_df) or len(meth_data) != len(labels):
            raise ValueError(
                "meth_data, emission_df, and labels must be the same length."
            )

        resolved_region_start = int(region_start)
        resolved_region_end = int(region_end)
        if resolved_region_end < resolved_region_start:
            raise ValueError(
                "region_end must be greater than or equal to region_start."
            )

        meth_chroms = meth_data["CpG_chrm"].astype(str)
        if region_chrom is None:
            unique_chroms = meth_chroms.unique()
            if len(unique_chroms) != 1:
                raise ValueError(
                    "region_chrom is required when meth_data contains multiple "
                    "chromosomes."
                )
            resolved_region_chrom = str(unique_chroms[0])
        else:
            resolved_region_chrom = str(region_chrom)

        cpg_beg = pd.to_numeric(meth_data["CpG_beg"], errors="raise").to_numpy(
            dtype=np.int64
        )
        cpg_end = pd.to_numeric(meth_data["CpG_end"], errors="raise").to_numpy(
            dtype=np.int64
        )
        highlight_mask = (
            (meth_chroms.to_numpy() == resolved_region_chrom)
            & (cpg_beg <= resolved_region_end)
            & (cpg_end >= resolved_region_start)
        )

        if not np.any(highlight_mask):
            warnings.warn(
                "No CpGs overlapped the requested genomic region for PCA highlighting: "
                f"{resolved_region_chrom}:{resolved_region_start}-{resolved_region_end}.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self._plot_pca_clusters_impl(
            emission_df=emission_df,
            labels=labels,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
            pca_hexbin=pca_hexbin,
            interactive=interactive,
            include_kmeans_metrics=include_kmeans_metrics,
            include_biplot=include_biplot,
            label_title=label_title,
            sample_name=sample_name,
            chrom=chrom,
            highlight_mask=highlight_mask,
        )

    def _format_train_chrom_label(
        self,
    ) -> str | None:
        train_chroms = getattr(self, "train_chroms", None)
        if train_chroms is None:
            legacy_train_chrom = getattr(self, "train_chrom", None)
            train_chroms = (
                [legacy_train_chrom] if legacy_train_chrom is not None else None
            )

        if not train_chroms:
            return None

        chroms = [str(chrom) for chrom in train_chroms]
        if len(chroms) == 1:
            return chroms[0]
        if set(chroms) == set(CANONICAL_AUTOSOMES) and len(chroms) == len(
            CANONICAL_AUTOSOMES
        ):
            return "autosomes"
        if len(chroms) <= 3:
            return ",".join(chroms)
        return f"{len(chroms)} chromosomes"

    def plot_train_pca_clusters(
        self,
        n_pca_plot: int = 2,
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,
        interactive: bool = False,
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        region_start: Optional[int] = None,
        region_end: Optional[int] = None,
        region_chrom: Optional[str] = None,
    ):
        """
        Convenience wrapper to plot the PCA embedding saved from k-means training.
        """
        region_requested = any(
            value is not None for value in (region_start, region_end, region_chrom)
        )
        if region_requested and (region_start is None or region_end is None):
            raise ValueError(
                "region_start and region_end must both be provided when requesting "
                "region highlighting."
            )

        required_attrs = ["train_emission_df", "train_labels"]
        if region_requested:
            required_attrs.append("train_meth")

        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise ValueError(
                "No saved training clustering artifacts found. "
                f"Missing attributes: {missing}. Train k-means first."
            )

        resolved_sample_name = getattr(self, "train_sample", None)
        resolved_chrom_label = self._format_train_chrom_label()

        if region_start is not None and region_end is not None:
            return self.plot_pca_clusters_with_region(
                meth_data=self.train_meth,
                emission_df=self.train_emission_df,
                labels=self.train_labels,
                region_start=region_start,
                region_end=region_end,
                region_chrom=region_chrom,
                n_pca_plot=n_pca_plot,
                top_n_loadings=top_n_loadings,
                pca_hexbin=pca_hexbin,
                interactive=interactive,
                include_kmeans_metrics=include_kmeans_metrics,
                include_biplot=include_biplot,
                sample_name=resolved_sample_name,
                chrom=resolved_chrom_label,
            )

        return self.plot_pca_clusters(
            emission_df=self.train_emission_df,
            labels=self.train_labels,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
            pca_hexbin=pca_hexbin,
            interactive=interactive,
            include_kmeans_metrics=include_kmeans_metrics,
            include_biplot=include_biplot,
            sample_name=resolved_sample_name,
            chrom=resolved_chrom_label,
        )

    def _subset_emission_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        windows_to_use: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if windows_to_use is None:
            return X, feature_names

        keep_cols = ["beta"]
        for label in windows_to_use:
            keep_cols.extend(
                [
                    f"{label}_avg_meth",
                    f"{label}_std",
                    f"{label}_high_pct",
                    f"{label}_int_pct",
                    f"{label}_low_pct",
                    f"{label}_n_cpg",
                ]
            )

        col_indices = [feature_names.index(c) for c in keep_cols]
        return X[:, col_indices], keep_cols

    def _prepare_filtered_sample_for_clustering(
        self,
        meth_data: pd.DataFrame,
        windows_to_use: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        if len(meth_data) == 0:
            raise ValueError("No CpGs remaining after filtering.")

        positions = meth_data["CpG_beg"].to_numpy(dtype=np.int64)
        betas = meth_data["beta"].to_numpy(dtype=np.float64)

        order = np.argsort(positions, kind="mergesort")
        positions = positions[order]
        betas = betas[order]
        meth_data = meth_data.iloc[order].reset_index(drop=True)

        X, feature_names = self.build_emission_matrix(
            positions=positions,
            betas=betas,
            window_specs=self.window_specs,
            int_low_cutoff=self.int_low_cutoff,
            int_high_cutoff=self.int_high_cutoff,
            high_cutoff=self.high_cutoff,
        )
        X, feature_names = self._subset_emission_features(
            X=X,
            feature_names=feature_names,
            windows_to_use=windows_to_use,
        )
        emission_df = pd.DataFrame(X, columns=feature_names)
        return meth_data, X, emission_df

    def prepare_sample_for_clustering(
        self,
        sample_info: SampleInfo,
        chrom: Optional[str] = None,
        windows_to_use: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Convenience wrapper to:
        1. Filter methylation DataFrame (optionally by chromosome).
        2. Build emission matrix using NumPy sliding windows.
        3. Optionally subset windows.

        Returns:
            meth_data (filtered DataFrame),
            emission_matrix (np.ndarray),
            emission_df (pandas DataFrame view)
        """

        meth_data = sample_info.meth_data.copy()

        if chrom is not None:
            meth_data = meth_data[meth_data["CpG_chrm"] == chrom]
            return self._prepare_filtered_sample_for_clustering(
                meth_data=meth_data,
                windows_to_use=windows_to_use,
            )

        if len(meth_data) == 0:
            raise ValueError("No CpGs remaining after filtering.")

        chrom_series = meth_data["CpG_chrm"].astype(str)
        chrom_order = chrom_series.drop_duplicates().tolist()
        if len(chrom_order) <= 1:
            return self._prepare_filtered_sample_for_clustering(
                meth_data=meth_data,
                windows_to_use=windows_to_use,
            )

        meth_frames = []
        emission_frames = []
        emission_arrays = []

        for chrom_name in chrom_order:
            chrom_meth = meth_data[chrom_series == chrom_name].copy()
            chrom_meth, chrom_X, chrom_emission_df = (
                self._prepare_filtered_sample_for_clustering(
                    meth_data=chrom_meth,
                    windows_to_use=windows_to_use,
                )
            )
            meth_frames.append(chrom_meth)
            emission_arrays.append(chrom_X)
            emission_frames.append(chrom_emission_df)

        combined_meth = pd.concat(meth_frames, ignore_index=True)
        combined_X = np.vstack(emission_arrays)
        combined_emission_df = pd.concat(emission_frames, ignore_index=True)
        return combined_meth, combined_X, combined_emission_df

    def _resolve_train_chroms(
        self,
        sample_info: SampleInfo,
        train_chroms: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        available_chroms = set(sample_info.meth_data["CpG_chrm"].astype(str).unique())

        if train_chroms is None:
            resolved = [
                chrom for chrom in CANONICAL_AUTOSOMES if chrom in available_chroms
            ]
            missing = []
        else:
            resolved = []
            missing = []
            for chrom in train_chroms:
                chrom = str(chrom)
                if chrom in available_chroms:
                    resolved.append(chrom)
                else:
                    missing.append(chrom)

        if not resolved:
            requested = (
                list(train_chroms)
                if train_chroms is not None
                else list(CANONICAL_AUTOSOMES)
            )
            raise ValueError(
                "No eligible training chromosomes remained after filtering. "
                f"Requested: {requested}"
            )

        return resolved, missing

    def _prepare_training_data_for_kmeans(
        self,
        sample_info: SampleInfo,
        train_chroms: Optional[List[str]] = None,
        windows_to_use: Optional[List[str]] = None,
        max_cpg_per_chrom: Optional[int] = 50_000,
        sampling_random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if max_cpg_per_chrom is not None and max_cpg_per_chrom <= 0:
            raise ValueError("max_cpg_per_chrom must be > 0 or None.")

        resolved_chroms, missing_chroms = self._resolve_train_chroms(
            sample_info=sample_info,
            train_chroms=train_chroms,
        )

        seed = (
            self.random_state
            if sampling_random_state is None
            else sampling_random_state
        )
        rng = np.random.default_rng(seed)

        meth_frames = []
        emission_frames = []
        summary_rows = []

        for chrom in resolved_chroms:
            meth_data, _, emission_df = self.prepare_sample_for_clustering(
                sample_info=sample_info,
                chrom=chrom,
                windows_to_use=windows_to_use,
            )

            n_total = len(meth_data)
            if max_cpg_per_chrom is not None and n_total > max_cpg_per_chrom:
                sampled_idx = np.sort(
                    rng.choice(n_total, size=max_cpg_per_chrom, replace=False)
                )
                was_sampled = True
            else:
                sampled_idx = np.arange(n_total)
                was_sampled = False

            meth_frames.append(meth_data.iloc[sampled_idx].reset_index(drop=True))
            emission_frames.append(emission_df.iloc[sampled_idx].reset_index(drop=True))

            summary_rows.append(
                {
                    "chrom": chrom,
                    "available": True,
                    "selected_for_training": True,
                    "n_cpg_total": int(n_total),
                    "n_cpg_sampled": int(len(sampled_idx)),
                    "sampled": was_sampled,
                    "sampling_fraction": float(len(sampled_idx) / n_total),
                }
            )

        for chrom in missing_chroms:
            summary_rows.append(
                {
                    "chrom": chrom,
                    "available": False,
                    "selected_for_training": False,
                    "n_cpg_total": 0,
                    "n_cpg_sampled": 0,
                    "sampled": False,
                    "sampling_fraction": 0.0,
                }
            )

        train_meth_data = pd.concat(meth_frames, ignore_index=True)
        train_emission_df = pd.concat(emission_frames, ignore_index=True)
        training_summary_df = pd.DataFrame(summary_rows)

        return train_meth_data, train_emission_df, training_summary_df

    # def prepare_sample_for_clustering(
    #     self,
    #     sample_info: SampleInfo,
    #     chrom: Optional[str] = None,
    #     windows_to_use: Optional[List[str]] = None,
    # ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     """
    #     Convenience wrapper to:
    #     1. Load methylation for a sample (optionally filtered to `chrom`).
    #     2. Compute multi-window summary stats (full chromosome).
    #     3. Create emission_df with only selected windows.

    #     Returns (meth_data, summary_stats, emission_df).
    #     """
    #     meth_data = sample_info.meth_data
    #     if chrom is not None:
    #         meth_data = meth_data[meth_data["CpG_chrm"] == chrom]

    #     summary_stats = self.generate_multi_window_summary_centered(
    #         meth_data,
    #         chrom=chrom,
    #     )

    #     emission_df = self.create_emission_df(
    #         summary_stats, windows_to_use=windows_to_use
    #     )
    #     return meth_data, summary_stats, emission_df

    def train_kmeans_for_sample(
        self,
        sample_info: SampleInfo,
        train_chroms: Optional[List[str]] = None,
        windows_to_use: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        max_cpg_per_chrom: Optional[int] = 50_000,
        sampling_random_state: Optional[int] = None,
    ):
        """
        High-level: given a sample ID and training chromosomes,
        train a KMeans model and return:
        - model (reusable)
        - meth_data
        - emission_df
        - pca_scores (or None when clustering in raw feature space)
        - labels
        """
        self.train_sample = sample_info.sample_id
        self.train_sample_info = sample_info
        meth_data, emission_df, training_summary_df = (
            self._prepare_training_data_for_kmeans(
                sample_info=sample_info,
                train_chroms=train_chroms,
                windows_to_use=windows_to_use,
                max_cpg_per_chrom=max_cpg_per_chrom,
                sampling_random_state=sampling_random_state,
            )
        )
        self.train_chroms, _ = self._resolve_train_chroms(
            sample_info=sample_info,
            train_chroms=train_chroms,
        )
        self.max_cpg_per_chrom = max_cpg_per_chrom
        self.train_meth = meth_data
        self.train_emission_df = emission_df
        self.training_summary_df = training_summary_df

        model, pca_scores, labels = self.fit_kmeans_on_emissions(
            emission_df=emission_df,
            feature_cols=feature_cols,
        )
        self.model = model
        self.train_pca_scores = pca_scores
        self.train_labels = labels

        return model, meth_data, emission_df, pca_scores, labels

    def apply_kmeans_to_sample(
        self,
        sample_info: SampleInfo,
        chrom: Optional[str] = None,
        windows_to_use: Optional[List[str]] = None,
        sample_meth_data: Optional[pd.DataFrame] = None,
    ):
        """
        High-level: apply an already-trained model to a new sample (same features).

        Returns:
        - meth_data
        - emission_df
        - pca_scores
        - raw_distances
        - raw_labels
        - relabeled_labels
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        meth_data, _emission_matrix, emission_df = self.prepare_sample_for_clustering(
            sample_info=sample_info,
            chrom=chrom,
            windows_to_use=windows_to_use,
        )

        missing = [c for c in self.model.feature_cols if c not in emission_df.columns]
        if missing:
            raise ValueError(
                f"emission_df for sample {sample_info.sample_id} is missing features: {missing}"
            )

        pca_scores, raw_distances, raw_labels, relabeled_labels = (
            self.apply_kmeans_to_emissions(emission_df)
        )
        return (
            meth_data,
            emission_df,
            pca_scores,
            raw_distances,
            raw_labels,
            relabeled_labels,
        )

    def get_pca_loadings(
        self,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of PCA loadings for the features used in the model.
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        pca = self.model.pca
        if pca is None:
            raise ValueError("model.pca is None: PCA loadings are not available.")

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=self.model.feature_cols,
        )
        return loadings


class MethylStateAnalyzer:
    def __init__(self, assigner: MethylStateAssigner, out_dir="."):
        self.assigner = assigner
        self.out_dir = out_dir
        self.train_joint = None
        self.window_specs = assigner.window_specs

    def _build_train_joint(self):
        if self.train_joint is not None:
            return
        if not hasattr(self.assigner, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        train_joint = pd.concat(
            [self.assigner.train_meth.copy(), self.assigner.train_emission_df.copy()],
            axis=1,
        )
        train_joint = train_joint.loc[:, ~train_joint.columns.duplicated()]
        train_joint["kmeans_label"] = self.assigner.train_labels
        self.train_joint = train_joint

    def plot_feature_distributions_by_kmeans_state(self, show_plots=True):
        self._build_train_joint()
        train_loadings = self.assigner.get_pca_loadings()
        for emission in train_loadings["PC2"].abs().sort_values(ascending=False).index:
            for state, df in self.train_joint.groupby("kmeans_label"):
                df[emission].hist(bins=50, alpha=0.5, label=f"{state}")
            plt.xlabel(emission)
            plt.ylabel("Count")
            plt.title(f"Distribution of {emission} by KMeans State")
            plt.legend()
            if show_plots:
                plt.show()
            elif self.out_dir is not None:
                plt.savefig(f"{self.out_dir}/feature_distribution_{emission}.png")
            plt.close()

    def define_states_by_rules_param(
        self,
        meth_emissions: pd.DataFrame,
        beta_low_max: float,
        beta_high_min: float,
        pmr_cutoffs: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        Rule-based state definition with tunable, per-window cutoffs.

        PMR is defined as:

          beta_low_max <= beta <= beta_high_min
          AND OR over regional windows of:
            {label}_int_pct >= label.int_min
            AND {label}_std <= label.std_max
            AND {label}_high_pct <= label.high_max
            AND {label}_low_pct <= label.low_max

        Parameters
        ----------
        meth_emissions : DataFrame
            Must contain:
              'beta',
              '{label}_int_pct', '{label}_std', '{label}_high_pct', '{label}_low_pct'
            for each window label in self.assigner.window_specs.
        beta_low_max : float
        beta_high_min : float
        pmr_cutoffs : dict
            {
              label: {
                'int_min': ...,
                'std_max': ...,
                'high_max': ...,
                'low_max': ...,
              },
              ...
            }

        States
        ------
        0 = low methylation outside PMRs
        1 = PMR
        2 = intermediate / other outside PMRs
        3 = high methylation outside PMRs
        """
        beta = meth_emissions["beta"].values
        n = len(meth_emissions)
        labels = np.full(len(meth_emissions), None, dtype=object)

        if not hasattr(self, "assigner"):
            raise AttributeError(
                "MethylStateAnalyzer must have an `assigner` with `window_specs`."
            )

        window_labels = [label for _, label in self.assigner.window_specs]
        if not window_labels:
            raise ValueError("No window_specs found on assigner.")
        regional_window_labels = self.assigner._get_regional_window_labels()

        pmr_window_masks = []

        for label in regional_window_labels:
            if label not in pmr_cutoffs:
                raise KeyError(
                    f"No PMR cutoffs provided for window '{label}'. "
                    f"Expected a key in pmr_cutoffs for each label in window_specs."
                )

            cfg = pmr_cutoffs[label]
            try:
                int_min = cfg["int_min"]
                std_max = cfg["std_max"]
                high_max = cfg["high_max"]
                low_max = cfg.get("low_max", high_max)
            except KeyError as e:
                raise KeyError(
                    f"PMR cutoff for window '{label}' must contain "
                    f"keys 'int_min', 'std_max', 'high_max'. Missing: {e}"
                )

            int_col = f"{label}_int_pct"
            std_col = f"{label}_std"
            high_col = f"{label}_high_pct"
            low_col = f"{label}_low_pct"

            if int_col not in meth_emissions.columns:
                raise KeyError(f"Column '{int_col}' not found in meth_emissions.")
            if std_col not in meth_emissions.columns:
                raise KeyError(f"Column '{std_col}' not found in meth_emissions.")
            if high_col not in meth_emissions.columns:
                raise KeyError(f"Column '{high_col}' not found in meth_emissions.")
            if low_col not in meth_emissions.columns:
                raise KeyError(f"Column '{low_col}' not found in meth_emissions.")

            int_vals = meth_emissions[int_col].values
            std_vals = meth_emissions[std_col].values
            high_vals = meth_emissions[high_col].values
            low_vals = meth_emissions[low_col].values

            pmr_window_masks.append(
                (int_vals >= int_min)
                & (std_vals <= std_max)
                & (high_vals <= high_max)
                & (low_vals <= low_max)
            )

        regional_any = np.logical_or.reduce(pmr_window_masks)
        pmr_mask = regional_any & (beta >= beta_low_max) & (beta <= beta_high_min)

        low_mask = (beta <= beta_low_max) & ~pmr_mask
        high_mask = (beta >= beta_high_min) & ~pmr_mask
        interm_mask = ~(pmr_mask | low_mask | high_mask)

        labels[low_mask] = MethylationStates.LOW
        labels[pmr_mask] = MethylationStates.PMR
        labels[interm_mask] = MethylationStates.INTERMEDIATE
        labels[high_mask] = MethylationStates.HIGH

        return labels

    def evaluate_rules_against_kmeans(
        self,
        meth_emissions: pd.DataFrame,
        kmeans_labels: np.ndarray = None,
        **rule_params,
    ):
        """
        Apply rule-based states and compare to KMeans labels.
        Returns (metrics_dict, rule_labels_array).

        rule_params must contain:
          - beta_low_max
          - beta_high_min
          - pmr_cutoffs (dict[label -> {'int_min','std_max','high_max','low_max'}])
        """
        y_true = np.asarray(kmeans_labels)
        y_pred = self.define_states_by_rules_param(meth_emissions, **rule_params)

        if isinstance(y_true.flat[0], Enum):
            y_true = np.array([lbl.value for lbl in y_true])
        if isinstance(y_pred.flat[0], Enum):
            y_pred_numeric = np.array([lbl.value for lbl in y_pred])
        else:
            y_pred_numeric = y_pred

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred_numeric),
            "F1_macro": f1_score(
                y_true, y_pred_numeric, average="macro", zero_division=0
            ),
            "F1_weighted": f1_score(
                y_true, y_pred_numeric, average="weighted", zero_division=0
            ),
            "Precision_macro": precision_score(
                y_true, y_pred_numeric, average="macro", zero_division=0
            ),
            "Recall_macro": recall_score(
                y_true, y_pred_numeric, average="macro", zero_division=0
            ),
            "ARI": adjusted_rand_score(y_true, y_pred_numeric),
            "NMI": normalized_mutual_info_score(y_true, y_pred_numeric),
        }

        return metrics, y_pred

    # TODO: improve the rules definition logic to improve accuracy when based against k-means, e.g. by adding interaction terms or more complex combinations of stats across windows.
    def optimize_rule_params_random(
        self,
        n_iter: int = 500,
        score_key: str = "F1_macro",
        random_state: int = 42,
        param_distributions: dict | None = None,
    ):
        """
        Random-search optimization of rule parameters, similar in spirit to
        sklearn.model_selection.RandomizedSearchCV.

        param_distributions structure:
        -------------------------------
        {
          "beta_low_max": (low, high),
          "beta_high_min": (low, high),
          "pmr": {
            "<label>": {
              "int_min": (low, high),
              "std_max": (low, high),
              "high_max": (low, high),
              "low_max": (low, high),
            },
            ...
          }
        }

        If param_distributions is None, sensible defaults are built
        for all window labels in self.assigner.window_specs.
        """
        self._build_train_joint()
        rng = np.random.default_rng(random_state)

        # Build default distributions if none provided
        if param_distributions is None:
            window_labels = [label for _, label in self.assigner.window_specs]
            pmr_dist = {
                label: {
                    "int_min": (0.40, 0.90),
                    "std_max": (0.10, 0.40),
                    "high_max": (0.05, 0.40),
                    "low_max": (0.05, 0.40),
                }
                for label in window_labels
            }
            param_distributions = {
                "beta_low_max": (0.05, 0.35),
                "beta_high_min": (0.60, 0.95),
                "pmr": pmr_dist,
            }

        # Helper: sample a valid param set
        def sample_params():
            while True:
                beta_low_max = float(rng.uniform(*param_distributions["beta_low_max"]))
                beta_high_min = float(
                    rng.uniform(*param_distributions["beta_high_min"])
                )
                if beta_low_max >= beta_high_min:
                    continue

                pmr_cutoffs = {}
                for label, ranges in param_distributions["pmr"].items():
                    pmr_cutoffs[label] = {
                        "int_min": float(rng.uniform(*ranges["int_min"])),
                        "std_max": float(rng.uniform(*ranges["std_max"])),
                        "high_max": float(rng.uniform(*ranges["high_max"])),
                        "low_max": float(rng.uniform(*ranges["low_max"])),
                    }

                return {
                    "beta_low_max": beta_low_max,
                    "beta_high_min": beta_high_min,
                    "pmr_cutoffs": pmr_cutoffs,
                }

        def flatten_rule_params(params: dict) -> dict:
            out = {
                "beta_low_max": params["beta_low_max"],
                "beta_high_min": params["beta_high_min"],
            }
            for label, cfg in params["pmr_cutoffs"].items():
                out[f"{label}_int_min"] = cfg["int_min"]
                out[f"{label}_std_max"] = cfg["std_max"]
                out[f"{label}_high_max"] = cfg["high_max"]
                out[f"{label}_low_max"] = cfg["low_max"]
            return out

        best_score = -np.inf
        best_params = None
        best_metrics = None
        best_labels = None
        history = []

        for i in tqdm(range(n_iter), desc="Random search"):
            rule_params = sample_params()

            metrics, labels = self.evaluate_rules_against_kmeans(
                self.assigner.train_emission_df,
                self.train_joint["kmeans_label"],
                **rule_params,
            )
            score = metrics[score_key]

            record = {"iter": i, "score": score}
            record.update(flatten_rule_params(rule_params))
            history.append(record)

            if score > best_score:
                best_score = score
                best_params = rule_params
                best_metrics = metrics
                best_labels = labels

        history_df = pd.DataFrame(history)
        self.cutoffs_set_manually = False
        self.state_cutoffs = best_params

        states_by_rules = self.define_states_by_rules(
            sample_info=self.assigner.train_sample_info,
            sample_emissions=self.assigner.train_emission_df,
        )
        self.train_joint["rule_based_label"] = states_by_rules

        best_configs = pd.DataFrame([flatten_rule_params(best_params)])
        best_results = pd.DataFrame(best_metrics, index=[0])
        optimization_summary = pd.concat([best_configs, best_results], axis=1)
        if self.out_dir is not None:
            optimization_summary.to_csv(
                f"{self.out_dir}/rule_based_optimization_summary.csv", index=False
            )
        return best_params, best_metrics, best_labels, history_df

    def define_states_by_rules(
        self, sample_info: SampleInfo, chrom=None, sample_emissions: pd.DataFrame = None
    ) -> np.ndarray:
        if sample_emissions is not None:
            meth_emissions = sample_emissions
        else:
            test_meth, summary_stats, test_emissions = (
                self.assigner.prepare_sample_for_clustering(sample_info, chrom)
            )
            meth_emissions = test_emissions

        if not hasattr(self, "state_cutoffs"):
            raise ValueError(
                "State cutoffs not defined. Please run optimization or set cutoffs manually."
            )

        states_by_rules = self.define_states_by_rules_param(
            meth_emissions, **self.state_cutoffs
        )
        return states_by_rules

    def __set_from_config(self, config: MethylSegConfig):
        state_cfg = config.get("state_cutoffs", None)
        if state_cfg is not None:
            cutoffs = state_cfg.get("cutoffs", {})
            self.set_state_cutoffs(
                beta_low_max=cutoffs.get("beta_low_max"),
                beta_high_min=cutoffs.get("beta_high_min"),
                pmr_cutoffs=cutoffs.get("pmr_cutoffs"),
            )
            self.cutoffs_set_manually = bool(state_cfg.get("set_manually", False))

    def set_state_cutoffs_from_yaml(self, yaml_file: str):
        """
        Load state cutoffs from a YAML configuration file.

        Expected YAML structure:
        beta_low_max: float
        beta_high_min: float
        pmr_cutoffs:
          <label>:
            int_min: float
            std_max: float
            high_max: float
            low_max: float
          ...
        """
        config = MethylSegConfig.from_yaml(yaml_file).config
        self.__set_from_config(config)

    def set_state_cutoffs(
        self,
        beta_low_max: float | None = None,
        beta_high_min: float | None = None,
        pmr_cutoffs: dict | None = None,
    ):
        """
        Simple manual state cutoff setter with clean pythonic defaults.

        - If user does not provide pmr_cutoffs, defaults are used:
            int_min = 0.56
            std_max = 0.264
            high_max = 0.246
            low_max = 0.246
        - Defaults apply to every window given in assigner.window_specs
        """
        int_min_default = 0.56
        std_max_default = 0.264
        high_max_default = 0.246
        low_max_default = 0.246
        beta_high_max_default = 0.694
        beta_low_max_default = 0.290
        if beta_low_max is None:
            beta_low_max = beta_low_max_default
        if beta_high_min is None:
            beta_high_min = beta_high_max_default

        final_pmr_cutoffs = {}

        for _, label in self.assigner.window_specs:
            cfg = (pmr_cutoffs or {}).get(label, {})
            final_pmr_cutoffs[label] = {
                "int_min": cfg.get("int_min", int_min_default),
                "std_max": cfg.get("std_max", std_max_default),
                "high_max": cfg.get("high_max", high_max_default),
                "low_max": cfg.get("low_max", low_max_default),
            }

        # Save all cutoffs
        self.state_cutoffs = {
            "beta_low_max": float(beta_low_max),
            "beta_high_min": float(beta_high_min),
            "pmr_cutoffs": final_pmr_cutoffs,
        }

        self.cutoffs_set_manually = True

    def pretty_print_rules(self):
        """
        Print the current rule-based state definitions in a concise, human-readable format.

        PMR:
          beta_low_cutoff <= beta <= beta_high_cutoff
          AND OR over regional windows of:
            {label}_int_pct >= label.int_min
            AND {label}_std <= label.std_max
            AND {label}_high_pct <= label.high_max
            AND {label}_low_pct <= label.low_max

        Low methylation:
          beta <= low_cutoff AND NOT PMR

        Intermediate:
          low_cutoff < beta < high_cutoff AND NOT PMR

        High methylation:
          beta >= high_cutoff AND NOT PMR
        """
        if not hasattr(self, "state_cutoffs"):
            raise ValueError(
                "State cutoffs not defined. Please run optimization or set cutoffs manually."
            )

        c = self.state_cutoffs
        beta_low_max = c["beta_low_max"]
        beta_high_min = c["beta_high_min"]
        pmr_cutoffs = c["pmr_cutoffs"]
        regional_window_labels = self.assigner._get_regional_window_labels()

        print("PMR:")
        print(f"{beta_low_max:.3f} <= beta <= {beta_high_min:.3f}")

        regional_parts = []
        for label in regional_window_labels:
            cfg = pmr_cutoffs[label]
            low_max = cfg.get("low_max", cfg["high_max"])
            regional_parts.append(
                "("
                f"{label}_int_pct >= {cfg['int_min']:.3f} AND "
                f"{label}_std <= {cfg['std_max']:.3f} AND "
                f"{label}_high_pct <= {cfg['high_max']:.3f} AND "
                f"{label}_low_pct <= {low_max:.3f}"
                ")"
            )
        print("AND " + " OR ".join(regional_parts) + "\n")

        print("Low methylation:")
        print(f"beta <= {beta_low_max:.3f} AND NOT PMR\n")

        print("Intermediate methylation:")
        print(f"{beta_low_max:.3f} < beta < {beta_high_min:.3f} AND NOT PMR\n")

        print("High methylation:")
        print(f"beta >= {beta_high_min:.3f} AND NOT PMR\n")

    def evaluate_clustering_concordance(
        self,
        use_train_data: bool = True,
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
    ):
        """
        Evaluate rule-based labels against KMeans cluster labels as ground truth.
        """
        if not hasattr(self, "state_cutoffs"):
            raise ValueError(
                "State cutoffs not defined. Please run optimization or set cutoffs manually."
            )
        if use_train_data:
            self._build_train_joint()
            y_true = self.train_joint["kmeans_label"]
            y_pred = self.train_joint["rule_based_label"]
        else:
            y_true = self.assigner.apply_kmeans_to_sample(
                sample_info=sample_info, chrom=chrom
            )[5]
            y_pred = self.define_states_by_rules(sample_info=sample_info, chrom=chrom)

        # --- Ensure numpy arrays ---
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # --- Convert Enum → integer for sklearn ---
        if isinstance(y_true[0], Enum):
            y_true = np.array([lbl.value for lbl in y_true])
        if isinstance(y_pred[0], Enum):
            y_pred_numeric = np.array([lbl.value for lbl in y_pred])
        else:
            y_pred_numeric = y_pred

        # ⬇️ Detailed per-class performance
        print("\nClassification Report:")
        target_names = [s.name for s in MethylationStates]
        print(
            classification_report(
                y_true, y_pred_numeric, target_names=target_names, zero_division=0
            )
        )

        # ⬇️ Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred_numeric)
        cm_df = pd.DataFrame(
            cm,
            index=[label.name for label in MethylationStates],
            columns=[label.name for label in MethylationStates],
        )

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix — KMeans (True) vs Rule-Based (Pred)")
        plt.ylabel("True (KMeans)")
        plt.xlabel("Predicted (Rule-Based)")
        plt.tight_layout()
        plt.show()

        return cm_df

    def plot_interactive_beta_by_label(
        self,
        label_type: str = "kmeans",
        use_train_data: bool = True,
        chrom: str | None = None,
        sample_info: SampleInfo | None = None,
        sample_info_removed: pd.DataFrame | None = None,
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,  # ← added
        color_pmr_only: bool = False,
        color_regions_df: pd.DataFrame | None = None,
    ):
        """
        Interactive scatter: genomic position vs beta, colored by label.
        """
        # -----------------------
        # Build plotting DataFrame
        # -----------------------
        if use_train_data:
            self._build_train_joint()
            df_plot = self.train_joint.copy()
        else:
            if sample_info is None:
                raise ValueError(
                    "sample_info must be provided when use_train_data is False."
                )

            meth_data, emission_df, _, _, _, labels = (
                self.assigner.apply_kmeans_to_sample(
                    sample_info=sample_info, chrom=chrom
                )
            )

            df_plot = pd.concat([meth_data, emission_df], axis=1)
            df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
            df_plot["kmeans_label"] = labels

            if label_type == "rule_based":
                rule_labels = self.define_states_by_rules(
                    sample_info=sample_info,
                    chrom=chrom,
                    sample_emissions=emission_df,
                )
                df_plot["rule_based_label"] = rule_labels
        label_col = f"{label_type}_label"
        return _plot_interactive_beta_scatter(
            df_plot=df_plot,
            sample_info=sample_info,
            sample_info_removed=sample_info_removed,
            chrom=chrom,
            out_dir=self.out_dir,
            label_col=label_col,
            x_col=x_col,
            y_col=y_col,
            label_title=label_title,
            show_plot=show_plot,
            max_points=max_points,
            color_pmr_only=color_pmr_only,
            color_regions_df=color_regions_df,
        )


class MethylSegHMM:
    def __init__(self, n_states: int):
        raise NotImplementedError("MethylSegHMM is an abstract class")

    def fit(self, emissions, sample_info=None, chrom=None):
        raise NotImplementedError("fit method not implemented")

    def create_model(self):
        raise NotImplementedError("fit method not implemented")

    def predict(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_fit(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_predict(self, emissions):
        raise NotImplementedError("fit method not implemented")


# TODO implement and test my custom DAHMM to test speed and performance against CTHMM
class DAMethylSegHMM(MethylSegHMM):
    def __init__(self, n_states: int):
        raise NotImplementedError("DAMethylSegHMM is not implemented")

    def fit(self, emissions, sample_info=None, chrom=None):
        raise NotImplementedError("fit method not implemented")

    def create_model(self):
        raise NotImplementedError("fit method not implemented")

    def predict(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_fit(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_predict(self, emissions):
        raise NotImplementedError("fit method not implemented")


class MultinomialSegHMM(MethylSegHMM):

    def __init__(
        self,
        n_states: int,
        random_state: int = 42,
        n_iter: int = 30,
        alpha: float = 0.7,
    ):
        self.random_state = random_state
        self.n_states = n_states
        self.n_iter = n_iter
        self.alpha = alpha
        self.lengths = None

    def create_model(self):
        self.hmm_model = hmm.MultinomialHMM(
            n_components=self.n_states,
            n_iter=self.n_iter,  # EM iterations for transitions
            random_state=self.random_state,
        )

    def format_fit(self, emissions):
        return np.eye(self.n_states, dtype=int)[emissions]

    def format_predict(self, emissions):
        return self.format_fit(emissions)

    def fit(self, emissions, sample_info=None, chrom=None):
        fit_emissions = self.format_fit(emissions)
        if self.lengths is None:
            self.hmm_model.fit(fit_emissions)
        else:
            self.hmm_model.fit(fit_emissions, lengths=self.lengths)

    def predict(self, emissions):
        predict_emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(predict_emissions)
        return self.hmm_model.predict(predict_emissions, lengths=self.lengths)


class StickyCategoricalMethylSegHMM(MethylSegHMM):
    DEFAULT_STAY_PROB = 0.995
    EMISSION_MISMATCH_PROB = 0.01
    TRANSITION_PRIOR_STRENGTH = 50.0

    def __init__(
        self,
        n_states: int,
        random_state: int = 42,
        n_iter: int = 30,
        stay_prob: float = DEFAULT_STAY_PROB,
        emission_mismatch_prob: float = EMISSION_MISMATCH_PROB,
        transition_prior_strength: float = TRANSITION_PRIOR_STRENGTH,
        fit_transitions: bool = False,
    ):
        if not np.isfinite(stay_prob) or not 0 <= stay_prob <= 1:
            raise ValueError(
                f"stay_prob must be a finite value strictly between 0 and 1. "
                f"Received: {stay_prob!r}"
            )
        if (
            not np.isfinite(emission_mismatch_prob)
            or emission_mismatch_prob <= 0
            or emission_mismatch_prob >= 1
        ):
            raise ValueError(
                "emission_mismatch_prob must be a finite value in the "
                f"range [0, 1). Received: {emission_mismatch_prob!r}"
            )
        if not np.isfinite(transition_prior_strength) or transition_prior_strength < 0:
            raise ValueError(
                "transition_prior_strength must be a finite non-negative "
                f"value. Received: {transition_prior_strength!r}"
            )
        print("STAY PROB:", stay_prob)
        self.random_state = random_state
        self.n_states = n_states
        self.n_iter = n_iter
        self.stay_prob = float(stay_prob)
        self.emission_mismatch_prob = float(emission_mismatch_prob)
        self.transition_prior_strength = float(transition_prior_strength)
        self.fit_transitions = fit_transitions
        self.lengths = None

    def format_fit(self, emissions):
        emissions = np.asarray(emissions, dtype=int)
        return emissions.reshape(-1, 1)

    def format_predict(self, emissions):
        return self.format_fit(emissions)

    def make_sticky_transmat(
        self,
        n_states: Optional[int] = None,
        stay_prob: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build a 'sticky' transition matrix.

        stay_prob controls the diagonal self-transition probability for every
        state. Remaining mass is shared uniformly across off-diagonal entries.
        """
        if n_states is None:
            n_states = self.n_states
        if stay_prob is None:
            stay_prob = self.stay_prob
        if n_states < 1:
            raise ValueError("n_states must be positive.")
        if n_states == 1:
            return np.ones((1, 1), dtype=float)

        switch_prob = (1.0 - stay_prob) / (n_states - 1)
        trans = np.full((n_states, n_states), switch_prob, dtype=float)
        np.fill_diagonal(trans, stay_prob)
        return trans

    def make_emissionprob(self) -> np.ndarray:
        if self.n_states == 1:
            return np.ones((1, 1), dtype=float)

        eps = self.emission_mismatch_prob
        emission = np.full(
            (self.n_states, self.n_states),
            eps / (self.n_states - 1),
            dtype=float,
        )
        np.fill_diagonal(emission, 1.0 - eps)
        return emission

    def create_model(self):
        self.prior_trans = self.make_sticky_transmat()
        self.hmm_model = hmm.CategoricalHMM(
            n_components=self.n_states,
            n_features=self.n_states,
            n_iter=self.n_iter,
            algorithm="viterbi",
            init_params="",
            params="t" if self.fit_transitions else "",
            random_state=self.random_state,
            transmat_prior=(1.0 + self.transition_prior_strength * self.prior_trans),
        )

        self.hmm_model.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.hmm_model.emissionprob_ = self.make_emissionprob()
        self.hmm_model.transmat_ = self.prior_trans.copy()

    def fit(self, emissions, sample_info=None, chrom=None):
        if self.fit_transitions:
            fit_emissions = self.format_fit(emissions)
            if self.lengths is None:
                self.hmm_model.fit(fit_emissions)
            else:
                self.hmm_model.fit(fit_emissions, lengths=self.lengths)

    def predict(self, emissions):
        predict_emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(predict_emissions)
        return self.hmm_model.predict(predict_emissions, lengths=self.lengths)


class GaussianMethylSegHMM(MethylSegHMM):
    TRANSITION_FLOOR = 1e-6
    COVARIANCE_FLOOR = 1e-3

    def __init__(
        self,
        n_states: int,
        random_state: int = 42,
        n_iter: int = 100,
        tol: float = 1e-3,
        covariance_type: str = "diag",
        init_params: str = "",
        params: str = "stmc",
    ):
        if covariance_type != "diag":
            raise ValueError(
                "GaussianMethylSegHMM currently supports only covariance_type='diag'."
            )
        self.random_state = random_state
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.params = params
        self.lengths = None

    def create_model(self):
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params=self.init_params,
            params=self.params,
        )

    def format_fit(self, emissions):
        emissions = np.asarray(emissions, dtype=np.float64)
        if emissions.ndim != 2:
            raise ValueError(
                "GaussianMethylSegHMM expects a 2D emission matrix shaped "
                "(n_observations, n_features)."
            )
        return emissions

    def format_predict(self, emissions):
        return self.format_fit(emissions)

    def _get_sequence_start_indices(
        self,
        n_observations: int,
        lengths: Optional[List[int]] = None,
    ) -> np.ndarray:
        if lengths is None:
            return np.array([0], dtype=int)

        if sum(lengths) != n_observations:
            raise ValueError(
                "The provided sequence lengths do not sum to the number of "
                "emission rows."
            )

        starts = [0]
        offset = 0
        for seq_len in lengths[:-1]:
            offset += int(seq_len)
            starts.append(offset)
        return np.asarray(starts, dtype=int)

    def _get_valid_transition_mask(
        self,
        n_observations: int,
        lengths: Optional[List[int]] = None,
    ) -> np.ndarray:
        if n_observations < 2:
            return np.zeros(0, dtype=bool)

        valid_mask = np.ones(n_observations - 1, dtype=bool)
        if lengths is None:
            return valid_mask

        boundary = 0
        for seq_len in lengths[:-1]:
            boundary += int(seq_len)
            valid_mask[boundary - 1] = False
        return valid_mask

    def initialize_from_kmeans(
        self,
        X_scaled: np.ndarray,
        km_labels: np.ndarray,
        lengths: Optional[List[int]] = None,
    ):
        X_scaled = self.format_fit(X_scaled)
        km_labels = np.asarray(km_labels, dtype=int)
        if len(X_scaled) != len(km_labels):
            raise ValueError("X_scaled and km_labels must have the same length.")
        if len(X_scaled) == 0:
            raise ValueError("Cannot initialize a Gaussian HMM on empty emissions.")
        if km_labels.min() < 0 or km_labels.max() >= self.n_states:
            raise ValueError(
                "KMeans initialization labels must be in the range "
                f"[0, {self.n_states - 1}] for the configured Gaussian HMM."
            )

        self.lengths = None if lengths is None else [int(length) for length in lengths]

        start_indices = self._get_sequence_start_indices(
            n_observations=len(km_labels),
            lengths=self.lengths,
        )
        startprob = np.bincount(
            km_labels[start_indices],
            minlength=self.n_states,
        ).astype(np.float64)
        startprob_sum = startprob.sum()
        if startprob_sum == 0:
            startprob = np.full(self.n_states, 1.0 / self.n_states, dtype=np.float64)
        else:
            startprob /= startprob_sum

        transmat = np.full(
            (self.n_states, self.n_states),
            self.TRANSITION_FLOOR,
            dtype=np.float64,
        )
        valid_transitions = self._get_valid_transition_mask(
            n_observations=len(km_labels),
            lengths=self.lengths,
        )
        for start_state, end_state in zip(
            km_labels[:-1][valid_transitions],
            km_labels[1:][valid_transitions],
        ):
            transmat[int(start_state), int(end_state)] += 1.0
        transmat /= transmat.sum(axis=1, keepdims=True)

        global_mean = np.mean(X_scaled, axis=0)
        global_var = np.var(X_scaled, axis=0) + self.COVARIANCE_FLOOR

        means = np.zeros((self.n_states, X_scaled.shape[1]), dtype=np.float64)
        covars = np.zeros((self.n_states, X_scaled.shape[1]), dtype=np.float64)
        for state_idx in range(self.n_states):
            members = X_scaled[km_labels == state_idx]
            if len(members) == 0:
                means[state_idx] = global_mean
                covars[state_idx] = global_var
                continue
            means[state_idx] = np.mean(members, axis=0)
            covars[state_idx] = np.var(members, axis=0) + self.COVARIANCE_FLOOR

        self.hmm_model.startprob_ = startprob
        self.hmm_model.transmat_ = transmat
        self.hmm_model.means_ = means
        self.hmm_model.covars_ = covars

    def fit(self, emissions, sample_info=None, chrom=None):
        emissions = self.format_fit(emissions)
        if self.lengths is None:
            self.hmm_model.fit(emissions)
        else:
            self.hmm_model.fit(emissions, lengths=self.lengths)

    def predict(self, emissions):
        emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(emissions)
        return self.hmm_model.predict(emissions, lengths=self.lengths)


class CTMethylSegHMM(MethylSegHMM):
    def __init__(
        self,
        n_states,
        n_emissions,
        holding_time_guess,
        time_scale: float = 1,
        max_iter: int = 20,
        tol: float = 1e-4,
        random_state: int = 42,
        algorithm="forward-backward",
    ):
        self.random_state = random_state
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.holding_time_guess = holding_time_guess
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.time_scale = time_scale

    def create_model(self):
        eps = 0.02  # mislabel rate

        emission_probs = np.full(
            (self.n_states, self.n_emissions), eps / (self.n_emissions - 1)
        )
        np.fill_diagonal(emission_probs, 1.0 - eps)
        self.hmm_model = cthmm.MultinomialCTHMM(
            n_states=self.n_states,
            n_emissions=self.n_emissions,
            emission_probs=emission_probs,  # our near-identity matrix
            holding_time=self.holding_time_guess,  # library builds a default Q from this
            seed=self.random_state,
        )

    def format_fit(self, emissions):
        obs_states = emissions
        times = self.times
        # print(len(obs_states), len(times))
        return [(obs_states, times)]

    def format_predict(self, emissions):
        obs_states = emissions
        times = self.times
        return (obs_states, times)

    def fit(self, emissions, sample_info, chrom):
        self.times = (
            sample_info.meth_data[sample_info.meth_data["CpG_chrm"] == chrom][
                "CpG_beg"
            ].values
            / self.time_scale
        )
        return self.hmm_model.fit(
            self.format_fit(emissions),
            verbose=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def predict(self, emissions):
        return self.hmm_model.predict(
            *self.format_predict(emissions), algorithm=self.algorithm
        )


class MethylSegmenter:
    """
    Class to handle segmentation of methylation data using HMMs.
    Recommend CTHMM for sparse data with variable probe spacing, and the
    sticky categorical smoother for dense discrete state-label smoothing.
    """

    def __init__(
        self,
        analyzer: MethylStateAnalyzer,
        hmm_model: MethylSegHMM,
        state_assignment_method: MethylStateAssignmentMethod = MethylStateAssignmentMethod.DEFINITION,
        hmm_observation_mode: HMMObservationMode = HMMObservationMode.DISCRETE_STATES,
        out_dir=".",
        random_state: int = 42,
    ):
        self.analyzer = analyzer
        self.hmm_model = hmm_model
        self.state_assignment_method = MethylStateAssignmentMethod(
            state_assignment_method
        )
        self.hmm_observation_mode = HMMObservationMode(hmm_observation_mode)
        self.out_dir = out_dir
        self.random_state = random_state
        self.segment_results = {}

    def _encode_states_for_hmm(self, states: np.ndarray) -> np.ndarray:
        numeric_states = MethylationStates.convert_to_numeric(states)
        unique_states = np.sort(np.unique(numeric_states))
        supported_observations = getattr(
            self.hmm_model,
            "n_emissions",
            getattr(self.hmm_model, "n_states", None),
        )
        if supported_observations is not None and len(unique_states) > int(
            supported_observations
        ):
            raise ValueError(
                "Observed state labels contain more distinct categories than the "
                "configured HMM can represent."
            )

        state_to_obs = {
            int(state_value): obs_idx
            for obs_idx, state_value in enumerate(unique_states.tolist())
        }
        return np.array(
            [state_to_obs[int(state_value)] for state_value in numeric_states],
            dtype=int,
        )

    def _get_state_cutoffs(self) -> Optional[Dict[str, object]]:
        return getattr(self.analyzer, "state_cutoffs", None)

    def _prepare_emissions(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        meth_data, _emission_matrix, emissions_df = (
            self.analyzer.assigner.prepare_sample_for_clustering(
                sample_info=sample_info,
                chrom=chrom,
            )
        )
        self.meth_data = meth_data.copy()
        self.emissions_df = emissions_df.copy()
        return self.meth_data, self.emissions_df

    def _derive_sequence_lengths(
        self,
        meth_data: pd.DataFrame,
        chrom: str | None = None,
    ) -> Optional[List[int]]:
        if chrom is not None or len(meth_data) == 0:
            return None

        chrom_values = meth_data["CpG_chrm"].astype(str).to_numpy()
        lengths = []
        current_chrom = chrom_values[0]
        current_length = 1
        for chrom_name in chrom_values[1:]:
            if chrom_name == current_chrom:
                current_length += 1
            else:
                lengths.append(current_length)
                current_chrom = chrom_name
                current_length = 1
        lengths.append(current_length)
        return lengths if len(lengths) > 1 else None

    def _set_hmm_sequence_lengths(
        self,
        lengths: Optional[List[int]],
    ) -> None:
        if not hasattr(self.hmm_model, "lengths"):
            return
        self.hmm_model.lengths = (
            None if lengths is None else [int(length) for length in lengths]
        )

    def _prepare_gaussian_feature_matrix(
        self,
        emission_df: pd.DataFrame,
    ) -> np.ndarray:
        feature_cols = self.analyzer.assigner._resolve_feature_cols(emission_df)
        X_scaled, _imputer, _scaler = (
            self.analyzer.assigner._preprocess_emission_features(
                emission_df=emission_df,
                feature_cols=feature_cols,
                fit=True,
            )
        )
        return X_scaled

    def _get_gaussian_init_labels(
        self,
        emission_df: pd.DataFrame,
        X_scaled: np.ndarray,
    ) -> np.ndarray:
        assigner = self.analyzer.assigner
        if hasattr(assigner, "model") and getattr(assigner, "model", None) is not None:
            try:
                _, _, raw_labels, _ = assigner.apply_kmeans_to_emissions(emission_df)
                return np.asarray(raw_labels, dtype=int)
            except Exception:
                pass

        temp_kmeans = KMeans(
            n_clusters=self.hmm_model.n_states,
            n_init=10,
            random_state=self.random_state,
        )
        return temp_kmeans.fit_predict(X_scaled)

    def _prepare_pca_feature_matrix(
        self,
        emission_df: pd.DataFrame,
    ) -> np.ndarray:
        assigner = self.analyzer.assigner
        feature_cols = assigner._resolve_feature_cols(emission_df)

        if hasattr(assigner, "model") and getattr(assigner, "model", None) is not None:
            model = assigner.model
            X_scaled = assigner._preprocess_emission_features(
                emission_df=emission_df,
                feature_cols=model.feature_cols,
                fit=False,
            )
            if model.pca is not None:
                return model.pca.transform(X_scaled)

        X_scaled = self._prepare_gaussian_feature_matrix(emission_df)
        if assigner.n_pca is None or assigner.n_pca <= 0:
            raise ValueError(
                "PCA-emission observation mode requires n_pca to be a positive integer."
            )

        n_components = min(assigner.n_pca, X_scaled.shape[0], X_scaled.shape[1])
        if n_components <= 0:
            raise ValueError(
                "Cannot fit PCA-emission features because the emission matrix is empty."
            )

        return PCA(
            n_components=n_components,
            random_state=self.random_state,
        ).fit_transform(X_scaled)

    def _get_pca_init_labels(
        self,
        emission_df: pd.DataFrame,
        X_pca: np.ndarray,
    ) -> np.ndarray:
        assigner = self.analyzer.assigner
        if (
            hasattr(assigner, "model")
            and getattr(assigner, "model", None) is not None
            and getattr(assigner.model, "pca", None) is not None
        ):
            try:
                _, _, raw_labels, _ = assigner.apply_kmeans_to_emissions(emission_df)
                return np.asarray(raw_labels, dtype=int)
            except Exception:
                pass

        temp_kmeans = KMeans(
            n_clusters=self.hmm_model.n_states,
            n_init=10,
            random_state=self.random_state,
        )
        return temp_kmeans.fit_predict(X_pca)

    def _segment_sample_discrete_states(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.assign_states(sample_info, chrom)
        states = self._encode_states_for_hmm(
            self.meth_data["state"].to_numpy(dtype=int)
        )
        sequence_lengths = self._derive_sequence_lengths(self.meth_data, chrom=chrom)
        self._set_hmm_sequence_lengths(sequence_lengths)

        self.hmm_model.create_model()
        self.hmm_model.fit(states, sample_info, chrom)
        hidden_states = self.hmm_model.predict(states)
        readable_states = self.analyzer.assigner.relabel_by_mean_emission(
            hidden_states,
            self.emissions_df,
            self._get_state_cutoffs(),
        )
        return hidden_states, readable_states

    def _segment_sample_gaussian_emissions(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._prepare_emissions(sample_info=sample_info, chrom=chrom)
        X_scaled = self._prepare_gaussian_feature_matrix(self.emissions_df)
        km_labels = self._get_gaussian_init_labels(self.emissions_df, X_scaled)

        init_readable_states = self.analyzer.assigner.relabel_by_mean_emission(
            km_labels,
            self.emissions_df,
            self._get_state_cutoffs(),
        )
        self.meth_data["state"] = MethylationStates.convert_to_numeric(
            init_readable_states
        )
        self.meth_data["state_readable"] = init_readable_states

        sequence_lengths = self._derive_sequence_lengths(self.meth_data, chrom=chrom)

        if not hasattr(self.hmm_model, "initialize_from_kmeans"):
            raise ValueError(
                "Gaussian-emission observation mode requires an HMM model that "
                "supports KMeans-based initialization."
            )

        self.hmm_model.create_model()
        self.hmm_model.initialize_from_kmeans(
            X_scaled=X_scaled,
            km_labels=km_labels,
            lengths=sequence_lengths,
        )
        self.hmm_model.fit(X_scaled, sample_info, chrom)
        hidden_states = self.hmm_model.predict(X_scaled)
        readable_states = self.analyzer.assigner.relabel_by_mean_emission(
            hidden_states,
            self.emissions_df,
            self._get_state_cutoffs(),
        )
        return hidden_states, readable_states

    def assign_states(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> np.ndarray:
        meth_data, emissions_df = self._prepare_emissions(
            sample_info=sample_info, chrom=chrom
        )
        if self.state_assignment_method.value == MethylStateAssignmentMethod.DEFINITION.value:
            states = self.analyzer.define_states_by_rules(
                sample_info=sample_info,
                chrom=chrom,
                sample_emissions=emissions_df,
            )
        elif self.state_assignment_method.value == MethylStateAssignmentMethod.KMEANS.value:
            _, _, _, states = self.analyzer.assigner.apply_kmeans_to_emissions(
                emissions_df
            )
        elif self.state_assignment_method.value == MethylStateAssignmentMethod.AUTO.value:
            raise NotImplementedError(
                "AUTO state assignment method not implemented yet."
            )
        else:
            raise ValueError(
                f"Unknown state assignment method: {self.state_assignment_method}"
            )
        meth_data = meth_data.copy()
        meth_data["state"] = MethylationStates.convert_to_numeric(states)
        meth_data["state_readable"] = states

        self.meth_data = meth_data
        self.emissions_df = emissions_df

        return meth_data, emissions_df
    def _segment_sample_pca_emissions(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._prepare_emissions(sample_info=sample_info, chrom=chrom)
        X_pca = self._prepare_pca_feature_matrix(self.emissions_df)
        km_labels = self._get_pca_init_labels(self.emissions_df, X_pca)

        init_readable_states = self.analyzer.assigner.relabel_by_mean_emission(
            km_labels,
            self.emissions_df,
            self._get_state_cutoffs(),
        )
        self.meth_data["state"] = MethylationStates.convert_to_numeric(
            init_readable_states
        )
        self.meth_data["state_readable"] = init_readable_states

        sequence_lengths = self._derive_sequence_lengths(self.meth_data, chrom=chrom)

        if not hasattr(self.hmm_model, "initialize_from_kmeans"):
            raise ValueError(
                "PCA-emission observation mode requires an HMM model that "
                "supports KMeans-based initialization."
            )

        self.hmm_model.create_model()
        self.hmm_model.initialize_from_kmeans(
            X_scaled=X_pca,
            km_labels=km_labels,
            lengths=sequence_lengths,
        )
        self.hmm_model.fit(X_pca, sample_info, chrom)
        hidden_states = self.hmm_model.predict(X_pca)
        readable_states = self.analyzer.assigner.relabel_by_mean_emission(
            hidden_states,
            self.emissions_df,
            self._get_state_cutoffs(),
        )
        return hidden_states, readable_states

    def segment_sample(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
        force_resegment: bool = False,
    ) -> Tuple[pd.DataFrame, object]:
        """
        Segment a sample and refresh probe-level results plus raw regions.

        Returns the segmented probe-level methylation table and fitted HMM
        object. Raw contiguous regions are stored on ``self.regions_df``.
        """
        chrom_segmented_on_sample = (
            sample_info.sample_id in self.segment_results
            and chrom in self.segment_results[sample_info.sample_id]
        )
        if not chrom_segmented_on_sample or force_resegment:
            if self.hmm_observation_mode.value == HMMObservationMode.DISCRETE_STATES.value:
                hidden_states, readable_states = self._segment_sample_discrete_states(
                    sample_info=sample_info,
                    chrom=chrom,
                )
            elif self.hmm_observation_mode.value == HMMObservationMode.GAUSSIAN_EMISSIONS.value:
                hidden_states, readable_states = (
                    self._segment_sample_gaussian_emissions(
                        sample_info=sample_info,
                        chrom=chrom,
                    )
                )
            elif self.hmm_observation_mode.value == HMMObservationMode.PCA_EMISSIONS.value:
                hidden_states, readable_states = (
                    self._segment_sample_pca_emissions(
                        sample_info=sample_info,
                        chrom=chrom,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown HMM observation mode: {self.hmm_observation_mode}"
                )

            self.segment_results.setdefault(sample_info.sample_id, {})
            self.segment_results[sample_info.sample_id][chrom] = {
                "meth_data": self.meth_data.copy(),
                "emissions_df": self.emissions_df.copy(),
                "hmm_state": hidden_states,
                "hmm_state_readable": readable_states,
            }
        else:
            self.meth_data = self.segment_results[sample_info.sample_id][chrom][
                "meth_data"
            ].copy()
            self.emissions_df = self.segment_results[sample_info.sample_id][chrom][
                "emissions_df"
            ].copy()
            hidden_states = self.segment_results[sample_info.sample_id][chrom][
                "hmm_state"
            ]
            readable_states = self.segment_results[sample_info.sample_id][chrom][
                "hmm_state_readable"
            ]

        cache_entry = self.segment_results[sample_info.sample_id][chrom]

        # Attach HMM states and refresh the raw contiguous regions.
        self.meth_data["hmm_state"] = hidden_states
        self.meth_data["hmm_state_readable"] = readable_states
        self.regions_df = self.create_regions(
            state_col="hmm_state_readable",
            region_min_probes=1,
        )
        cache_entry["meth_data"] = self.meth_data.copy()
        cache_entry["regions_df"] = self.regions_df.copy()

        # print(f"State relabeling completed in {time.time() - start:.2f} seconds.")

        return self.meth_data, self.hmm_model.hmm_model

    def create_regions(self, state_col="hmm_state_readable", region_min_probes=1):
        """
        Create regions (start, end) for contiguous segments of the same state.

        Parameters
        ----------
        meth_data : DataFrame
            Must contain 'CpG_chrm', 'CpG_beg', 'CpG_end', and state_col.
        state_col : str
            Column name for the state labels.
        region_min_probes : int
            Minimum number of probes required to form a region.
        Returns
        -------
        regions_df : DataFrame
            Columns: 'CpG_chrm', 'start', 'end', state_col
        """

        for col in ["CpG_chrm", "CpG_beg", "CpG_end", state_col]:
            if col not in self.meth_data.columns:
                raise ValueError(f"Column {col} not found in meth_data.")

        regions = []
        current_chrom = None
        current_state = None
        region_start = None
        region_end = None
        current_meth_sum = 0
        current_probe_count = 0

        for idx, row in self.meth_data.iterrows():
            chrom = row["CpG_chrm"]
            state = row[state_col]
            beg = row["CpG_beg"]
            end = row["CpG_end"]
            beta_val = float(row["beta"])

            if (chrom != current_chrom) or (state != current_state):
                # Save previous region
                if (
                    current_chrom is not None
                    and current_probe_count >= region_min_probes
                ):
                    regions.append(
                        {
                            "CpG_chrm": current_chrom,
                            "start": region_start,
                            "end": region_end,
                            "avg_beta": current_meth_sum / current_probe_count,
                            "probe_count": current_probe_count,
                            "state": current_state,
                        }
                    )
                # Start new region
                current_chrom = chrom
                current_state = state
                region_start = beg
                region_end = end
                current_meth_sum = beta_val
                current_probe_count = 1
            else:
                # Extend current region
                region_end = end
                current_meth_sum += beta_val
                current_probe_count += 1
        # Save last region
        if current_chrom is not None and current_probe_count >= region_min_probes:
            regions.append(
                {
                    "CpG_chrm": current_chrom,
                    "start": region_start,
                    "end": region_end,
                    "avg_beta": current_meth_sum / current_probe_count,
                    "probe_count": current_probe_count,
                    "state": current_state,
                }
            )
        regions_df = pd.DataFrame(
            regions,
            columns=["CpG_chrm", "start", "end", "avg_beta", "probe_count", "state"],
        )
        self.regions_df = regions_df
        return regions_df

    def regions_to_bed(self, bed_path: str, separate_beds_by_state: bool = False):
        """
        Save regions DataFrame to BED file.

        Parameters
        ----------
        regions_df : DataFrame
            Must contain 'CpG_chrm', 'start', 'end', and 'state'.
        bed_path : str
            Output path for the BED file.
        """
        if not bed_path.lower().endswith(".bed"):
            bed_path += ".bed"
        regions_df = self.regions_df.copy()
        regions_df["start"] = regions_df["start"].astype(int)
        regions_df["end"] = regions_df["end"].astype(int)
        if not separate_beds_by_state:
            bed_df = regions_df[["CpG_chrm", "start", "end", "state"]].copy()
            bed_df.to_csv(bed_path, sep="\t", header=False, index=False)
        else:
            for state in MethylationStates:
                state_df = regions_df[regions_df["state"] == state]
                bed_df = state_df[["CpG_chrm", "start", "end", "state"]].copy()
                state_bed_path = bed_path.replace(".bed", f"_{state.name}.bed")
                bed_df.to_csv(state_bed_path, sep="\t", header=False, index=False)

    # TODO: move to utils to share with MethylStateAnalyzer
    def plot_interactive_beta_by_label(
        self,
        sample_info: SampleInfo,
        sample_info_removed: pd.DataFrame | None = None,
        label_type: str = "hmm",
        use_train_data: bool = True,
        chrom: str | None = None,
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,
        color_pmr_only: bool = False,
        color_regions_df: pd.DataFrame | None = None,
    ):
        """
        Interactive scatter: genomic position vs beta, colored by label.
        """
        meth_data, _ = self.segment_sample(sample_info=sample_info, chrom=chrom)
        df_plot = meth_data.copy()
        if label_type == "hmm":
            label_col = f"{label_type}_state_readable"
        else:
            label_col = "state_readable"
        return _plot_interactive_beta_scatter(
            df_plot=df_plot,
            sample_info=sample_info,
            sample_info_removed=sample_info_removed,
            chrom=chrom,
            out_dir=self.out_dir,
            label_col=label_col,
            x_col=x_col,
            y_col=y_col,
            label_title=label_title,
            show_plot=show_plot,
            max_points=max_points,
            color_pmr_only=color_pmr_only,
            color_regions_df=color_regions_df,
        )


class MethylSegPathway:

    @classmethod
    def get_pretrained_model(cls, out_dir, hmm_type="ct"):
        pretrained_yaml = FILES / f"methyl_seg_config_{hmm_type}.yaml"

        model = cls.from_yaml(pretrained_yaml)

        model.out_dir = out_dir

        return model

    @staticmethod
    def prepare_sample_info(
        sample_name: str,
        sample_file: str | Path,
        resolution: str = "auto",
        min_coverage: int = 5,
        remove_low_coverage_like_cpgs: bool = False,
    ) -> tuple[SampleInfo, pd.DataFrame]:
        return MethylDataPrep(
            meth_file=sample_file,
            sample_id=sample_name,
            resolution=resolution,
            min_coverage=min_coverage,
            remove_low_coverage_like_cpgs=remove_low_coverage_like_cpgs,
        ).prepare()

    @staticmethod
    def subset_sample_info_by_chroms(
        sample_info: SampleInfo,
        chroms: Optional[List[str]] = None,
    ) -> SampleInfo:
        meth_data = sample_info.meth_data.copy()
        if meth_data.empty:
            raise ValueError("SampleInfo contains no methylation data.")

        chrom_series = meth_data["CpG_chrm"].astype(str)
        available_chroms = chrom_series.drop_duplicates().tolist()

        if chroms is None:
            resolved_chroms = available_chroms
        else:
            requested_chroms = []
            missing_chroms = []
            available_chroms_set = set(available_chroms)
            for chrom in chroms:
                chrom = str(chrom)
                if chrom in available_chroms_set:
                    if chrom not in requested_chroms:
                        requested_chroms.append(chrom)
                else:
                    missing_chroms.append(chrom)
            if missing_chroms:
                raise ValueError(
                    "Requested chromosomes are missing from the sample: "
                    f"{missing_chroms}"
                )
            resolved_chroms = requested_chroms

        if not resolved_chroms:
            raise ValueError("No chromosomes were selected for segmentation.")

        chrom_frames = [
            meth_data.loc[chrom_series == chrom].copy() for chrom in resolved_chroms
        ]
        filtered_meth_data = pd.concat(chrom_frames, ignore_index=True)
        return SampleInfo(
            sample_id=sample_info.sample_id,
            meth_data=filtered_meth_data,
        )

    def __init__(
        self,
        n_states: int = 4,
        int_low_cutoff: float = 0.2,
        int_high_cutoff: float = 0.7,
        high_cutoff: float = 0.8,
        window_specs: list[tuple[int, str]] = [(500_000, "500kb")],
        train_sample_info: SampleInfo | None = None,
        train_sample_file: str | None = None,
        train_sample_name: str | None = None,
        train_chroms: list[str] | None = None,
        max_cpg_per_chrom: int | None = 50_000,
        state_assignment_method: MethylStateAssignmentMethod = MethylStateAssignmentMethod.DEFINITION,
        out_dir: str = ".",
        random_state: int = 42,
        cluster_space: str = "pca",
        n_pca: int | None = 5,
        hmm_type: str = "ct",
        hmm_params: dict = {},
        min_region_length: int = 10_000,
        min_region_cpgs: int = 1,
        merge_gap_bp: int = 0,
        hmm_observation_mode: HMMObservationMode = HMMObservationMode.DISCRETE_STATES,
    ):
        self.window_specs = window_specs
        self.n_states = n_states
        self.int_low_cutoff = int_low_cutoff
        self.int_high_cutoff = int_high_cutoff
        self.high_cutoff = high_cutoff
        self.min_region_length = min_region_length
        if (
            train_sample_info is None
            and train_sample_file is not None
            and train_sample_name is not None
        ):
            self.train_sample_info, _ = self.prepare_sample_info(
                sample_name=train_sample_name,
                sample_file=train_sample_file,
                resolution="auto",
            )
        elif train_sample_info is not None:
            self.train_sample_info = train_sample_info
        else:
            raise ValueError(
                "Must provide either train_sample_info or train_sample_file and train_sample_name."
            )
        self.train_chroms = train_chroms
        self.max_cpg_per_chrom = max_cpg_per_chrom
        self.random_state = random_state
        self.out_dir = out_dir
        self.cluster_space = cluster_space
        self.n_pca = n_pca
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.hmm_type = hmm_type
        self.hmm_params = hmm_params
        self.state_assignment_method = MethylStateAssignmentMethod(
            state_assignment_method
        )
        self.min_region_cpgs = int(min_region_cpgs)
        self.merge_gap_bp = int(merge_gap_bp)
        self.hmm_observation_mode = HMMObservationMode(hmm_observation_mode)
        if (
            self.hmm_observation_mode
            in {
                HMMObservationMode.GAUSSIAN_EMISSIONS,
                HMMObservationMode.PCA_EMISSIONS,
            }
            and self.hmm_type != "gaussian"
        ):
            raise ValueError(
                "Gaussian-backed observation modes require "
                "hmm_type='gaussian'."
            )
        if (
            self.hmm_type == "gaussian"
            and self.hmm_observation_mode
            not in {
                HMMObservationMode.GAUSSIAN_EMISSIONS,
                HMMObservationMode.PCA_EMISSIONS,
            }
        ):
            raise ValueError(
                "hmm_type='gaussian' requires "
                "a Gaussian-backed observation mode."
            )
        self._init_hmm()

        self.assigner = MethylStateAssigner(
            window_specs=self.window_specs,
            n_states=n_states,
            int_low_cutoff=int_low_cutoff,
            int_high_cutoff=int_high_cutoff,
            high_cutoff=high_cutoff,
            out_dir=out_dir,
            random_state=self.random_state,
            cluster_space=cluster_space,
            n_pca=n_pca,
        )
        self.cluster_space = self.assigner.cluster_space
        self.n_pca = self.assigner.n_pca
        self.analyzer = MethylStateAnalyzer(
            assigner=self.assigner,
            out_dir=out_dir,
        )
        self.segmentor = MethylSegmenter(
            analyzer=self.analyzer,
            hmm_model=self.hmm_model,
            state_assignment_method=self.state_assignment_method,
            hmm_observation_mode=self.hmm_observation_mode,
            out_dir=out_dir,
            random_state=self.random_state,
        )

    def _init_hmm(self):
        if self.hmm_type == "multinomial":
            self.hmm_model = MultinomialSegHMM(
                n_states=self.n_states,
                random_state=self.random_state,
                **self.hmm_params,
            )
        elif self.hmm_type == "sticky":
            self.hmm_model = StickyCategoricalMethylSegHMM(
                n_states=self.n_states,
                random_state=self.random_state,
                **self.hmm_params,
            )
        elif self.hmm_type == "ct":
            self.hmm_model = CTMethylSegHMM(
                n_states=self.n_states,
                **self.hmm_params,
            )
        elif self.hmm_type == "gaussian":
            self.hmm_model = GaussianMethylSegHMM(
                n_states=self.n_states,
                random_state=self.random_state,
                **self.hmm_params,
            )
        else:
            raise ValueError(f"Unknown HMM type: { self.hmm_type}")

    def fit_pathway(
        self,
        force_optimize_rules: bool = False,
    ):
        model, train_meth, train_emissions, train_pca, train_labels = (
            self.assigner.train_kmeans_for_sample(
                sample_info=self.train_sample_info,
                train_chroms=self.train_chroms,
                windows_to_use=None,
                max_cpg_per_chrom=self.max_cpg_per_chrom,
            )
        )
        if (
            self.state_assignment_method.value != MethylStateAssignmentMethod.KMEANS.value
            or force_optimize_rules
        ):
            self.analyzer.optimize_rule_params_random()

    def generate_regions(
        self,
        sample_info: SampleInfo | None = None,
        chrom: str = "chr1",
        min_probes: int = 3,
        sample_name: str | None = None,
        sample_file: str | None = None,
        force_resegment: bool = False,
    ):
        if sample_info is None:
            if sample_file is not None and sample_name is not None:
                sample_info, _ = self.prepare_sample_info(
                    sample_name=sample_name,
                    sample_file=sample_file,
                    resolution="auto",
                )
            else:
                raise ValueError(
                    "Must provide either sample_info or sample_file and sample_name."
                )
        meth_data, hmm_model = self.segmentor.segment_sample(
            sample_info=sample_info, chrom=chrom, force_resegment=force_resegment
        )
        regions_df = self.segmentor.create_regions(
            state_col="hmm_state_readable", region_min_probes=min_probes
        )
        bed_path = f"{self.out_dir}/segments_{chrom}_{sample_info.sample_id}.bed"
        self.segmentor.regions_to_bed(bed_path, separate_beds_by_state=True)
        regions_df = regions_df.copy()
        regions_df["length"] = regions_df["end"] - regions_df["start"]
        regions_df = regions_df.loc[
            regions_df["length"] > self.min_region_length
        ].copy()
        return regions_df

    @staticmethod
    def _empty_regions_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["CpG_chrm", "start", "end", "avg_beta", "probe_count", "state"]
        )

    @staticmethod
    def _empty_clean_regions_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "CpG_chrm",
                "start",
                "end",
                "avg_beta",
                "probe_count",
                "state",
                "length",
            ]
        )

    @staticmethod
    def _coerce_region_state(state: MethylationStates | str) -> MethylationStates:
        if isinstance(state, MethylationStates):
            return state
        if isinstance(state, str):
            state = state.strip()
            try:
                return MethylationStates[state]
            except KeyError:
                pass
            try:
                return MethylationStates(int(state))
            except (TypeError, ValueError):
                pass
        raise ValueError(f"Unknown methylation state: {state}")

    def get_clean_regions(
        self,
        regions_df: pd.DataFrame | None = None,
        state: MethylationStates | str = MethylationStates.PMR,
        merge_gap_bp: int | None = None,
        min_region_length: int | None = None,
        min_cpgs: int | None = None,
    ) -> pd.DataFrame:
        target_state = self._coerce_region_state(state)
        merge_gap_bp = self.merge_gap_bp if merge_gap_bp is None else int(merge_gap_bp)
        min_region_length = (
            self.min_region_length
            if min_region_length is None
            else int(min_region_length)
        )
        min_cpgs = self.min_region_cpgs if min_cpgs is None else int(min_cpgs)

        if regions_df is None:
            regions_df = getattr(self.segmentor, "regions_df", None)
        if regions_df is None or regions_df.empty:
            return self._empty_clean_regions_df()

        required_columns = [
            "CpG_chrm",
            "start",
            "end",
            "avg_beta",
            "probe_count",
            "state",
        ]
        missing_columns = [
            col for col in required_columns if col not in regions_df.columns
        ]
        if missing_columns:
            raise ValueError(
                "regions_df is missing required columns for cleaning: "
                f"{missing_columns}"
            )

        clean_df = regions_df.copy()
        clean_df["CpG_chrm"] = clean_df["CpG_chrm"].astype(str)
        clean_df["start"] = pd.to_numeric(clean_df["start"], errors="raise").astype(int)
        clean_df["end"] = pd.to_numeric(clean_df["end"], errors="raise").astype(int)
        clean_df["avg_beta"] = pd.to_numeric(clean_df["avg_beta"], errors="raise")
        clean_df["probe_count"] = pd.to_numeric(
            clean_df["probe_count"], errors="raise"
        ).astype(int)
        clean_df["state"] = clean_df["state"].apply(self._coerce_region_state)
        clean_df = clean_df.loc[clean_df["state"] == target_state].copy()

        if clean_df.empty:
            return self._empty_clean_regions_df()

        clean_df = clean_df.sort_values(["CpG_chrm", "start", "end"]).reset_index(
            drop=True
        )

        merged_regions = []
        current_region = None

        for row in clean_df.itertuples(index=False):
            if current_region is None:
                current_region = {
                    "CpG_chrm": row.CpG_chrm,
                    "start": int(row.start),
                    "end": int(row.end),
                    "beta_weighted_sum": float(row.avg_beta) * int(row.probe_count),
                    "probe_count": int(row.probe_count),
                    "state": target_state,
                }
                continue

            gap_bp = int(row.start) - int(current_region["end"])
            can_merge = (
                merge_gap_bp > 0
                and row.CpG_chrm == current_region["CpG_chrm"]
                and gap_bp <= merge_gap_bp
            )

            if can_merge:
                current_region["end"] = max(int(current_region["end"]), int(row.end))
                current_region["beta_weighted_sum"] += float(row.avg_beta) * int(
                    row.probe_count
                )
                current_region["probe_count"] += int(row.probe_count)
            else:
                merged_regions.append(current_region)
                current_region = {
                    "CpG_chrm": row.CpG_chrm,
                    "start": int(row.start),
                    "end": int(row.end),
                    "beta_weighted_sum": float(row.avg_beta) * int(row.probe_count),
                    "probe_count": int(row.probe_count),
                    "state": target_state,
                }

        if current_region is not None:
            merged_regions.append(current_region)

        merged_df = pd.DataFrame(merged_regions)
        merged_df["avg_beta"] = (
            merged_df["beta_weighted_sum"] / merged_df["probe_count"]
        )
        merged_df = merged_df.drop(columns=["beta_weighted_sum"])
        merged_df["length"] = merged_df["end"] - merged_df["start"]
        merged_df = merged_df.loc[
            (merged_df["length"] >= int(min_region_length))
            & (merged_df["probe_count"] >= int(min_cpgs))
        ].copy()

        if merged_df.empty:
            return self._empty_clean_regions_df()

        return merged_df.loc[
            :,
            ["CpG_chrm", "start", "end", "avg_beta", "probe_count", "state", "length"],
        ].reset_index(drop=True)

    def _write_regions_by_chrom_and_state(
        self,
        regions_df: pd.DataFrame,
        sample_id: str,
        chroms: List[str],
    ) -> None:
        output_dir = Path(self.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if regions_df is None or regions_df.empty:
            regions_df = self._empty_regions_df()
        else:
            regions_df = regions_df.copy()
            regions_df["CpG_chrm"] = regions_df["CpG_chrm"].astype(str)
            regions_df["start"] = pd.to_numeric(
                regions_df["start"], errors="raise"
            ).astype(int)
            regions_df["end"] = pd.to_numeric(regions_df["end"], errors="raise").astype(
                int
            )

        for chrom in chroms:
            chrom_regions = regions_df.loc[
                regions_df["CpG_chrm"].astype(str) == str(chrom)
            ].copy()
            state_names = (
                chrom_regions["state"].astype(str) if not chrom_regions.empty else None
            )

            for state in MethylationStates:
                if chrom_regions.empty:
                    state_df = chrom_regions.copy()
                else:
                    state_mask = (chrom_regions["state"] == state) | (
                        state_names == state.name
                    )
                    state_df = chrom_regions.loc[state_mask].copy()

                bed_path = output_dir / f"segments_{chrom}_{sample_id}_{state.name}.bed"
                bed_df = state_df[["CpG_chrm", "start", "end", "state"]].copy()
                bed_df.to_csv(bed_path, sep="\t", header=False, index=False)

    def run_on_all_chroms(
        self,
        sample_info: SampleInfo,
        chroms: Optional[List[str]] = None,
        min_probes: int = 3,
        force_resegment: bool = False,
    ) -> pd.DataFrame:
        filtered_sample_info = self.subset_sample_info_by_chroms(
            sample_info=sample_info,
            chroms=chroms,
        )
        resolved_chroms = (
            filtered_sample_info.meth_data["CpG_chrm"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        joint_hmm_types = {"sticky", "multinomial", "gaussian"}
        if self.hmm_type in joint_hmm_types:
            self.segmentor.segment_sample(
                sample_info=filtered_sample_info,
                chrom=None,
                force_resegment=force_resegment,
            )
            regions_df = self.segmentor.create_regions(
                state_col="hmm_state_readable",
                region_min_probes=min_probes,
            )
            self._write_regions_by_chrom_and_state(
                regions_df=regions_df,
                sample_id=filtered_sample_info.sample_id,
                chroms=resolved_chroms,
            )
            return regions_df

        region_frames = []
        for chrom in resolved_chroms:
            region_frames.append(
                self.generate_regions(
                    sample_info=filtered_sample_info,
                    chrom=chrom,
                    min_probes=min_probes,
                    force_resegment=force_resegment,
                )
            )

        if region_frames:
            combined_regions_df = pd.concat(region_frames, ignore_index=True)
        else:
            combined_regions_df = self._empty_regions_df()
        self.segmentor.regions_df = combined_regions_df.copy()
        return combined_regions_df

    # TODO: fix this, it is not saving train_sample_info or train_sample_file and train_sample_name correctly
    def to_yaml(self, yaml_path: str, include_learned: bool = True):
        """
        Serialize pathway configuration and optionally learned artifacts.
        """

        base_dir = Path(self.out_dir or ".")
        base_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "pathway": {
                "n_states": self.n_states,
                "int_low_cutoff": self.int_low_cutoff,
                "int_high_cutoff": self.int_high_cutoff,
                "high_cutoff": self.high_cutoff,
                "window_specs": self.window_specs,
                "train_chroms": self.train_chroms,
                "max_cpg_per_chrom": self.max_cpg_per_chrom,
                "out_dir": str(base_dir),
                "random_state": self.random_state,
                "cluster_space": self.assigner.cluster_space,
                "n_pca": self.assigner.n_pca,
                "hmm_type": self.hmm_type,
                "min_region_length": self.min_region_length,
                "min_region_cpgs": self.min_region_cpgs,
                "merge_gap_bp": self.merge_gap_bp,
                "state_assignment_method": self.state_assignment_method.value,
                "hmm_observation_mode": self.hmm_observation_mode.value,
            }
        }

        # --- Serialize hmm_params safely ---
        safe_hmm_params = {}
        hmm_param_dir = base_dir / "hmm_params"
        hmm_param_dir.mkdir(exist_ok=True)

        for k, v in (self.hmm_params or {}).items():
            if isinstance(v, np.ndarray):
                path = hmm_param_dir / f"{k}.npy"
                np.save(path, v)
                safe_hmm_params[k] = {"__npy_path__": str(path)}
            elif isinstance(v, (int, float, str, bool)) or v is None:
                safe_hmm_params[k] = v
            elif isinstance(v, (list, tuple)):
                safe_hmm_params[k] = list(v)
            else:
                safe_hmm_params[k] = str(v)

        cfg["hmm_params"] = safe_hmm_params

        # --- Save SampleInfo ---
        if self.train_sample_info is not None:
            sample_path = base_dir / "train_sample_meth.feather"
            self.train_sample_info.meth_data.reset_index(drop=True).to_feather(
                sample_path
            )

            cfg["train_sample_info"] = {
                "sample_id": self.train_sample_info.sample_id,
                "meth_data_path": str(sample_path),
            }

        # --- Save learned artifacts ---
        if include_learned and hasattr(self.assigner, "model"):

            model_dir = base_dir / "models"
            model_dir.mkdir(exist_ok=True)

            model_cfg = {}

            if self.assigner.model.kmeans is not None:
                kmeans_path = model_dir / "kmeans.joblib"
                joblib.dump(self.assigner.model.kmeans, kmeans_path)
                model_cfg["kmeans"] = str(kmeans_path)

            if self.assigner.model.scaler is not None:
                scaler_path = model_dir / "scaler.joblib"
                joblib.dump(self.assigner.model.scaler, scaler_path)
                model_cfg["scaler"] = str(scaler_path)

            if self.assigner.model.imputer is not None:
                imputer_path = model_dir / "imputer.joblib"
                joblib.dump(self.assigner.model.imputer, imputer_path)
                model_cfg["imputer"] = str(imputer_path)

            if self.assigner.model.pca is not None:
                pca_path = model_dir / "pca.joblib"
                joblib.dump(self.assigner.model.pca, pca_path)
                model_cfg["pca"] = str(pca_path)

            model_cfg["feature_cols"] = self.assigner.model.feature_cols
            model_cfg["n_states"] = self.assigner.model.n_states
            model_cfg["cluster_space"] = self.assigner.model.cluster_space
            model_cfg["n_pca"] = self.assigner.model.n_pca

            cfg["trained_model"] = model_cfg

        # --- Save rule cutoffs ---
        if hasattr(self.analyzer, "state_cutoffs"):
            cfg["state_cutoffs"] = self.analyzer.state_cutoffs

        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: str, load_learned: bool = True):
        """
        Reconstruct MethylSegPathway from YAML file.
        """

        with open(yaml_path, "r") as fh:
            cfg = yaml.safe_load(fh)

        pathway_cfg = cfg.get("pathway", {})

        n_states = pathway_cfg.get("n_states", 4)
        int_low_cutoff = pathway_cfg.get("int_low_cutoff", 0.2)
        int_high_cutoff = pathway_cfg.get("int_high_cutoff", 0.7)
        high_cutoff = pathway_cfg.get("high_cutoff", 0.8)
        window_specs = pathway_cfg.get("window_specs", [(500_000, "500kb")])
        train_chroms = pathway_cfg.get("train_chroms", None)
        if train_chroms is None and "train_chrom" in pathway_cfg:
            legacy_train_chrom = pathway_cfg.get("train_chrom")
            train_chroms = (
                [legacy_train_chrom] if legacy_train_chrom is not None else None
            )
        max_cpg_per_chrom = pathway_cfg.get("max_cpg_per_chrom", 50_000)
        out_dir = pathway_cfg.get("out_dir", ".")
        random_state = pathway_cfg.get("random_state", 42)
        cluster_space = pathway_cfg.get("cluster_space", "pca")
        n_pca = pathway_cfg.get("n_pca", 5)
        hmm_type = pathway_cfg.get("hmm_type", "ct")
        min_region_length = pathway_cfg.get("min_region_length", 10_000)
        min_region_cpgs = pathway_cfg.get("min_region_cpgs", 1)
        merge_gap_bp = pathway_cfg.get("merge_gap_bp", 0)
        state_assignment_method = pathway_cfg.get(
            "state_assignment_method",
            MethylStateAssignmentMethod.DEFINITION.value,
        )
        hmm_observation_mode = pathway_cfg.get(
            "hmm_observation_mode",
            HMMObservationMode.DISCRETE_STATES.value,
        )

        # --- Load hmm_params ---
        hmm_params_cfg = cfg.get("hmm_params", {})
        loaded_hmm_params = {}

        for k, v in hmm_params_cfg.items():
            if isinstance(v, dict) and "__npy_path__" in v:
                loaded_hmm_params[k] = np.load(v["__npy_path__"], allow_pickle=False)
            else:
                loaded_hmm_params[k] = v

        # --- Load SampleInfo ---
        train_sample_info = None
        if "train_sample_info" in cfg:
            sample_cfg = cfg["train_sample_info"]
            sample_id = sample_cfg["sample_id"]
            meth_path = sample_cfg["meth_data_path"]
            meth_df = pd.read_feather(meth_path)
            train_sample_info = SampleInfo(sample_id=sample_id, meth_data=meth_df)

        # --- Create instance ---
        inst = cls(
            n_states=n_states,
            int_low_cutoff=int_low_cutoff,
            int_high_cutoff=int_high_cutoff,
            high_cutoff=high_cutoff,
            window_specs=window_specs,
            train_sample_info=train_sample_info,
            train_chroms=train_chroms,
            max_cpg_per_chrom=max_cpg_per_chrom,
            out_dir=out_dir,
            random_state=random_state,
            cluster_space=cluster_space,
            n_pca=n_pca,
            hmm_type=hmm_type,
            hmm_params=loaded_hmm_params,
            min_region_length=min_region_length,
            min_region_cpgs=min_region_cpgs,
            merge_gap_bp=merge_gap_bp,
            state_assignment_method=state_assignment_method,
            hmm_observation_mode=hmm_observation_mode,
        )

        # --- Restore trained model ---
        if load_learned and "trained_model" in cfg:
            model_cfg = cfg["trained_model"]

            kmeans = joblib.load(model_cfg["kmeans"]) if "kmeans" in model_cfg else None
            scaler = joblib.load(model_cfg["scaler"]) if "scaler" in model_cfg else None
            imputer = (
                joblib.load(model_cfg["imputer"]) if "imputer" in model_cfg else None
            )
            pca = joblib.load(model_cfg["pca"]) if "pca" in model_cfg else None
            model_cluster_space = model_cfg.get("cluster_space")
            if model_cluster_space is None:
                model_cluster_space = "pca" if pca is not None else "raw"
            model_n_pca = model_cfg.get("n_pca")
            if model_n_pca is None:
                model_n_pca = n_pca if model_cluster_space == "pca" else None

            inst.assigner.model = KMeansMethylationModel(
                kmeans=kmeans,
                scaler=scaler,
                imputer=imputer,
                pca=pca,
                feature_cols=model_cfg.get("feature_cols", []),
                n_states=model_cfg.get("n_states", n_states),
                cluster_space=model_cluster_space,
                n_pca=model_n_pca,
            )
            inst.assigner.cluster_space = model_cluster_space
            inst.assigner.n_pca = model_n_pca
            inst.cluster_space = model_cluster_space
            inst.n_pca = model_n_pca

        # --- Restore rule cutoffs ---
        if "state_cutoffs" in cfg:
            inst.analyzer.state_cutoffs = cfg["state_cutoffs"]

        inst._loaded_config = cfg
        return inst
