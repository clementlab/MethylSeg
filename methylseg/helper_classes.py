import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CODE_DIR = Path(__file__).resolve().parent.parent
FILES = CODE_DIR / "data" / "reference_files"
CANONICAL_AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))


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

    #TODO: Add 27k resolution support
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
            compression="gzip" if self.meth_file.suffix in {".gz", ".gzip"} else None,
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
            compression="gzip" if self.meth_file.suffix in {".gz", ".gzip"} else "infer",
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
    PMD = 1
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
