"""Shared enums, data containers, and input-preparation helpers for methylseg."""

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
DATA_DIR = CODE_DIR / "data"
FILES = DATA_DIR / "reference_files"
CANONICAL_AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))


class MethylEnum(Enum):
    """Base class for methylseg enums with case-insensitive string parsing."""

    @classmethod
    def from_string(cls, s: str):
        """
        Parse a string or scalar value into an enum member.

        Parameters
        ----------
        s
            Candidate enum name or value. Matching is case-insensitive for
            string-valued names and members.

        Returns
        -------
        MethylEnum
            Matching enum member from ``cls``.

        Raises
        ------
        ValueError
            If ``s`` does not match any member name or value.
        """
        s = s.strip().lower()
        for member in cls:
            if member.name.lower() == s:
                return member
            if isinstance(member.value, str) and member.value.lower() == s:
                return member
            if member.value == s:
                return member
        raise ValueError(
            f"Invalid {cls.__name__} value: {s}. "
            f"Valid options are: {[m.name for m in cls]}"
        )

    def __eq__(self, other):
        if isinstance(other, str):
            try:
                other = self.__class__.from_string(other)
            except ValueError:
                return NotImplemented
        return super().__eq__(other)

    __hash__ = Enum.__hash__

    def __str__(self):
        return self.name


@dataclass
class KMeansMethylationModel:
    """Container for the trained clustering model and its preprocessing steps."""

    kmeans: KMeans
    scaler: StandardScaler
    imputer: Optional[SimpleImputer]
    pca: Optional[PCA]
    feature_cols: List[str]
    n_states: int
    cluster_space: str = "pca"
    n_pca: Optional[int] = 5


class MethylStateAssignmentMethod(MethylEnum):
    """Strategies for mapping emissions to biological methylation states."""

    DEFINITION = "definition"
    KMEANS = "kmeans"
    AUTO = "auto"


class HMMType(MethylEnum):
    """Supported HMM model types for segmentation."""

    CT = "continuous-time"
    STICKY = "sticky"
    GAUSSIAN = "gaussian"
    MULTINOMIAL = "multinomial"


class HMMObservationMode(MethylEnum):
    """Observation representations supported by the downstream HMM segmentor."""

    DISCRETE_STATES = "discrete_states"
    GAUSSIAN_EMISSIONS = "gaussian_emissions"
    PCA_EMISSIONS = "pca_emissions"


@dataclass
class SampleInfo:
    """
    Simple container for sample metadata and prepared methylation rows.

    Parameters
    ----------
    sample_id
        Unique sample identifier such as a TCGA barcode.
    meth_data
        DataFrame with canonical methylation columns ``CpG_chrm``, ``CpG_beg``,
        ``CpG_end``, and ``beta``.
    """

    sample_id: str
    meth_data: pd.DataFrame
    resolution: Optional[str] = None

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
    """Normalize methylation input tables into the canonical ``SampleInfo`` schema."""

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
        min_coverage=10,
        remove_low_coverage_like_cpgs=False,
        chunk_size=1_000_000,
        retain_removed_rows=True,
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
            - "auto": Automatically detect format
            - "wgbs": Whole-genome bisulfite sequencing format
            - "450k": Illumina 450k array format
            - "27k": Illumina 27k array format
            - "850k": Illumina 850k array format
        min_coverage : int, default=5
            Minimum coverage threshold for WGBS data.
        remove_low_coverage_like_cpgs : bool, default=False
            If True, remove CpGs with beta values commonly produced by very
            low coverage counts, such as 0.0, 0.25, 0.33, 0.5, 0.66/0.67,
            0.75, and 1.0.
        chunk_size: int, default=1_000_000
            Number of rows to read at a time when processing large files.
        retain_removed_rows: bool, default=True
            If True, retain removed rows in a separate DataFrame for downstream
            analysis. If False, removed rows will be discarded.
        """

        self.meth_file = Path(meth_file)
        self.sample_id = sample_id
        self.resolution = resolution
        self.min_coverage = min_coverage
        self.remove_low_coverage_like_cpgs = remove_low_coverage_like_cpgs
        self.chunk_size = chunk_size
        self.retain_removed_rows = retain_removed_rows

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

    def _is_microarray_format(self) -> bool:
        if self.resolution in {"450k", "27k", "850k"}:
            return True
        if self.resolution == "wgbs":
            return False
        else:
            raise ValueError(
                f"Unsupported resolution for format inference: {self.resolution}"
            )

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
                elif self._is_microarray_format():
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

    def _check_concat_memory(
        self,
        retained_bytes: int,
        processed_rows: int,
        chunk_number: int,
    ) -> None:
        """Raise before concatenation is likely to exhaust available memory."""
        try:
            import psutil
        except ImportError:
            return

        available = psutil.virtual_memory().available

        # Concatenation can temporarily require another copy of retained data.
        required_headroom = retained_bytes + 512 * 1024**2

        if available < required_headroom:
            raise MemoryError(
                "Insufficient memory to finish preparing this methylation file. "
                f"Stopped after chunk {chunk_number:,} and "
                f"{processed_rows:,} input rows. Retained data currently uses "
                f"approximately {retained_bytes / 1024**3:.2f} GiB, with "
                f"{available / 1024**3:.2f} GiB available. Consider reducing "
                "chunk_size or setting retain_removed_rows=False."
            )

    def _load_wgbs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and filter WGBS methylation data in chunks.

        The expected input columns are:

        1. CpG chromosome
        2. CpG start
        3. CpG end
        4. Methylated read count
        5. Total coverage

        Files containing only four columns are treated as precomputed beta-value
        files and passed to ``_load_450k()``.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            The filtered canonical methylation table and the rows removed during
            coverage or beta-value filtering.
        """
        compression = "gzip" if self.meth_file.suffix in {".gz", ".gzip"} else "infer"

        try:
            preview = pd.read_csv(
                self.meth_file,
                sep="\t",
                header=None,
                nrows=1,
                compression=compression,
            )
        except pd.errors.EmptyDataError:
            filtered_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
            removed_df = pd.DataFrame(
                columns=[
                    "CpG_chrm",
                    "CpG_beg",
                    "CpG_end",
                    "meth",
                    "coverage",
                    "beta",
                    self.INPUT_ROW_INDEX_COL,
                ]
            )
            return filtered_df, self._format_removed_dataframe(removed_df)

        column_count = preview.shape[1]

        if column_count == 4:
            # Four-column inputs already contain beta values.
            return self._load_microarray()

        if column_count < 5:
            raise ValueError(
                "Expected at least 5 columns for WGBS input, "
                f"but found {column_count}: {self.meth_file}"
            )

        has_header = self._looks_like_header_row(preview.iloc[0].tolist())

        columns = [
            "CpG_chrm",
            "CpG_beg",
            "CpG_end",
            "meth",
            "coverage",
        ]

        reader = pd.read_csv(
            self.meth_file,
            sep="\t",
            header=None,
            names=columns,
            usecols=range(5),
            skiprows=1 if has_header else None,
            compression=compression,
            chunksize=self.chunk_size,
            dtype={
                "CpG_chrm": "string",
                "CpG_beg": np.int64,
                "CpG_end": np.int64,
                "meth": np.float64,
                "coverage": np.float64,
            },
        )

        filtered_chunks: list[pd.DataFrame] = []
        removed_chunks: list[pd.DataFrame] = []

        input_row_offset = 0
        retained_bytes = 0

        for chunk_number, chunk in enumerate(reader, start=1):
            chunk_length = len(chunk)

            # Preserve each row's original position in the input file.
            chunk[self.INPUT_ROW_INDEX_COL] = np.arange(
                input_row_offset,
                input_row_offset + chunk_length,
                dtype=np.int64,
            )
            input_row_offset += chunk_length

            # Calculate beta values from methylated and total read counts.
            # Coverage-zero rows will subsequently be removed by the coverage mask.
            with np.errstate(divide="ignore", invalid="ignore"):
                chunk["beta"] = chunk["meth"] / chunk["coverage"]

            coverage_mask = chunk["coverage"] >= self.min_coverage

            if self.retain_removed_rows and (~coverage_mask).any():
                coverage_removed = chunk.loc[~coverage_mask].copy()
                removed_chunks.append(coverage_removed)

                retained_bytes += coverage_removed.memory_usage(
                    index=True,
                    deep=True,
                ).sum()

            retained_chunk = chunk.loc[coverage_mask].copy()

            if self.remove_low_coverage_like_cpgs:
                low_coverage_like_mask = retained_chunk["beta"].isin(
                    self.LOW_COVERAGE_LIKE_BETA_VALUES
                )

                if self.retain_removed_rows and low_coverage_like_mask.any():
                    beta_removed = retained_chunk.loc[low_coverage_like_mask].copy()

                    removed_chunks.append(beta_removed)

                    retained_bytes += beta_removed.memory_usage(
                        index=True,
                        deep=True,
                    ).sum()

                retained_chunk = retained_chunk.loc[~low_coverage_like_mask]

            filtered_chunk = retained_chunk.loc[
                :,
                self.REQUIRED_COLUMNS,
            ].copy()

            filtered_chunks.append(filtered_chunk)

            retained_bytes += filtered_chunk.memory_usage(
                index=True,
                deep=True,
            ).sum()

            self._check_concat_memory(
                retained_bytes=retained_bytes,
                processed_rows=input_row_offset,
                chunk_number=chunk_number,
            )

        if filtered_chunks:
            filtered_df = pd.concat(
                filtered_chunks,
                axis=0,
                ignore_index=True,
                copy=False,
            )
        else:
            filtered_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        removed_columns = [
            "CpG_chrm",
            "CpG_beg",
            "CpG_end",
            "meth",
            "coverage",
            "beta",
            self.INPUT_ROW_INDEX_COL,
        ]

        if self.retain_removed_rows and removed_chunks:
            removed_df = pd.concat(
                removed_chunks,
                axis=0,
                ignore_index=True,
                copy=False,
            )
        else:
            removed_df = pd.DataFrame(columns=removed_columns)

        return filtered_df, self._format_removed_dataframe(removed_df)

    def _load_microarray(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        compression = "gzip" if self.meth_file.suffix in {".gz", ".gzip"} else "infer"

        try:
            preview = pd.read_csv(
                self.meth_file,
                sep="\t",
                header=None,
                nrows=1,
                compression=compression,
            )
        except pd.errors.EmptyDataError:
            filtered_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
            removed_df = pd.DataFrame(
                columns=self.REQUIRED_COLUMNS + [self.INPUT_ROW_INDEX_COL]
            )
            return filtered_df, self._format_removed_dataframe(removed_df)

        column_count = preview.shape[1]

        if column_count not in {4, 5}:
            raise ValueError(
                "Expected 4 or 5 columns for microarray input, "
                f"but found {column_count}: {self.meth_file}"
            )

        has_header = self._looks_like_header_row(preview.iloc[0].tolist())

        columns = ["CpG_chrm", "CpG_beg", "CpG_end", "beta"]
        if column_count == 5:
            columns.append("probe")

        dtypes = {
            "CpG_chrm": "string",
            "CpG_beg": np.int64,
            "CpG_end": np.int64,
            "beta": np.float64,
        }
        if column_count == 5:
            dtypes["probe"] = "string"

        reader = pd.read_csv(
            self.meth_file,
            sep="\t",
            header=None,
            names=columns,
            usecols=range(column_count),
            skiprows=1 if has_header else None,
            compression=compression,
            chunksize=self.chunk_size,
            dtype=dtypes,
        )

        filtered_chunks = []
        removed_chunks = []
        input_row_offset = 0
        retained_bytes = 0

        for chunk_number, chunk in enumerate(reader, start=1):
            chunk_length = len(chunk)

            chunk[self.INPUT_ROW_INDEX_COL] = np.arange(
                input_row_offset,
                input_row_offset + chunk_length,
                dtype=np.int64,
            )
            input_row_offset += chunk_length

            if self.remove_low_coverage_like_cpgs:
                removal_mask = chunk["beta"].isin(self.LOW_COVERAGE_LIKE_BETA_VALUES)

                if self.retain_removed_rows and removal_mask.any():
                    removed_chunk = chunk.loc[removal_mask].copy()
                    removed_chunks.append(removed_chunk)
                    retained_bytes += removed_chunk.memory_usage(
                        index=True,
                        deep=True,
                    ).sum()

                chunk = chunk.loc[~removal_mask]

            filtered_chunk = chunk.loc[:, self.REQUIRED_COLUMNS].copy()
            filtered_chunks.append(filtered_chunk)

            retained_bytes += filtered_chunk.memory_usage(
                index=True,
                deep=True,
            ).sum()

            self._check_concat_memory(
                retained_bytes=retained_bytes,
                processed_rows=input_row_offset,
                chunk_number=chunk_number,
            )

        if filtered_chunks:
            filtered_df = pd.concat(
                filtered_chunks,
                axis=0,
                ignore_index=True,
                copy=False,
            )
        else:
            filtered_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        if self.retain_removed_rows and removed_chunks:
            removed_df = pd.concat(
                removed_chunks,
                axis=0,
                ignore_index=True,
                copy=False,
            )
        else:
            removed_df = pd.DataFrame(columns=columns + [self.INPUT_ROW_INDEX_COL])

        return filtered_df, self._format_removed_dataframe(removed_df)

    def _load_auto(self) -> tuple[pd.DataFrame, pd.DataFrame]:

        try:
            return self._load_microarray()
        except Exception:
            pass

        return self._load_wgbs()

    def prepare_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and normalize the configured methylation file.

        Returns
        -------
        tuple of pandas.DataFrame
            Two data frames containing the filtered canonical methylation table
            and the rows removed during preparation. The filtered table contains
            ``CpG_chrm``, ``CpG_beg``, ``CpG_end``, and ``beta`` columns.

        Raises
        ------
        ValueError
            If the requested ``resolution`` is unsupported or the input cannot
            be normalized into the canonical schema.
        """
        if self.resolution == "wgbs":
            filtered_df, removed_df = self._load_wgbs()
        elif self._is_microarray_format():
            filtered_df, removed_df = self._load_microarray()
        elif self.resolution == "auto":
            filtered_df, removed_df = self._load_auto()
        else:
            raise ValueError(f"Unsupported methylation resolution: {self.resolution}")
        return filtered_df, removed_df

    def prepare(self) -> tuple[SampleInfo, pd.DataFrame]:
        """
        Prepare the methylation file and wrap it in ``SampleInfo``.

        Returns
        -------
        tuple
            A ``(sample_info, removed_df)`` pair where ``sample_info`` contains
            the normalized methylation rows and ``removed_df`` contains excluded
            input rows indexed by original row position when available.
        """
        filtered_df, removed_df = self.prepare_dataframe()
        return (
            SampleInfo(
                sample_id=self.sample_id,
                meth_data=filtered_df,
                resolution=self.resolution,
            ),
            removed_df,
        )

    def write_prepared_tsv(self, out_file, sep="\t") -> Path:
        """
        Write the prepared methylation table to disk.

        Parameters
        ----------
        out_file
            Destination path for the normalized TSV-like output.
        sep
            Delimiter used when writing the prepared table.

        Returns
        -------
        pathlib.Path
            Resolved output path that was written.
        """
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        filtered_df, _ = self.prepare_dataframe()
        filtered_df.to_csv(out_file, sep=sep, index=False)
        return out_file


class MethylationStates(MethylEnum):
    """Canonical biological methylation states used throughout the package."""

    LOW = 0
    PMD = 1
    INTERMEDIATE = 2
    HIGH = 3

    def __lt__(self, other):
        if isinstance(other, MethylationStates):
            return self.value < other.value
        return NotImplemented

    @staticmethod
    def convert_to_numeric(arr):
        """
        Convert methylation-state labels into integer codes.

        Parameters
        ----------
        arr
            Array-like of ``MethylationStates`` values or already numeric state
            labels.

        Returns
        -------
        numpy.ndarray
            Integer array suitable for model fitting, serialization, or plotting.
        """
        arr = np.asarray(arr)
        if isinstance(arr[0], Enum):
            return np.array([a.value for a in arr], dtype=int)
        return arr.astype(int)
