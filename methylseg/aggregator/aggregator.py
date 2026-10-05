"""Peak-based aggregation of completed per-sample MethylSeg region tracks."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import math
from pathlib import Path

import pandas as pd

from ..helper_classes import MethylationStates


class AggregationMode(str, Enum):
    """Filters that can be applied to a peak signal track."""

    COMMON = "common"
    RARE = "rare"
    SHARED = "shared"
    NEVER = "never"
    SIGNAL_ONLY = "signal_only"


class AggregatorConfig:
    """Inputs and output settings for a cohort aggregation operation.

    The comma-delimited ``input_manifest`` has ``sample_id`` and
    ``methylseg_output_dir`` columns. Each output directory must contain the
    selected MethylSeg summary BED under ``summary_files``.
    """

    def __init__(
        self,
        input_manifest,
        output_root,
        *,
        cohort_name: str | None = None,
        region_type: MethylationStates | None = None,
        use_cleaned_regions: bool = True,
        chrom: str | None = None,
        chrom_sizes_path: str | Path | None = None,
        force_recreate: bool = False,
    ):
        self.input_manifest = Path(input_manifest).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.cohort_name = cohort_name
        if region_type is not None and not isinstance(region_type, MethylationStates):
            raise TypeError(
                "region_type must be a MethylationStates member or None."
            )
        self.region_type = region_type
        self.use_cleaned_regions = bool(use_cleaned_regions)
        self.chrom = None if chrom is None else str(chrom)
        self.chrom_sizes_path = (
            None
            if chrom_sizes_path is None
            else Path(chrom_sizes_path).expanduser().resolve()
        )
        self.force_recreate = bool(force_recreate)

class MethylSegAggregator:
    """Aggregate state-specific per-sample MethylSeg BED tracks."""

    MANIFEST_COLUMNS = {"sample_id", "methylseg_output_dir"}
    SIGNAL_COLUMNS = ["chrom", "cpg_start", "cpg_end", "signal"]
    METADATA_COLUMNS = SIGNAL_COLUMNS + ["contributing_sample_ids"]

    def __init__(self, config: AggregatorConfig):
        self.config = config

    def _manifest(self) -> pd.DataFrame:
        """Load and validate the completed-sample aggregation manifest."""
        if not self.config.input_manifest.is_file():
            raise FileNotFoundError(
                f"Aggregation manifest does not exist: {self.config.input_manifest}"
            )
        manifest = pd.read_csv(self.config.input_manifest)
        if set(manifest.columns) != self.MANIFEST_COLUMNS:
            raise ValueError(
                "Aggregation manifest must contain exactly "
                f"{sorted(self.MANIFEST_COLUMNS)}; found {list(manifest.columns)}."
            )
        if manifest.empty:
            raise ValueError("Aggregation manifest contains no samples.")
        manifest = manifest.copy()
        manifest["sample_id"] = manifest["sample_id"].astype(str)
        if manifest["sample_id"].duplicated().any():
            raise ValueError("Aggregation manifest has duplicate sample IDs.")
        manifest_base = self.config.input_manifest.parent
        manifest["methylseg_output_dir"] = manifest["methylseg_output_dir"].map(
            lambda value: str(
                (manifest_base / value).resolve()
                if not Path(value).is_absolute()
                else Path(value).resolve()
            )
        )
        missing_dirs = [
            output_dir
            for output_dir in manifest["methylseg_output_dir"]
            if not Path(output_dir).is_dir()
        ]
        if missing_dirs:
            raise FileNotFoundError(
                f"MethylSeg output directories do not exist: {missing_dirs}"
            )
        return manifest

    def _summary_path(self, output_dir: str) -> Path:
        """Return the selected combined raw or cleaned state-summary BED path."""
        prefix = "cleaned" if self.config.use_cleaned_regions else "raw"
        return (
            Path(output_dir)
            / "summary_files"
            / f"segments_{prefix}_{self.config.region_type.name}.bed"
        )

    def _load_sample_intervals(self, sample_id: str, output_dir: str) -> pd.DataFrame:
        """Load, chromosome-filter, and union one sample's state intervals."""
        summary_path = self._summary_path(output_dir)
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Selected summary BED for sample {sample_id!r} does not exist: "
                f"{summary_path}"
            )
        try:
            intervals = pd.read_csv(
                summary_path, sep="\t", header=None, usecols=[0, 1, 2]
            )
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["chrom", "start", "end"])
        intervals.columns = ["chrom", "start", "end"]
        if intervals.empty:
            return intervals
        intervals["chrom"] = intervals["chrom"].astype(str)
        intervals["start"] = pd.to_numeric(intervals["start"], errors="raise").astype(int)
        intervals["end"] = pd.to_numeric(intervals["end"], errors="raise").astype(int)
        if (intervals["end"] < intervals["start"]).any():
            raise ValueError(f"Summary BED has an end before its start: {summary_path}")
        intervals = intervals.loc[intervals["end"] > intervals["start"]].copy()
        if self.config.chrom is not None:
            intervals = intervals.loc[intervals["chrom"] == self.config.chrom].copy()
        return self._union_sample_intervals(intervals)

    def _chrom_sizes(self, aggregation_mode: AggregationMode) -> dict[str, int] | None:
        """Load chromosome bounds for complement-based never aggregation."""
        if aggregation_mode is not AggregationMode.NEVER:
            return None
        try:
            chrom_sizes = pd.read_csv(
                self.config.chrom_sizes_path,
                sep=r"\s+",
                header=None,
                usecols=[0, 1],
                names=["chrom", "size"],
                comment="#",
            )
        except pd.errors.EmptyDataError as error:
            raise ValueError("Chromosome sizes file contains no chromosome bounds.") from error
        chrom_sizes["chrom"] = chrom_sizes["chrom"].astype(str)
        chrom_sizes["size"] = pd.to_numeric(chrom_sizes["size"], errors="raise").astype(int)
        if chrom_sizes.empty or chrom_sizes["chrom"].duplicated().any():
            raise ValueError("Chromosome sizes must be non-empty with unique chromosomes.")
        if (chrom_sizes["size"] <= 0).any():
            raise ValueError("Chromosome sizes must be positive integers.")
        if self.config.chrom is not None:
            chrom_sizes = chrom_sizes.loc[
                chrom_sizes["chrom"] == self.config.chrom
            ].copy()
            if chrom_sizes.empty:
                raise ValueError(
                    f"Chromosome {self.config.chrom!r} is absent from chrom_sizes_path."
                )
        return dict(zip(chrom_sizes["chrom"], chrom_sizes["size"]))

    @staticmethod
    def _union_sample_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
        """Merge overlapping/touching intervals so one sample counts once."""
        if intervals.empty:
            return intervals
        merged_rows = []
        for chrom, chrom_intervals in intervals.groupby("chrom", sort=True):
            current_start = None
            current_end = None
            for row in chrom_intervals.sort_values(["start", "end"]).itertuples(index=False):
                if current_start is None:
                    current_start, current_end = row.start, row.end
                elif row.start <= current_end:
                    current_end = max(current_end, row.end)
                else:
                    merged_rows.append((chrom, current_start, current_end))
                    current_start, current_end = row.start, row.end
            merged_rows.append((chrom, current_start, current_end))
        return pd.DataFrame(merged_rows, columns=["chrom", "start", "end"])

    @staticmethod
    def _resolved_cutoff(cutoff: int | float, sample_count: int, name: str) -> float:
        """Convert an integer count or float sample proportion to a count."""
        if not isinstance(cutoff, (int, float)):
            raise TypeError(f"{name} must be an integer count or float proportion.")
        if isinstance(cutoff, int):
            if cutoff < 0:
                raise ValueError(f"{name} must be non-negative.")
            return float(cutoff)
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError(f"Float {name} must be between 0 and 1.")
        return cutoff * sample_count

    @staticmethod
    def _direct_cutoff(cutoff: int | float, name: str) -> float:
        """Validate a cutoff expressed directly in a signal file's value scale."""
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
            raise TypeError(f"{name} must be a non-negative numeric value.")
        cutoff = float(cutoff)
        if not math.isfinite(cutoff) or cutoff < 0:
            raise ValueError(f"{name} must be a finite non-negative value.")
        return cutoff

    def _signal_rows(
        self, manifest: pd.DataFrame, chrom_sizes: dict[str, int] | None
    ) -> list[dict]:
        """Sweep sample boundaries into signal intervals and contributor sets."""
        boundary_events: dict[str, dict[int, list[tuple[int, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for sample in manifest.itertuples(index=False):
            intervals = self._load_sample_intervals(
                sample.sample_id, sample.methylseg_output_dir
            )
            for interval in intervals.itertuples(index=False):
                boundary_events[interval.chrom][interval.start].append((1, sample.sample_id))
                boundary_events[interval.chrom][interval.end].append((-1, sample.sample_id))

        if chrom_sizes is not None:
            unknown_chroms = set(boundary_events).difference(chrom_sizes)
            if unknown_chroms:
                raise ValueError(
                    "Summary BED chromosomes are absent from chrom_sizes_path: "
                    f"{sorted(unknown_chroms)}"
                )
            for chrom, events in boundary_events.items():
                if any(position < 0 or position > chrom_sizes[chrom] for position in events):
                    raise ValueError(
                        f"Summary BED coordinates exceed chromosome bounds for {chrom!r}."
                    )

        signal_rows = []
        chromosomes = sorted(chrom_sizes) if chrom_sizes is not None else sorted(boundary_events)
        for chrom in chromosomes:
            active_samples: set[str] = set()
            events = boundary_events[chrom]
            positions = sorted(events)
            if chrom_sizes is not None:
                positions = sorted({0, chrom_sizes[chrom], *positions})
            for index, position in enumerate(positions[:-1]):
                for delta, sample_id in events[position]:
                    if delta == 1:
                        active_samples.add(sample_id)
                    else:
                        active_samples.discard(sample_id)
                next_position = positions[index + 1]
                if next_position > position and (active_samples or chrom_sizes is not None):
                    signal_rows.append(
                        {
                            "chrom": chrom,
                            "cpg_start": position,
                            "cpg_end": next_position,
                            "signal": len(active_samples),
                            "contributing_sample_ids": ",".join(sorted(active_samples)),
                        }
                    )
        return signal_rows

    @staticmethod
    def _select_signal_rows(
        signal_rows: list[dict],
        aggregation_mode: AggregationMode,
        minimum: float,
        maximum: float,
    ) -> list[dict]:
        """Return signal intervals selected by one aggregation mode."""
        if aggregation_mode is AggregationMode.COMMON:
            return [row for row in signal_rows if row["signal"] >= minimum]
        if aggregation_mode is AggregationMode.RARE:
            return [row for row in signal_rows if 0 < row["signal"] <= maximum]
        if aggregation_mode is AggregationMode.SHARED:
            return [
                row for row in signal_rows if minimum <= row["signal"] <= maximum
            ]
        if aggregation_mode is AggregationMode.NEVER:
            return [row for row in signal_rows if row["signal"] == 0]
        return []

    @classmethod
    def _read_signal_rows(cls, signal_path: Path) -> list[dict]:
        """Read and validate a headerless four-column signal bedGraph."""
        signal_path = Path(signal_path).expanduser()
        if not signal_path.is_file():
            raise FileNotFoundError(f"Signal file does not exist: {signal_path}")
        try:
            signal_frame = pd.read_csv(
                signal_path,
                sep="\t",
                header=None,
            )
        except pd.errors.EmptyDataError:
            return []
        if signal_frame.empty:
            return []
        if signal_frame.shape[1] != len(cls.SIGNAL_COLUMNS):
            raise ValueError(
                "Signal file must be a headerless tab-delimited bedGraph with "
                "exactly four columns: chrom, start, end, signal. "
                f"Found {signal_frame.shape[1]} columns in {signal_path}."
            )
        signal_frame.columns = cls.SIGNAL_COLUMNS
        signal_frame["chrom"] = signal_frame["chrom"].astype(str)
        for column in ("cpg_start", "cpg_end"):
            signal_frame[column] = pd.to_numeric(signal_frame[column], errors="raise")
            if signal_frame[column].isna().any() or (~signal_frame[column].map(
                float
            ).map(math.isfinite)).any() or (
                signal_frame[column] % 1 != 0
            ).any():
                raise ValueError(
                    f"Signal file has non-integer {column} coordinates: {signal_path}"
                )
            signal_frame[column] = signal_frame[column].astype(int)
        signal_frame["signal"] = pd.to_numeric(
            signal_frame["signal"], errors="raise"
        )
        if signal_frame["signal"].isna().any() or (
            ~signal_frame["signal"].map(float).map(math.isfinite)
        ).any():
            raise ValueError(f"Signal file has non-finite signal values: {signal_path}")
        signal_frame["signal"] = signal_frame["signal"].astype(float)
        if (signal_frame["cpg_start"] < 0).any():
            raise ValueError(f"Signal file has negative start coordinates: {signal_path}")
        if (signal_frame["cpg_end"] <= signal_frame["cpg_start"]).any():
            raise ValueError(f"Signal file has invalid interval bounds: {signal_path}")
        if (signal_frame["signal"] < 0).any():
            raise ValueError(f"Signal file has negative signal values: {signal_path}")
        return signal_frame.sort_values(
            ["chrom", "cpg_start", "cpg_end"], kind="stable"
        ).to_dict("records")

    @staticmethod
    def _merge_selected_rows(selected_rows: list[dict]) -> pd.DataFrame:
        """Merge adjacent selected signal spans into BED3 aggregate regions."""
        merged_rows = []
        for row in selected_rows:
            if (
                merged_rows
                and merged_rows[-1]["chrom"] == row["chrom"]
                and merged_rows[-1]["end"] == row["cpg_start"]
            ):
                merged_rows[-1]["end"] = row["cpg_end"]
            else:
                merged_rows.append(
                    {"chrom": row["chrom"], "start": row["cpg_start"], "end": row["cpg_end"]}
                )
        return pd.DataFrame(merged_rows, columns=["chrom", "start", "end"])

    @staticmethod
    def collect_regions(
        signal_file: str | Path,
        aggregation_mode: AggregationMode | str = AggregationMode.COMMON,
        peak_min_cutoff: int | float = 0.75,
        peak_max_cutoff: int | float = 0.25,
    ) -> pd.DataFrame:
        """Select and merge BED3 regions from a four-column signal bedGraph.

        ``signal_file`` must be a headerless, tab-delimited file containing
        ``chrom``, ``start``, ``end``, and a non-negative numeric signal value.
        Cutoffs are compared directly with those signal values.  The returned
        frame has BED3 columns ``chrom``, ``start``, and ``end``.
        """
        aggregation_mode = AggregationMode(aggregation_mode)
        if aggregation_mode is AggregationMode.SIGNAL_ONLY:
            raise ValueError("signal_only does not select aggregate regions.")
        minimum = MethylSegAggregator._direct_cutoff(
            peak_min_cutoff, "peak_min_cutoff"
        )
        maximum = MethylSegAggregator._direct_cutoff(
            peak_max_cutoff, "peak_max_cutoff"
        )
        signal_rows = MethylSegAggregator._read_signal_rows(Path(signal_file))
        selected_rows = MethylSegAggregator._select_signal_rows(
            signal_rows, aggregation_mode, minimum, maximum
        )
        return MethylSegAggregator._merge_selected_rows(selected_rows)

    def aggregate(
        self,
        aggregation_mode: AggregationMode | str = AggregationMode.COMMON,
        peak_min_cutoff: int | float = 0.75,
        peak_max_cutoff: int | float = 0.25,
    ) -> tuple[Path | None, Path, Path, Path]:
        """Write regions, signal, and contributor metadata for the cohort.

        Integer cutoffs represent sample counts; float cutoffs represent a
        proportion of manifest samples. The returned tuple is
        ``(aggregate_region_path_or_none, signal_path, normalized_signal_path,
        metadata_path)``. The normalized signal is each count divided by the
        number of manifest samples and is therefore in the interval [0, 1].
        """
        aggregation_mode = AggregationMode(aggregation_mode)
        if not self.config.cohort_name or not str(self.config.cohort_name).strip():
            raise ValueError("cohort_name is required for aggregation.")
        if self.config.region_type is None:
            raise ValueError("region_type is required for aggregation.")
        if aggregation_mode is AggregationMode.NEVER:
            if self.config.chrom_sizes_path is None:
                raise ValueError("chrom_sizes_path is required for never aggregation.")
            if not self.config.chrom_sizes_path.is_file():
                raise FileNotFoundError(
                    "Chromosome sizes file does not exist: "
                    f"{self.config.chrom_sizes_path}"
                )
        manifest = self._manifest()
        minimum = self._resolved_cutoff(
            peak_min_cutoff, len(manifest), "peak_min_cutoff"
        )
        maximum = self._resolved_cutoff(
            peak_max_cutoff, len(manifest), "peak_max_cutoff"
        )
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        stem = f"{self.config.cohort_name}.{self.config.region_type.name}"
        signal_path = self.config.output_root / f"{stem}.signal.bedgraph"
        normalized_signal_path = self.config.output_root / (
            f"{stem}.signal_normalized.bedgraph"
        )
        metadata_path = self.config.output_root / f"{stem}.signal_metadata.tsv"
        if signal_path.is_file() and not self.config.force_recreate:
            signal_rows = self._read_signal_rows(signal_path)
            if not normalized_signal_path.is_file():
                normalized_signal_frame = pd.DataFrame(
                    signal_rows, columns=self.SIGNAL_COLUMNS
                )
                normalized_signal_frame["signal"] = (
                    normalized_signal_frame["signal"] / len(manifest)
                )
                normalized_signal_frame.to_csv(
                    normalized_signal_path, sep="\t", header=False, index=False
                )
            if not metadata_path.is_file():
                metadata_rows = self._signal_rows(
                    manifest, self._chrom_sizes(aggregation_mode)
                )
                pd.DataFrame(metadata_rows, columns=self.METADATA_COLUMNS).to_csv(
                    metadata_path, sep="\t", index=False
                )
        else:
            signal_rows = self._signal_rows(
                manifest, self._chrom_sizes(aggregation_mode)
            )
            signal_frame = pd.DataFrame(signal_rows, columns=self.METADATA_COLUMNS)
            signal_frame.loc[:, self.SIGNAL_COLUMNS].to_csv(
                signal_path, sep="\t", header=False, index=False
            )
            normalized_signal_frame = signal_frame.loc[:, self.SIGNAL_COLUMNS].copy()
            normalized_signal_frame["signal"] = (
                normalized_signal_frame["signal"] / len(manifest)
            )
            normalized_signal_frame.to_csv(
                normalized_signal_path, sep="\t", header=False, index=False
            )
            signal_frame.to_csv(metadata_path, sep="\t", index=False)

        if aggregation_mode is AggregationMode.SIGNAL_ONLY:
            return None, signal_path, normalized_signal_path, metadata_path
        regions = self.collect_regions(
            signal_path, aggregation_mode, minimum, maximum
        )
        aggregate_path = self.config.output_root / (
            f"{stem}.{aggregation_mode.value}.bed"
        )
        if self.config.force_recreate or not aggregate_path.is_file():
            regions.to_csv(
                aggregate_path, sep="\t", header=False, index=False
            )
        return aggregate_path, signal_path, normalized_signal_path, metadata_path
