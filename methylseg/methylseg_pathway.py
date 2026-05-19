"""Top-level workflow orchestration for preparing, training, segmenting, and exporting."""

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from .methyl_state_analyzer import MethylStateAnalyzer
from .methylseg_config import MethylSegConfig
from .methylseg_hmm import (
    CTMethylSegHMM,
    GaussianMethylSegHMM,
    MultinomialSegHMM,
    StickyCategoricalMethylSegHMM,
)
from .methyl_segmentor import MethylSegmentor
from .methyl_state_assigner import MethylStateAssigner
from .helper_classes import (
    FILES,
    HMMObservationMode,
    KMeansMethylationModel,
    MethylDataPrep,
    MethylStateAssignmentMethod,
    MethylationStates,
    SampleInfo,
    HMMType,
)
from .data_manager import download_data_files, is_lfs_pointer
from .utils import resolve_region_overlay_df


class MethylSegPathway:
    """High-level API that coordinates preparation, training, and segmentation."""

    @classmethod
    def get_pretrained_model(
        cls, out_dir, resolution="450k", download_if_missing=False
    ):
        pretrained_yaml = (
            FILES / f"tcga_hm450k_model.yaml"
            if resolution == "450k"
            else FILES / f"wgbs_colon_model.yaml"
        )
        if not pretrained_yaml.exists() or is_lfs_pointer(pretrained_yaml):
            if download_if_missing:
                download_data_files()
            else:
                raise FileNotFoundError(
                    "Pretrained model not found, if you would like to download it, set download_if_missing=True."
                )
        model = cls.from_yaml(pretrained_yaml)
        model.set_out_dir(out_dir)
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

    # TODO: think about if train_sample* naming makes sense
    def __init__(
        self,
        n_states: int = 4,
        int_low_cutoff: float = 0.2,
        int_high_cutoff: float = 0.7,
        high_cutoff: float = 0.7,
        window_specs: list[tuple[int, str]] = [
            (40_000, "40kb"),
            (450_000, "450kb"),
        ],
        train_sample_info: SampleInfo | None = None,
        train_sample_file: str | None = None,
        train_sample_name: str | None = None,
        train_chroms: list[str] | None = None,
        max_cpg_per_chrom: int | None = 50_000,
        state_assignment_method: MethylStateAssignmentMethod = MethylStateAssignmentMethod.KMEANS,
        out_dir: str = ".",
        random_state: int = 42,
        # TODO: make cluster_space an enum
        cluster_space: str = "pca",
        n_pca: int | None = 5,
        # TODO: make hmm_type an enum
        hmm_type: HMMType = HMMType.CT,
        hmm_params: dict = {},
        min_region_length: int = 0,
        min_region_cpgs: int = 6,
        merge_gap_bp: int = 100_000,
        allow_pmd_expansion: bool = True,
        expansion_merge_bp: int = 100_000,
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
        self.train_sample_file = (
            str(train_sample_file) if train_sample_file is not None else None
        )
        self.train_sample_name = (
            str(train_sample_name)
            if train_sample_name is not None
            else str(self.train_sample_info.sample_id)
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
        self.allow_pmd_expansion = bool(allow_pmd_expansion)
        self.expansion_merge_bp = int(expansion_merge_bp)
        self.hmm_observation_mode = HMMObservationMode(hmm_observation_mode)
        if (
            self.hmm_observation_mode
            in (
                HMMObservationMode.GAUSSIAN_EMISSIONS,
                HMMObservationMode.PCA_EMISSIONS,
            )
            and self.hmm_type != HMMType.GAUSSIAN
        ):
            raise ValueError(
                "Gaussian-backed observation modes require " "hmm_type='gaussian'."
            )
        if self.hmm_type == HMMType.GAUSSIAN and self.hmm_observation_mode not in (
            HMMObservationMode.GAUSSIAN_EMISSIONS,
            HMMObservationMode.PCA_EMISSIONS,
        ):
            raise ValueError(
                "hmm_type='gaussian' requires " "a Gaussian-backed observation mode."
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
        self.assigner.train_sample_info = self.train_sample_info
        self.assigner.train_sample = self.train_sample_name
        self.analyzer = MethylStateAnalyzer(
            assigner=self.assigner,
            out_dir=out_dir,
        )
        self.segmentor = MethylSegmentor(
            analyzer=self.analyzer,
            hmm_model=self.hmm_model,
            state_assignment_method=self.state_assignment_method,
            hmm_observation_mode=self.hmm_observation_mode,
            out_dir=out_dir,
            random_state=self.random_state,
        )
        self.segmentor.default_sample_info = self.train_sample_info

    def set_out_dir(self, out_dir: str | Path) -> None:
        resolved_out_dir = str(Path(out_dir))
        Path(resolved_out_dir).mkdir(parents=True, exist_ok=True)
        self.out_dir = resolved_out_dir
        self.assigner.out_dir = resolved_out_dir
        self.analyzer.out_dir = resolved_out_dir
        self.segmentor.out_dir = resolved_out_dir

    def _resolve_sample_info(
        self,
        sample_info: SampleInfo | None = None,
        sample_name: str | None = None,
        sample_file: str | Path | None = None,
    ) -> SampleInfo:
        if sample_info is not None:
            return sample_info
        if sample_file is not None and sample_name is not None:
            resolved_sample_info, _ = self.prepare_sample_info(
                sample_name=sample_name,
                sample_file=sample_file,
                resolution="auto",
            )
            return resolved_sample_info
        if self.train_sample_info is not None:
            return self.train_sample_info
        raise ValueError(
            "Must provide either sample_info, sample_file and sample_name, "
            "or configure train_sample_info."
        )

    @staticmethod
    def _resolve_region_chrom(
        sample_info: SampleInfo,
        chrom: str | None = None,
        region_chrom: str | None = None,
    ) -> str:
        if region_chrom is not None:
            return str(region_chrom)
        if chrom is not None:
            return str(chrom)

        chrom_values = sample_info.meth_data["CpG_chrm"].dropna().astype(str).unique()
        if len(chrom_values) != 1:
            raise ValueError(
                "region_chrom must be provided when plotting multiple chromosomes."
            )
        return str(chrom_values[0])

    def _resolve_overlay_regions(
        self,
        *,
        sample_info: SampleInfo,
        chrom: str | None = None,
        overlay_regions_df: pd.DataFrame | None = None,
        overlay_state: MethylationStates | str | None = None,
        use_cleaned_regions: bool = False,
    ) -> tuple[pd.DataFrame | None, str]:
        direct_overlay_df, direct_overlay_style = resolve_region_overlay_df(
            overlay_regions_df=overlay_regions_df,
        )
        if direct_overlay_df is not None:
            return direct_overlay_df, direct_overlay_style

        if not use_cleaned_regions and overlay_state is None:
            return None, "state"

        target_state = MethylationStates.PMD if overlay_state is None else overlay_state
        target_state = self._coerce_region_state(target_state)
        resolved_chrom = self._resolve_region_chrom(
            sample_info=sample_info,
            chrom=chrom,
            region_chrom=None,
        )
        metadata_path = self._clean_region_metadata_path(
            chrom=resolved_chrom,
            sample_id=sample_info.sample_id,
            state=target_state,
        )
        if not metadata_path.exists():
            raise ValueError(
                "Clean regions must be generated first using get_clean_regions()."
            )
        clean_df = pd.read_csv(metadata_path, sep="\t")
        if not clean_df.empty and "state" in clean_df.columns:
            clean_df["state"] = clean_df["state"].apply(self._coerce_region_state)
        return clean_df, "state"

    @staticmethod
    def _resolve_artifact_path(
        yaml_dir: Path,
        raw_path: str | Path,
        field_name: str,
    ) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = yaml_dir / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Serialized artifact for '{field_name}' was not found: {path}"
            )
        return path

    @staticmethod
    def _relativize_artifact_path(base_dir: Path, path: Path) -> str:
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)

    @staticmethod
    def _normalize_train_chroms(train_chroms) -> list[str] | None:
        if train_chroms is None:
            return None
        if isinstance(train_chroms, str):
            return [train_chroms]
        return [str(chrom) for chrom in train_chroms]

    @staticmethod
    def _normalize_window_specs(window_specs) -> list[tuple[int, str]]:
        normalized_specs = []
        for spec in window_specs:
            if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                raise ValueError(
                    "window_specs entries must be two-item sequences of "
                    "(window_size_bp, label)."
                )
            window_size, label = spec
            normalized_specs.append((int(window_size), str(label)))
        return normalized_specs

    @staticmethod
    def _validate_state_cutoffs(state_cutoffs: dict) -> dict:
        if not isinstance(state_cutoffs, dict):
            raise ValueError("state_cutoffs must be a dictionary.")
        required_top_level = {"beta_low_max", "beta_high_min", "pmd_cutoffs"}
        missing_top_level = required_top_level.difference(state_cutoffs)
        if missing_top_level:
            raise ValueError(
                "state_cutoffs is missing required keys: "
                f"{sorted(missing_top_level)}"
            )
        pmd_cutoffs = state_cutoffs["pmd_cutoffs"]
        if not isinstance(pmd_cutoffs, dict) or not pmd_cutoffs:
            raise ValueError("state_cutoffs['pmd_cutoffs'] must be a non-empty dict.")

        normalized_cutoffs = {
            "beta_low_max": float(state_cutoffs["beta_low_max"]),
            "beta_high_min": float(state_cutoffs["beta_high_min"]),
            "pmd_cutoffs": {},
        }
        for label, cutoff_cfg in pmd_cutoffs.items():
            if not isinstance(cutoff_cfg, dict):
                raise ValueError(
                    f"state_cutoffs['pmd_cutoffs']['{label}'] must be a dict."
                )
            required_cutoff_keys = {"int_min", "std_max", "high_max"}
            missing_cutoff_keys = required_cutoff_keys.difference(cutoff_cfg)
            if missing_cutoff_keys:
                raise ValueError(
                    "PMD cutoff config is missing required keys for "
                    f"'{label}': {sorted(missing_cutoff_keys)}"
                )
            normalized_cutoffs["pmd_cutoffs"][str(label)] = {
                "int_min": float(cutoff_cfg["int_min"]),
                "std_max": float(cutoff_cfg["std_max"]),
                "high_max": float(cutoff_cfg["high_max"]),
            }
            if "low_max" in cutoff_cfg and cutoff_cfg["low_max"] is not None:
                normalized_cutoffs["pmd_cutoffs"][str(label)]["low_max"] = float(
                    cutoff_cfg["low_max"]
                )
        return normalized_cutoffs

    def _init_hmm(self):
        if isinstance(self.hmm_type, str):
            try:
                self.hmm_type = HMMType.from_string(self.hmm_type)
            except ValueError:
                raise ValueError(
                    f"Invalid hmm_type string: {self.hmm_type}. "
                    f"Valid options are: {[e.value for e in HMMType]}"
                )
        if self.hmm_type == HMMType.MULTINOMIAL:
            self.hmm_model = MultinomialSegHMM(
                n_states=self.n_states,
                random_state=self.random_state,
                **self.hmm_params,
            )
        elif self.hmm_type == HMMType.STICKY:
            self.hmm_model = StickyCategoricalMethylSegHMM(
                n_states=self.n_states,
                random_state=self.random_state,
                **self.hmm_params,
            )
        elif self.hmm_type == HMMType.CT:
            self.hmm_model = CTMethylSegHMM(
                n_states=self.n_states,
                **self.hmm_params,
            )
        elif self.hmm_type == HMMType.GAUSSIAN:
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
            self.state_assignment_method.value
            != MethylStateAssignmentMethod.KMEANS.value
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
    ) -> pd.DataFrame:
        sample_info = self._resolve_sample_info(
            sample_info=sample_info,
            sample_name=sample_name,
            sample_file=sample_file,
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
                "contains_intermediate",
                "n_segments",
                "n_pmd_segments",
                "n_intermediate_segments",
            ]
        )

    @staticmethod
    def _clean_metadata_columns(state: MethylationStates) -> list[str]:
        base_columns = [
            "CpG_chrm",
            "start",
            "end",
            "avg_beta",
            "probe_count",
            "state",
            "length",
            "n_segments",
        ]
        if state == MethylationStates.PMD:
            return base_columns + [
                "contains_intermediate",
                "n_pmd_segments",
                "n_intermediate_segments",
            ]
        return base_columns

    def _clean_regions_dir(self) -> Path:
        return Path(self.out_dir) / "clean_regions"

    def _resolve_clean_sample_id(self, sample_id: str | None = None) -> str:
        if sample_id is not None:
            return str(sample_id)
        default_sample = getattr(self.segmentor, "default_sample_info", None)
        if default_sample is not None and getattr(default_sample, "sample_id", None):
            return str(default_sample.sample_id)
        train_sample = getattr(self, "train_sample_info", None)
        if train_sample is not None and getattr(train_sample, "sample_id", None):
            return str(train_sample.sample_id)
        return "sample"

    def _clean_region_bed_path(
        self,
        *,
        chrom: str,
        sample_id: str,
        state: MethylationStates,
    ) -> Path:
        return self._clean_regions_dir() / (
            f"segments_cleaned_{chrom}_{sample_id}_{state.name}.bed"
        )

    def _clean_region_metadata_path(
        self,
        *,
        chrom: str,
        sample_id: str,
        state: MethylationStates,
    ) -> Path:
        return self._clean_regions_dir() / (
            f"metadata_cleaned_{chrom}_{sample_id}_{state.name}.tsv"
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

    def _merge_clean_region_records(
        self,
        records: list[dict[str, object]],
        *,
        gap_bp: int,
        state: MethylationStates,
    ) -> list[dict[str, object]]:
        if not records:
            return []

        sorted_records = sorted(
            records,
            key=lambda record: (
                str(record["CpG_chrm"]),
                int(record["start"]),
                int(record["end"]),
            ),
        )
        merged_records: list[dict[str, object]] = []
        current = sorted_records[0].copy()

        for record in sorted_records[1:]:
            same_chrom = str(record["CpG_chrm"]) == str(current["CpG_chrm"])
            current_end = int(current["end"])
            record_start = int(record["start"])
            if same_chrom and (record_start - current_end) <= int(gap_bp):
                current["end"] = max(current_end, int(record["end"]))
                current["beta_weighted_sum"] = float(
                    current["beta_weighted_sum"]
                ) + float(record["beta_weighted_sum"])
                current["probe_count"] = int(current["probe_count"]) + int(
                    record["probe_count"]
                )
                current["n_segments"] = int(current["n_segments"]) + int(
                    record["n_segments"]
                )
                current["contains_intermediate"] = bool(
                    current["contains_intermediate"]
                ) or bool(record["contains_intermediate"])
                current["n_pmd_segments"] = int(current["n_pmd_segments"]) + int(
                    record["n_pmd_segments"]
                )
                current["n_intermediate_segments"] = int(
                    current["n_intermediate_segments"]
                ) + int(record["n_intermediate_segments"])
                continue

            merged_records.append(
                {
                    "CpG_chrm": str(current["CpG_chrm"]),
                    "start": int(current["start"]),
                    "end": int(current["end"]),
                    "avg_beta": float(current["beta_weighted_sum"])
                    / int(current["probe_count"]),
                    "probe_count": int(current["probe_count"]),
                    "state": state,
                    "length": int(current["end"]) - int(current["start"]),
                    "contains_intermediate": bool(current["contains_intermediate"]),
                    "n_segments": int(current["n_segments"]),
                    "n_pmd_segments": int(current["n_pmd_segments"]),
                    "n_intermediate_segments": int(current["n_intermediate_segments"]),
                }
            )
            current = record.copy()

        merged_records.append(
            {
                "CpG_chrm": str(current["CpG_chrm"]),
                "start": int(current["start"]),
                "end": int(current["end"]),
                "avg_beta": float(current["beta_weighted_sum"])
                / int(current["probe_count"]),
                "probe_count": int(current["probe_count"]),
                "state": state,
                "length": int(current["end"]) - int(current["start"]),
                "contains_intermediate": bool(current["contains_intermediate"]),
                "n_segments": int(current["n_segments"]),
                "n_pmd_segments": int(current["n_pmd_segments"]),
                "n_intermediate_segments": int(current["n_intermediate_segments"]),
            }
        )
        return merged_records

    def _build_expanded_pmd_records(
        self,
        chrom_rows: list[object],
        *,
        expansion_merge_bp: int,
    ) -> list[dict[str, object]]:
        if not chrom_rows:
            return []

        expanded_records: list[dict[str, object]] = []
        block_rows: list[object] = []

        def flush_block() -> None:
            nonlocal block_rows
            if not block_rows:
                return

            has_pmd = any(row.state == MethylationStates.PMD for row in block_rows)
            if not has_pmd:
                block_rows = []
                return

            first_row = block_rows[0]
            record = {
                "CpG_chrm": str(first_row.CpG_chrm),
                "start": int(first_row.start),
                "end": int(first_row.end),
                "beta_weighted_sum": float(first_row.avg_beta)
                * int(first_row.probe_count),
                "probe_count": int(first_row.probe_count),
                "contains_intermediate": (
                    first_row.state == MethylationStates.INTERMEDIATE
                ),
                "n_segments": 1,
                "n_pmd_segments": (
                    1 if first_row.state == MethylationStates.PMD else 0
                ),
                "n_intermediate_segments": (
                    1 if first_row.state == MethylationStates.INTERMEDIATE else 0
                ),
            }
            for row in block_rows[1:]:
                record["end"] = max(int(record["end"]), int(row.end))
                record["beta_weighted_sum"] = float(record["beta_weighted_sum"]) + (
                    float(row.avg_beta) * int(row.probe_count)
                )
                record["probe_count"] = int(record["probe_count"]) + int(
                    row.probe_count
                )
                record["n_segments"] = int(record["n_segments"]) + 1
                if row.state == MethylationStates.PMD:
                    record["n_pmd_segments"] = int(record["n_pmd_segments"]) + 1
                elif row.state == MethylationStates.INTERMEDIATE:
                    record["contains_intermediate"] = True
                    record["n_intermediate_segments"] = (
                        int(record["n_intermediate_segments"]) + 1
                    )
            expanded_records.append(record)
            block_rows = []

        for row in chrom_rows:
            if row.state in (MethylationStates.PMD, MethylationStates.INTERMEDIATE):
                if block_rows:
                    gap_bp = int(row.start) - int(block_rows[-1].end)
                    if gap_bp <= int(expansion_merge_bp):
                        block_rows.append(row)
                        continue
                    flush_block()
                block_rows = [row]
                continue

            flush_block()

        flush_block()
        return expanded_records

    def _records_to_clean_state_df(
        self,
        records: list[dict[str, object]],
        *,
        state: MethylationStates,
        min_region_length: int,
        min_cpgs: int,
    ) -> pd.DataFrame:
        clean_columns = self._empty_clean_regions_df().columns.tolist()
        if not records:
            return self._empty_clean_regions_df()

        state_df = pd.DataFrame(records)
        state_df = state_df.loc[
            (state_df["length"] >= int(min_region_length))
            & (state_df["probe_count"] >= int(min_cpgs))
        ].copy()
        if state_df.empty:
            return self._empty_clean_regions_df()

        state_df["state"] = state_df["state"].astype(str)
        state_df["n_segments"] = pd.to_numeric(
            state_df["n_segments"], errors="raise"
        ).astype(int)
        if state == MethylationStates.PMD:
            state_df["contains_intermediate"] = state_df[
                "contains_intermediate"
            ].astype("boolean")
            state_df["n_pmd_segments"] = pd.to_numeric(
                state_df["n_pmd_segments"], errors="raise"
            ).astype("Int64")
            state_df["n_intermediate_segments"] = pd.to_numeric(
                state_df["n_intermediate_segments"], errors="raise"
            ).astype("Int64")
        else:
            state_df["contains_intermediate"] = pd.Series(
                pd.NA,
                index=state_df.index,
                dtype="boolean",
            )
            state_df["n_pmd_segments"] = pd.Series(
                pd.NA,
                index=state_df.index,
                dtype="Int64",
            )
            state_df["n_intermediate_segments"] = pd.Series(
                pd.NA,
                index=state_df.index,
                dtype="Int64",
            )
        return state_df.loc[:, clean_columns].reset_index(drop=True)

    def get_clean_regions(
        self,
        regions_df: pd.DataFrame | None = None,
        merge_gap_bp: int | None = None,
        min_region_length: int | None = None,
        min_cpgs: int | None = None,
        sample_id: str | None = None,
        chrom: str | None = None,
        allow_pmd_expansion: bool | None = None,
        expansion_merge_bp: int | None = None,
        generate_summary_files: bool = True,
    ) -> tuple[dict[MethylationStates, Path], Path]:
        """
        Build cleaned region tracks from the raw segmentation and write them to disk.

        The cleaning logic is chromosome-local and proceeds left-to-right over the
        full sorted segmentation table. Per-chromosome cleaned outputs are written
        to ``clean_regions/`` as chromosome-local BED and metadata TSV artifacts.
        When ``chrom`` is omitted, optional combined summary files can also be
        written to ``summary_files/segments_cleaned_{STATE}.bed`` and
        ``summary_files/metadata_cleaned_{STATE}.tsv``.

        The metadata TSV is the full cleaned record used by plotting; the BED is
        the slim interval export.

        PMD cleanup is the only cross-state expansion rule. When
        ``allow_pmd_expansion`` is true, PMD regions may absorb adjacent
        ``Intermediate`` segments if they are close enough under
        ``expansion_merge_bp``. ``LOW`` and ``HIGH`` always stop PMD expansion.
        Raw state labels are never relabeled in ``regions_df``; only the cleaned
        PMD output reflects the absorbed transitional segments.

        Parameters
        ----------
        regions_df
            Raw segmentation table. If omitted, uses ``self.segmentor.regions_df``.
        merge_gap_bp
            Maximum gap for ordinary same-state merging.
        min_region_length
            Minimum cleaned region length retained after merging.
        min_cpgs
            Minimum cleaned region probe count retained after merging.
        sample_id
            Optional sample identifier used in per-chrom cleaned filenames.
        chrom
            If provided, clean only this chromosome and write only chromosome-local
            cleaned artifacts for it.
        allow_pmd_expansion
            Whether PMD clean regions may absorb ``Intermediate`` segments.
        expansion_merge_bp
            Maximum gap allowed when PMD expands through ``Intermediate``.
        generate_summary_files
            When ``chrom`` is ``None``, also build combined cleaned summary files.

        Returns
        -------
        tuple[dict[MethylationStates, Path], Path]
            A mapping of combined cleaned summary BED paths plus the directory that
            holds the per-chromosome cleaned artifacts. The summary mapping is empty
            when summary generation is skipped.
        """
        # Resolve cleaning parameters from explicit inputs or pathway defaults.
        merge_gap_bp = self.merge_gap_bp if merge_gap_bp is None else int(merge_gap_bp)
        min_region_length = (
            self.min_region_length
            if min_region_length is None
            else int(min_region_length)
        )
        min_cpgs = self.min_region_cpgs if min_cpgs is None else int(min_cpgs)
        allow_pmd_expansion = (
            getattr(self, "allow_pmd_expansion", True)
            if allow_pmd_expansion is None
            else bool(allow_pmd_expansion)
        )
        expansion_merge_bp = (
            getattr(self, "expansion_merge_bp", 100_000)
            if expansion_merge_bp is None
            else int(expansion_merge_bp)
        )

        # Resolve the source regions and output location for this cleaning pass.
        if regions_df is None:
            regions_df = getattr(self.segmentor, "regions_df", None)
        resolved_sample_id = self._resolve_clean_sample_id(sample_id)
        clean_dir = self._clean_regions_dir()
        clean_dir.mkdir(parents=True, exist_ok=True)
        resolved_chrom = str(chrom) if chrom is not None else None

        # If there is nothing to clean, still write empty artifacts so downstream
        # plotting/export code can rely on the expected files being present.
        if regions_df is None or regions_df.empty:
            summary_paths = {}
            if resolved_chrom is not None:
                for state in MethylationStates:
                    empty_df = self._empty_clean_regions_df()
                    self._write_bed(
                        empty_df,
                        self._clean_region_bed_path(
                            chrom=resolved_chrom,
                            sample_id=resolved_sample_id,
                            state=state,
                        ),
                    )
                    empty_df.to_csv(
                        self._clean_region_metadata_path(
                            chrom=resolved_chrom,
                            sample_id=resolved_sample_id,
                            state=state,
                        ),
                        sep="\t",
                        index=False,
                    )
            elif generate_summary_files:
                self._write_summary_files(
                    raw_regions_df=None,
                    sample_id=resolved_sample_id,
                    clean_regions=True,
                    chroms=[],
                    write_raw_summaries=False,
                )
                summary_paths = {
                    state: Path(self.out_dir)
                    / "summary_files"
                    / f"segments_cleaned_{state.name}.bed"
                    for state in MethylationStates
                }
            return summary_paths, clean_dir

        # Validate and normalize the raw region table into one consistent schema
        # before running any chromosome-local cleaning stages.
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
        if resolved_chrom is not None:
            clean_df = clean_df.loc[
                clean_df["CpG_chrm"].astype(str) == resolved_chrom
            ].copy()
        clean_df = clean_df.sort_values(["CpG_chrm", "start", "end"]).reset_index(
            drop=True
        )

        # Decide which chromosomes need outputs. In chromosome-scoped mode we still
        # emit empty files even if the selected chromosome has no rows after filtering.
        target_chroms = (
            clean_df["CpG_chrm"].dropna().astype(str).drop_duplicates().tolist()
        )
        if not target_chroms and resolved_chrom is not None:
            target_chroms = [resolved_chrom]

        # Stage 1: split the normalized rows into per-chromosome streams so each
        # chromosome can be cleaned independently.
        grouped_rows = {
            str(chrom_name): list(chrom_df.itertuples(index=False))
            for chrom_name, chrom_df in clean_df.groupby("CpG_chrm", sort=False)
        }
        merged_records_by_chrom = {
            str(chrom_name): {state: [] for state in MethylationStates}
            for chrom_name in target_chroms
        }
        for chrom_name in target_chroms:
            chrom_rows = grouped_rows.get(str(chrom_name), [])
            raw_records_by_state: dict[MethylationStates, list[dict[str, object]]] = {
                state: [] for state in MethylationStates
            }

            # Stage 2: if PMD expansion is enabled, build the PMD candidates first.
            # Otherwise PMD is treated like the other states in the ordinary raw pass.
            if allow_pmd_expansion:
                raw_records_by_state[MethylationStates.PMD] = (
                    self._build_expanded_pmd_records(
                        chrom_rows,
                        expansion_merge_bp=expansion_merge_bp,
                    )
                )

            # Stage 3: build the ordinary same-state inputs directly from the raw
            # rows. When PMD expansion is disabled, raw PMD rows are included here too.
            for row in chrom_rows:
                if allow_pmd_expansion and row.state == MethylationStates.PMD:
                    continue
                raw_records_by_state[row.state].append(
                    {
                        "CpG_chrm": str(row.CpG_chrm),
                        "start": int(row.start),
                        "end": int(row.end),
                        "beta_weighted_sum": float(row.avg_beta) * int(row.probe_count),
                        "probe_count": int(row.probe_count),
                        "contains_intermediate": (
                            row.state == MethylationStates.INTERMEDIATE
                        ),
                        "n_segments": 1,
                        "n_pmd_segments": (
                            1 if row.state == MethylationStates.PMD else 0
                        ),
                        "n_intermediate_segments": (
                            1 if row.state == MethylationStates.INTERMEDIATE else 0
                        ),
                    }
                )

            # Stage 4: once PMD expansion is done, every state goes through the same
            # ordinary same-state merge rule.
            for state in MethylationStates:
                merged_records_by_chrom[str(chrom_name)][state] = (
                    self._merge_clean_region_records(
                        raw_records_by_state[state],
                        gap_bp=merge_gap_bp,
                        state=state,
                    )
                )

        # Stage 5: apply the shared filtering/formatting step and write the
        # chromosome-local BED/metadata outputs for each state.
        for chrom_name in target_chroms:
            for state in MethylationStates:
                state_df = self._records_to_clean_state_df(
                    merged_records_by_chrom[str(chrom_name)][state],
                    state=state,
                    min_region_length=min_region_length,
                    min_cpgs=min_cpgs,
                )
                self._write_bed(
                    state_df,
                    self._clean_region_bed_path(
                        chrom=chrom_name,
                        sample_id=resolved_sample_id,
                        state=state,
                    ),
                )
                state_df.loc[:, self._clean_metadata_columns(state)].to_csv(
                    self._clean_region_metadata_path(
                        chrom=chrom_name,
                        sample_id=resolved_sample_id,
                        state=state,
                    ),
                    sep="\t",
                    index=False,
                )

        summary_paths: dict[MethylationStates, Path] = {}
        # When running across all chromosomes, also build the combined summary
        # artifacts from the chromosome-local cleaned outputs written above.
        if resolved_chrom is None and generate_summary_files:
            self._write_summary_files(
                raw_regions_df=None,
                sample_id=resolved_sample_id,
                clean_regions=True,
                chroms=target_chroms,
                write_raw_summaries=False,
            )
            summary_paths = {
                state: Path(self.out_dir)
                / "summary_files"
                / f"segments_cleaned_{state.name}.bed"
                for state in MethylationStates
            }
        return summary_paths, clean_dir

    def _normalize_regions_for_bed(
        self,
        regions_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if regions_df is None or regions_df.empty:
            return self._empty_regions_df()

        normalized_df = regions_df.copy()
        normalized_df["CpG_chrm"] = normalized_df["CpG_chrm"].astype(str)
        normalized_df["start"] = pd.to_numeric(
            normalized_df["start"], errors="raise"
        ).astype(int)
        normalized_df["end"] = pd.to_numeric(
            normalized_df["end"], errors="raise"
        ).astype(int)
        return normalized_df

    def _write_bed(
        self,
        regions_df: pd.DataFrame | None,
        out_path: str | Path,
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bed_df = self._normalize_regions_for_bed(regions_df)
        bed_df = bed_df.loc[:, ["CpG_chrm", "start", "end", "state"]].copy()
        bed_df.to_csv(out_path, sep="\t", header=False, index=False)
        return out_path

    def _get_state_regions(
        self,
        regions_df: pd.DataFrame | None,
        state: MethylationStates,
    ) -> pd.DataFrame:
        normalized_df = self._normalize_regions_for_bed(regions_df)
        if normalized_df.empty:
            return normalized_df

        state_names = normalized_df["state"].astype(str)
        state_mask = (normalized_df["state"] == state) | (state_names == state.name)
        return normalized_df.loc[state_mask].copy()

    def _write_summary_files(
        self,
        raw_regions_df: pd.DataFrame | None,
        sample_id: str,
        clean_regions: bool = True,
        chroms: list[str] | None = None,
        write_raw_summaries: bool = True,
    ) -> list[str]:
        summary_dir = Path(self.out_dir) / "summary_files"
        summary_dir.mkdir(parents=True, exist_ok=True)

        written_paths: list[str] = []
        if write_raw_summaries:
            for state in MethylationStates:
                raw_state_df = self._get_state_regions(raw_regions_df, state)
                raw_path = summary_dir / f"segments_raw_{state.name}.bed"
                written_paths.append(str(self._write_bed(raw_state_df, raw_path)))

        if clean_regions:
            clean_columns = self._empty_clean_regions_df().columns.tolist()
            resolved_chroms = chroms
            if resolved_chroms is None:
                resolved_chroms = []
                for metadata_path in sorted(
                    self._clean_regions_dir().glob(
                        f"metadata_cleaned_*_{sample_id}_{MethylationStates.PMD.name}.tsv"
                    )
                ):
                    stem = metadata_path.stem.removeprefix("metadata_cleaned_")
                    resolved_chroms.append(
                        stem[: -len(f"_{sample_id}_{MethylationStates.PMD.name}")]
                    )
            for state in MethylationStates:
                frames = []
                for chrom_name in resolved_chroms:
                    metadata_path = self._clean_region_metadata_path(
                        chrom=str(chrom_name),
                        sample_id=sample_id,
                        state=state,
                    )
                    if not metadata_path.exists():
                        continue
                    state_df = pd.read_csv(metadata_path, sep="\t")
                    if not state_df.empty:
                        frames.append(state_df)
                if frames:
                    combined_df = pd.concat(frames, ignore_index=True)
                    combined_df["CpG_chrm"] = combined_df["CpG_chrm"].astype(str)
                    combined_df["start"] = pd.to_numeric(
                        combined_df["start"], errors="raise"
                    ).astype(int)
                    combined_df["end"] = pd.to_numeric(
                        combined_df["end"], errors="raise"
                    ).astype(int)
                    combined_df = combined_df.sort_values(
                        ["CpG_chrm", "start", "end"]
                    ).reset_index(drop=True)
                    combined_df = combined_df.reindex(columns=clean_columns)
                else:
                    combined_df = self._empty_clean_regions_df()
                clean_path = summary_dir / f"segments_cleaned_{state.name}.bed"
                self._write_bed(combined_df, clean_path)
                combined_df.loc[:, self._clean_metadata_columns(state)].to_csv(
                    summary_dir / f"metadata_cleaned_{state.name}.tsv",
                    sep="\t",
                    index=False,
                )
                written_paths.append(str(clean_path))

        return written_paths

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
                self._write_bed(state_df, bed_path)

    def run_on_all_chroms(
        self,
        sample_info: SampleInfo | None = None,
        chroms: Optional[List[str]] = None,
        min_probes: int = 3,
        force_resegment: bool = False,
        clean_regions: bool = True,
    ) -> list[str]:
        sample_info = self._resolve_sample_info(sample_info=sample_info)
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

        joint_hmm_types = (
            HMMType.STICKY,
            HMMType.MULTINOMIAL,
            HMMType.GAUSSIAN,
        )
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
            self.segmentor.regions_df = regions_df.copy()
            if clean_regions:
                self.get_clean_regions(
                    regions_df=regions_df,
                    sample_id=filtered_sample_info.sample_id,
                    chrom=None,
                    generate_summary_files=False,
                )
            return self._write_summary_files(
                raw_regions_df=regions_df,
                sample_id=filtered_sample_info.sample_id,
                clean_regions=clean_regions,
                chroms=resolved_chroms,
            )

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
        if clean_regions:
            self.get_clean_regions(
                regions_df=combined_regions_df,
                sample_id=filtered_sample_info.sample_id,
                chrom=None,
                generate_summary_files=False,
            )
        return self._write_summary_files(
            raw_regions_df=combined_regions_df,
            sample_id=filtered_sample_info.sample_id,
            clean_regions=clean_regions,
            chroms=resolved_chroms,
        )

    def run_pathway(
        self,
        sample_info: SampleInfo | None = None,
        chroms: Optional[List[str]] = None,
        min_probes: int = 3,
        force_optimize_rules: bool = False,
        force_resegment: bool = False,
        clean_regions: bool = True,
        verbose: bool = True,
    ) -> list[str]:
        sample_info = self._resolve_sample_info(sample_info=sample_info)
        if verbose:
            print("Fitting pathway...")
        self.fit_pathway(force_optimize_rules=force_optimize_rules)
        if verbose:
            print("Generating regions ...")
        return self.run_on_all_chroms(
            sample_info=sample_info,
            chroms=chroms,
            min_probes=min_probes,
            force_resegment=force_resegment,
            clean_regions=clean_regions,
        )

    # TODO fix region highlighting so that it adds verticle lines and zooms in on the region but still colors by state
    def plot_labels(
        self,
        *,
        label_source: str = "hmm",
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
        sample_info_removed: pd.DataFrame | None = None,
        overlay_regions_df: pd.DataFrame | None = None,
        overlay_state: MethylationStates | str | None = None,
        use_cleaned_regions: bool = False,
        region_start: int | None = None,
        region_end: int | None = None,
        region_chrom: str | None = None,
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,
    ):
        """
        Plot genomic labels for one chromosome with optional region overlays.

        ``use_cleaned_regions=True`` is a read-only plotting mode. It loads the
        prebuilt chromosome-local cleaned metadata TSV for ``overlay_state`` from
        ``clean_regions/metadata_cleaned_{chrom}_{sample_id}_{STATE}.tsv``.
        It does not generate cleaned regions on demand; call
        ``get_clean_regions(..., chrom=...)`` first when you want cleaned
        overlays.
        """
        resolved_sample_info = self._resolve_sample_info(sample_info=sample_info)
        overlay_df, overlay_style = self._resolve_overlay_regions(
            sample_info=resolved_sample_info,
            chrom=chrom,
            overlay_regions_df=overlay_regions_df,
            overlay_state=overlay_state,
            use_cleaned_regions=use_cleaned_regions,
        )

        label_source = str(label_source).lower()
        if label_source == "hmm":
            return self.segmentor.plot_labels(
                sample_info=resolved_sample_info,
                chrom=chrom,
                sample_info_removed=sample_info_removed,
                overlay_regions_df=overlay_df,
                overlay_style=overlay_style,
                region_start=region_start,
                region_end=region_end,
                region_chrom=(
                    self._resolve_region_chrom(
                        sample_info=resolved_sample_info,
                        chrom=chrom,
                        region_chrom=region_chrom,
                    )
                    if any(
                        value is not None
                        for value in (region_start, region_end, region_chrom)
                    )
                    else None
                ),
                x_col=x_col,
                y_col=y_col,
                label_title=label_title if label_title is not None else "HMM state",
                show_plot=show_plot,
                max_points=max_points,
            )
        if label_source in {"kmeans", "rule_based"}:
            return self.analyzer.plot_labels(
                sample_info=resolved_sample_info,
                chrom=chrom,
                sample_info_removed=sample_info_removed,
                label_source=label_source,
                overlay_regions_df=overlay_df,
                overlay_style=overlay_style,
                region_start=region_start,
                region_end=region_end,
                region_chrom=(
                    self._resolve_region_chrom(
                        sample_info=resolved_sample_info,
                        chrom=chrom,
                        region_chrom=region_chrom,
                    )
                    if any(
                        value is not None
                        for value in (region_start, region_end, region_chrom)
                    )
                    else None
                ),
                x_col=x_col,
                y_col=y_col,
                label_title=label_title,
                show_plot=show_plot,
                max_points=max_points,
            )

        raise ValueError(
            "label_source must be one of 'kmeans', 'hmm', or 'rule_based'. "
            f"Received: {label_source!r}"
        )

    def plot_embedding(
        self,
        *,
        label_source: str = "kmeans",
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
        method: str = "pca",
        n_components: int = 2,
        top_n_loadings: int = 5,
        hexbin: bool = True,
        interactive: bool = False,
        include_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        region_start: int | None = None,
        region_end: int | None = None,
        region_chrom: str | None = None,
        use_pca_features: bool = False,
        use_parallel: bool = True,
        show_plot: bool = True,
        force_resegment: bool = False,
    ):
        label_source = str(label_source).lower()
        if label_source == "kmeans" and sample_info is None:
            return self.assigner.plot_training_embedding(
                method=method,
                n_components=n_components,
                top_n_loadings=top_n_loadings,
                hexbin=hexbin,
                interactive=interactive,
                include_metrics=include_metrics,
                include_biplot=include_biplot,
                label_title=label_title,
                region_start=region_start,
                region_end=region_end,
                region_chrom=region_chrom,
                use_pca_features=use_pca_features,
                use_parallel=use_parallel,
                show_plot=show_plot,
            )

        resolved_sample_info = self._resolve_sample_info(sample_info=sample_info)

        if label_source == "kmeans":
            meth_data, emission_df, _, _, _, labels = (
                self.assigner.apply_kmeans_to_sample(
                    sample_info=resolved_sample_info,
                    chrom=chrom,
                )
            )
            return self.assigner.plot_embedding(
                emission_df=emission_df,
                labels=labels,
                meth_data=meth_data,
                method=method,
                sample_info=resolved_sample_info,
                chrom=chrom,
                n_components=n_components,
                top_n_loadings=top_n_loadings,
                hexbin=hexbin,
                interactive=interactive,
                include_metrics=include_metrics,
                include_biplot=include_biplot,
                label_title=label_title,
                region_start=region_start,
                region_end=region_end,
                region_chrom=region_chrom,
                use_pca_features=use_pca_features,
                use_parallel=use_parallel,
                show_plot=show_plot,
            )

        if label_source == "hmm":
            meth_data, _ = self.segmentor.segment_sample(
                sample_info=resolved_sample_info,
                chrom=chrom,
                force_resegment=force_resegment,
            )
            return self.assigner.plot_embedding(
                emission_df=self.segmentor.emissions_df,
                labels=meth_data["hmm_state_readable"].to_numpy(),
                meth_data=meth_data,
                method=method,
                sample_info=resolved_sample_info,
                chrom=chrom,
                n_components=n_components,
                top_n_loadings=top_n_loadings,
                hexbin=hexbin,
                interactive=interactive,
                include_metrics=include_metrics,
                include_biplot=include_biplot,
                label_title=label_title,
                region_start=region_start,
                region_end=region_end,
                region_chrom=region_chrom,
                use_pca_features=use_pca_features,
                use_parallel=use_parallel,
                show_plot=show_plot,
            )

        raise ValueError(
            "label_source must be either 'kmeans' or 'hmm' for embedding plots. "
            f"Received: {label_source!r}"
        )

    def to_yaml(self, yaml_path: str, include_learned: bool = True):
        """
        Serialize pathway configuration and optionally learned artifacts.
        """
        MethylSegConfig.from_instance(
            self,
            out_dir=str(Path(yaml_path).parent),
            include_learned=include_learned,
        ).to_yaml(yaml_path)

    @classmethod
    def from_yaml(cls, yaml_path: str, load_learned: bool = True):
        """
        Reconstruct MethylSegPathway from YAML file.
        """
        return MethylSegConfig.from_yaml(yaml_path).build_pathway(
            load_learned=load_learned,
        )
