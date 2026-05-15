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
        clean_regions: bool = False,
        region_start: int | None = None,
        region_end: int | None = None,
        region_chrom: str | None = None,
        force_resegment: bool = False,
        min_probes: int = 3,
        min_region_length: int | None = None,
        min_cpgs: int | None = None,
        merge_gap_bp: int | None = None,
    ) -> tuple[pd.DataFrame | None, str]:
        direct_overlay_df, direct_overlay_style = resolve_region_overlay_df(
            overlay_regions_df=overlay_regions_df,
        )
        if direct_overlay_df is not None:
            return direct_overlay_df, direct_overlay_style

        if not clean_regions and overlay_state is None:
            return None, "state"

        target_state = MethylationStates.PMD if overlay_state is None else overlay_state

        if chrom is None:
            self.run_on_all_chroms(
                sample_info=sample_info,
                chroms=None,
                min_probes=min_probes,
                force_resegment=force_resegment,
                clean_regions=False,
            )
            raw_regions_df = getattr(self.segmentor, "regions_df", None)
        else:
            raw_regions_df = self.generate_regions(
                sample_info=sample_info,
                chrom=chrom,
                min_probes=min_probes,
                force_resegment=force_resegment,
            )

        clean_df = self.get_clean_regions(
            regions_df=raw_regions_df,
            state=target_state,
            merge_gap_bp=merge_gap_bp,
            min_region_length=min_region_length,
            min_cpgs=min_cpgs,
            sample_id=sample_info.sample_id,
            chrom=chrom,
            force_resegment=force_resegment,
        )
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
            ]
        )

    def _clean_region_cache_path(
        self,
        *,
        sample_id: str,
        chrom: str,
        state: MethylationStates,
        merge_gap_bp: int,
        min_region_length: int,
        min_cpgs: int,
    ) -> Path:
        return Path(self.out_dir) / (
            "segments_cleaned_"
            f"{chrom}_{sample_id}_{state.name}_"
            f"gap{merge_gap_bp}_len{min_region_length}_cpg{min_cpgs}.bed"
        )

    @staticmethod
    def _write_clean_region_cache(
        clean_df: pd.DataFrame,
        out_path: Path,
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if clean_df is None or clean_df.empty:
            bed_df = pd.DataFrame(
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
        else:
            bed_df = clean_df.loc[
                :,
                [
                    "CpG_chrm",
                    "start",
                    "end",
                    "avg_beta",
                    "probe_count",
                    "state",
                    "length",
                ],
            ].copy()
            bed_df["state"] = bed_df["state"].astype(str)
        bed_df.to_csv(out_path, sep="\t", header=False, index=False)
        return out_path

    def _read_clean_region_cache(
        self,
        cache_path: Path,
        *,
        state: MethylationStates,
    ) -> pd.DataFrame:
        if not cache_path.exists():
            return self._empty_clean_regions_df()

        try:
            clean_df = pd.read_csv(
                cache_path,
                sep="\t",
                header=None,
                names=[
                    "CpG_chrm",
                    "start",
                    "end",
                    "avg_beta",
                    "probe_count",
                    "state",
                    "length",
                ],
            )
        except pd.errors.EmptyDataError:
            return self._empty_clean_regions_df()
        if clean_df.empty:
            return self._empty_clean_regions_df()

        clean_df["CpG_chrm"] = clean_df["CpG_chrm"].astype(str)
        clean_df["start"] = pd.to_numeric(clean_df["start"], errors="raise").astype(int)
        clean_df["end"] = pd.to_numeric(clean_df["end"], errors="raise").astype(int)
        clean_df["avg_beta"] = pd.to_numeric(clean_df["avg_beta"], errors="raise")
        clean_df["probe_count"] = pd.to_numeric(
            clean_df["probe_count"], errors="raise"
        ).astype(int)
        clean_df["state"] = clean_df["state"].apply(self._coerce_region_state)
        clean_df["length"] = pd.to_numeric(clean_df["length"], errors="raise").astype(
            int
        )
        clean_df = clean_df.loc[clean_df["state"] == state].copy()
        if clean_df.empty:
            return self._empty_clean_regions_df()
        return clean_df.reset_index(drop=True)

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
        state: MethylationStates | str = MethylationStates.PMD,
        merge_gap_bp: int | None = None,
        min_region_length: int | None = None,
        min_cpgs: int | None = None,
        sample_id: str | None = None,
        chrom: str | None = None,
        force_resegment: bool = False,
    ) -> pd.DataFrame:
        target_state = self._coerce_region_state(state)
        merge_gap_bp = self.merge_gap_bp if merge_gap_bp is None else int(merge_gap_bp)
        min_region_length = (
            self.min_region_length
            if min_region_length is None
            else int(min_region_length)
        )
        min_cpgs = self.min_region_cpgs if min_cpgs is None else int(min_cpgs)

        resolved_chrom = str(chrom) if chrom is not None else None
        cache_path = None
        if sample_id is not None:
            if regions_df is not None and not regions_df.empty and resolved_chrom is None:
                chrom_values = regions_df["CpG_chrm"].dropna().astype(str).unique().tolist()
                if len(chrom_values) == 1:
                    resolved_chrom = chrom_values[0]
            if resolved_chrom is not None:
                cache_path = self._clean_region_cache_path(
                    sample_id=str(sample_id),
                    chrom=resolved_chrom,
                    state=target_state,
                    merge_gap_bp=merge_gap_bp,
                    min_region_length=min_region_length,
                    min_cpgs=min_cpgs,
                )
                if cache_path.exists() and not force_resegment:
                    return self._read_clean_region_cache(
                        cache_path,
                        state=target_state,
                    )

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
        if resolved_chrom is not None:
            clean_df = clean_df.loc[
                clean_df["CpG_chrm"].astype(str) == resolved_chrom
            ].copy()
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
        # pylint: disable=unsubscriptable-object,unsupported-assignment-operation
        merged_df: pd.DataFrame = pd.DataFrame(merged_regions)
        merged_df["avg_beta"] = (
            merged_df["beta_weighted_sum"] / merged_df["probe_count"]
        )
        merged_df: pd.DataFrame = merged_df.drop(columns=["beta_weighted_sum"])
        merged_df["length"] = merged_df["end"] - merged_df["start"]
        merged_df = merged_df.loc[
            (merged_df["length"] >= int(min_region_length))
            & (merged_df["probe_count"] >= int(min_cpgs))
        ].copy()

        if merged_df.empty:
            if cache_path is not None:
                self._write_clean_region_cache(self._empty_clean_regions_df(), cache_path)
            return self._empty_clean_regions_df()

        merged_df = merged_df.loc[
            :,
            ["CpG_chrm", "start", "end", "avg_beta", "probe_count", "state", "length"],
        ].reset_index(drop=True)
        if cache_path is not None:
            self._write_clean_region_cache(merged_df, cache_path)
        return merged_df

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
        clean_regions: bool = True,
    ) -> list[str]:
        summary_dir = Path(self.out_dir) / "summary_files"
        summary_dir.mkdir(parents=True, exist_ok=True)

        written_paths: list[str] = []
        for state in MethylationStates:
            raw_state_df = self._get_state_regions(raw_regions_df, state)
            raw_path = summary_dir / f"segments_raw_{state.name}.bed"
            written_paths.append(str(self._write_bed(raw_state_df, raw_path)))

            if clean_regions:
                clean_state_df = self.get_clean_regions(
                    regions_df=raw_regions_df,
                    state=state,
                )
                clean_path = summary_dir / f"segments_cleaned_{state.name}.bed"
                written_paths.append(str(self._write_bed(clean_state_df, clean_path)))

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
            return self._write_summary_files(
                raw_regions_df=regions_df,
                clean_regions=clean_regions,
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
        return self._write_summary_files(
            raw_regions_df=combined_regions_df,
            clean_regions=clean_regions,
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
    # TODO fix this so that it does not need to recalculate cleaned reagions for every plot if they are already calculated
    def plot_labels(
        self,
        *,
        label_source: str = "hmm",
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
        sample_info_removed: pd.DataFrame | None = None,
        overlay_regions_df: pd.DataFrame | None = None,
        overlay_state: MethylationStates | str | None = None,
        clean_regions: bool = False,
        region_start: int | None = None,
        region_end: int | None = None,
        region_chrom: str | None = None,
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,
        min_probes: int = 3,
        force_resegment: bool = False,
        min_region_length: int | None = None,
        min_cpgs: int | None = None,
        merge_gap_bp: int | None = None,
    ):
        resolved_sample_info = self._resolve_sample_info(sample_info=sample_info)
        overlay_df, overlay_style = self._resolve_overlay_regions(
            sample_info=resolved_sample_info,
            chrom=chrom,
            overlay_regions_df=overlay_regions_df,
            overlay_state=overlay_state,
            clean_regions=clean_regions,
            region_start=region_start,
            region_end=region_end,
            region_chrom=region_chrom,
            force_resegment=force_resegment,
            min_probes=min_probes,
            min_region_length=min_region_length,
            min_cpgs=min_cpgs,
            merge_gap_bp=merge_gap_bp,
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
