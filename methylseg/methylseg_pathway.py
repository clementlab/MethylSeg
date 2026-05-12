from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from .methyl_state_analyzer import MethylStateAnalyzer
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
                "Gaussian-backed observation modes require " "hmm_type='gaussian'."
            )
        if self.hmm_type == "gaussian" and self.hmm_observation_mode not in {
            HMMObservationMode.GAUSSIAN_EMISSIONS,
            HMMObservationMode.PCA_EMISSIONS,
        }:
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
        state: MethylationStates | str = MethylationStates.PMD,
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
