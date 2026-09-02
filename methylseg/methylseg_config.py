"""Serialization helpers for saving and restoring methylseg workflow settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import yaml

from .helper_classes import (
    HMMObservationMode,
    MethylStateAssignmentMethod,
    MethylationStates,
    SampleInfo,
)

if TYPE_CHECKING:
    from .methylseg_pathway import MethylSegPathway


class MethylSegConfig:
    """
    Lightweight helper that knows how to build a serializable dictionary
    from a MethylSegPathway instance and how to write/read YAML.
    """

    def __init__(self, config_dict: dict):
        """
        Initialize a serializable MethylSeg configuration wrapper.

        Parameters
        ----------
        config_dict
            Configuration mapping to validate, serialize, or use when restoring
            a pathway.
        """
        self.config = config_dict
        self.source_path: Path | None = None

    @staticmethod
    def _resolve_artifact_path(
        yaml_dir: Path,
        raw_path: str | Path,
        field_name: str,
    ) -> Path:
        path = Path(raw_path)

        if not path.is_absolute():
            path = (yaml_dir / path).resolve()
        else:
            path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Missing artifact for {field_name}: {path}")

        return path

    @staticmethod
    def _relativize_artifact_path(base_dir: Path, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            return str(path.resolve())

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
                    "window_specs entries must be length-2 sequences of (size, label)."
                )

            size, label = spec
            normalized_specs.append((int(size), str(label)))

        return normalized_specs

    @staticmethod
    def _validate_state_cutoffs(state_cutoffs: dict) -> dict:
        if not isinstance(state_cutoffs, dict):
            raise ValueError("state_cutoffs must be a dictionary.")

        required_top_level = {"beta_low_max", "beta_high_min", "pmd_cutoffs"}

        missing_top_level = required_top_level.difference(state_cutoffs)

        if missing_top_level:
            raise ValueError(
                f"state_cutoffs is missing required keys: "
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
                raise ValueError(f"pmd_cutoffs['{label}'] must be a dict.")

            missing_keys = {"int_min", "std_max", "high_max"}.difference(cutoff_cfg)

            if missing_keys:
                raise ValueError(
                    f"pmd_cutoffs['{label}'] is missing keys: "
                    f"{sorted(missing_keys)}"
                )

            normalized_cutoffs["pmd_cutoffs"][label] = {
                "int_min": float(cutoff_cfg["int_min"]),
                "std_max": float(cutoff_cfg["std_max"]),
                "high_max": float(cutoff_cfg["high_max"]),
                "low_max": float(cutoff_cfg.get("low_max", cutoff_cfg["high_max"])),
            }

        return normalized_cutoffs

    @staticmethod
    def _normalize_table_for_feather(table: pd.DataFrame) -> pd.DataFrame:
        normalized_table = table.copy()

        for column_name in normalized_table.columns:

            column = normalized_table[column_name]

            if not pd.api.types.is_object_dtype(column.dtype):
                continue

            non_null_values = column.dropna()

            if non_null_values.empty:
                continue

            if all(isinstance(value, Enum) for value in non_null_values):
                normalized_table[column_name] = column.map(
                    lambda value: value.value if isinstance(value, Enum) else value
                )
                continue

            if all(isinstance(value, (str, bytes)) for value in non_null_values):
                normalized_table[column_name] = column.astype("string")
                continue

            normalized_table[column_name] = column.map(
                lambda value: (
                    str(value) if value is not None and not pd.isna(value) else value
                )
            )

        return normalized_table

    @staticmethod
    def _coerce_state_value(value):
        if value is None or pd.isna(value):
            return value

        if isinstance(value, MethylationStates):
            return value

        if isinstance(value, str):
            try:
                return MethylationStates.from_string(value)
            except ValueError:
                try:
                    return MethylationStates(int(value))
                except (TypeError, ValueError):
                    return value

        try:
            return MethylationStates(int(value))
        except (TypeError, ValueError):
            return value

    @classmethod
    def _restore_table_after_feather(
        cls,
        table_name: str,
        table: pd.DataFrame,
    ) -> pd.DataFrame:

        restored_table = table.copy()

        if "kmeans_label" in restored_table.columns:
            restored_table["kmeans_label"] = (
                restored_table["kmeans_label"]
                .astype(str)
            )

        if "rule_based_label" in restored_table.columns:
            restored_table["rule_based_label"] = (
                restored_table["rule_based_label"]
                .astype(str)
            )

        if table_name == "train_labels" and not restored_table.empty:
            restored_table.iloc[:, 0] = restored_table.iloc[:, 0].map(
                cls._coerce_state_value
            )

        return restored_table

    @staticmethod
    def _write_feather_table(table: pd.DataFrame, table_path: Path) -> None:
        normalized_table = MethylSegConfig._normalize_table_for_feather(table)

        normalized_table.reset_index(drop=True).to_feather(table_path)

    @classmethod
    def _relocate_artifact_reference(
        cls,
        raw_path: str | Path | None,
        source_dir: Path,
        target_dir: Path,
    ) -> str | None:
        if raw_path is None:
            return None

        path = Path(raw_path)
        if not path.is_absolute():
            path = (source_dir / path).resolve()
        else:
            path = path.resolve()

        return cls._relativize_artifact_path(target_dir, path)

    def rewrite_artifact_paths(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
    ) -> "MethylSegConfig":
        """
        Rewrite serialized artifact references from one base directory to another.

        Parameters
        ----------
        source_dir
            Directory that relative artifact paths in ``self.config`` currently
            resolve against.
        target_dir
            Directory the rewritten paths should resolve against, typically the
            parent directory of the YAML file being written.

        Returns
        -------
        MethylSegConfig
            The same config wrapper with rewritten artifact references.
        """

        source_dir = Path(source_dir).resolve()
        target_dir = Path(target_dir).resolve()
        cfg = self.config

        for field_name, raw_path in list(cfg.get("saved_tables", {}).items()):
            cfg["saved_tables"][field_name] = self._relocate_artifact_reference(
                raw_path,
                source_dir,
                target_dir,
            )

        train_sample_info = cfg.get("train_sample_info")
        if isinstance(train_sample_info, dict) and "meth_data_path" in train_sample_info:
            train_sample_info["meth_data_path"] = self._relocate_artifact_reference(
                train_sample_info.get("meth_data_path"),
                source_dir,
                target_dir,
            )

        models_cfg = cfg.get("models", {})
        for field_name in ("kmeans", "scaler", "imputer", "pca"):
            if field_name not in models_cfg:
                continue
            models_cfg[field_name] = self._relocate_artifact_reference(
                models_cfg.get(field_name),
                source_dir,
                target_dir,
            )

        training_cfg = cfg.get("training_artifacts", {})
        if isinstance(training_cfg, dict):
            for field_name, raw_path in list(training_cfg.items()):
                training_cfg[field_name] = self._relocate_artifact_reference(
                    raw_path,
                    source_dir,
                    target_dir,
                )

        hmm_cfg = cfg.get("hmm", {})
        hmm_params = hmm_cfg.get("params", {})
        if isinstance(hmm_params, dict):
            for param_name, param_value in hmm_params.items():
                if not isinstance(param_value, dict):
                    continue
                npy_path = param_value.get("__npy_path__")
                if npy_path is None:
                    continue
                param_value["__npy_path__"] = self._relocate_artifact_reference(
                    npy_path,
                    source_dir,
                    target_dir,
                )

        return self

    @classmethod
    def from_instance(
        cls,
        inst: "MethylSegPathway",
        out_dir: str | None = None,
        include_learned: bool = True,
    ) -> "MethylSegConfig":
        """
        Serialize a fitted pathway into a portable configuration bundle.

        Parameters
        ----------
        inst
            Fitted ``MethylSegPathway`` instance to serialize.
        out_dir
            Directory where the YAML-adjacent artifacts should be written. When
            omitted, uses the pathway output directory.
        include_learned
            If ``True``, also write learned models and training artifacts such as
            PCA scores, labels, and feather tables.

        Returns
        -------
        MethylSegConfig
            Config wrapper whose ``config`` dictionary references any artifacts
            written during serialization.
        """

        base_dir = Path(out_dir or inst.out_dir or ".").resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

        cfg = {}

        cfg["pathway"] = {
            "data_path": getattr(inst, "data_path", None),
            "meth_ref_path": getattr(inst, "meth_ref_path", None),
            "samples_info_path": getattr(inst, "samples_info_path", None),
            "out_dir": ".",
            "train_sample": getattr(inst, "train_sample_name", None),
            "train_sample_file": getattr(inst, "train_sample_file", None),
            "train_chroms": getattr(inst, "train_chroms", None),
            "max_cpg_per_chrom": getattr(inst, "max_cpg_per_chrom", None),
            "random_state": getattr(inst, "random_state", None),
            "cluster_space": getattr(inst, "cluster_space", None),
            "n_pca": getattr(inst, "n_pca", None),
            "min_region_length": getattr(inst, "min_region_length", None),
            "min_region_cpgs": getattr(inst, "min_region_cpgs", None),
            "merge_gap_bp": getattr(inst, "merge_gap_bp", None),
        }

        cfg["state_assigner"] = {
            "window_specs": getattr(inst.assigner, "window_specs", None),
            "n_states": getattr(inst.assigner, "n_states", None),
            "int_low_cutoff": getattr(inst.assigner, "int_low_cutoff", None),
            "int_high_cutoff": getattr(inst.assigner, "int_high_cutoff", None),
            "high_cutoff": getattr(inst.assigner, "high_cutoff", None),
        }

        if hasattr(inst.analyzer, "state_cutoffs"):
            cfg["state_cutoffs"] = cls._validate_state_cutoffs(
                inst.analyzer.state_cutoffs
            )

            cfg["state_cutoffs_set_manually"] = bool(
                getattr(inst.analyzer, "cutoffs_set_manually", False)
            )

        hmm_params = getattr(inst, "hmm_params", {}) or {}

        hmm_param_dir = base_dir / "hmm_params"
        hmm_param_dir.mkdir(exist_ok=True, parents=True)

        serializable_hmm_params = {}

        for k, v in hmm_params.items():

            if isinstance(v, np.ndarray):

                filename = f"{k}.npy"
                p = hmm_param_dir / filename

                np.save(p, np.asarray(v))

                serializable_hmm_params[k] = {
                    "__npy_path__": cls._relativize_artifact_path(base_dir, p)
                }

            elif isinstance(v, (list, tuple)):
                serializable_hmm_params[k] = list(v)

            else:
                serializable_hmm_params[k] = v

        cfg["hmm"] = {
            "type": getattr(inst, "hmm_type", None),
            "params": serializable_hmm_params,
        }

        hmm_observation_mode = getattr(inst, "hmm_observation_mode", None)

        if hasattr(getattr(inst, "hmm_type", None), "value"):
            cfg["hmm"]["type"] = getattr(inst, "hmm_type").value

        if isinstance(hmm_observation_mode, HMMObservationMode):
            hmm_observation_mode = hmm_observation_mode.value

        cfg["hmm"]["observation_mode"] = hmm_observation_mode

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
            "out_dir": ".",
        }

        saved = {}

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

            if not isinstance(df, pd.DataFrame):
                continue

            path = base_dir / f"{name}.feather"

            MethylSegConfig._write_feather_table(df, path)

            saved[name] = cls._relativize_artifact_path(base_dir, path)

        cfg["saved_tables"] = saved

        if getattr(inst, "train_sample_info", None) is not None:

            sample_path = base_dir / "train_sample_meth.feather"

            MethylSegConfig._write_feather_table(
                inst.train_sample_info.meth_data.reset_index(drop=True),
                sample_path,
            )

            cfg["train_sample_info"] = {
                "sample_id": inst.train_sample_info.sample_id,
                "meth_data_path": cls._relativize_artifact_path(
                    base_dir,
                    sample_path,
                ),
            }

        cfg["train_sample"] = {
            "name": getattr(inst, "train_sample_name", None),
            "file": getattr(inst, "train_sample_file", None),
        }

        models_saved = {}

        model_dir = base_dir / "models"
        model_dir.mkdir(exist_ok=True)

        if (
            include_learned
            and hasattr(inst.assigner, "model")
            and inst.assigner.model is not None
        ):

            model = inst.assigner.model

            if getattr(model, "kmeans", None) is not None:

                kmeans_path = model_dir / "kmeans.joblib"

                joblib.dump(model.kmeans, kmeans_path)

                models_saved["kmeans"] = cls._relativize_artifact_path(
                    base_dir,
                    kmeans_path,
                )

            if getattr(model, "pca", None) is not None:

                pca_path = model_dir / "pca.joblib"

                joblib.dump(model.pca, pca_path)

                models_saved["pca"] = cls._relativize_artifact_path(
                    base_dir,
                    pca_path,
                )

            if getattr(model, "scaler", None) is not None:

                scaler_path = model_dir / "scaler.joblib"

                joblib.dump(model.scaler, scaler_path)

                models_saved["scaler"] = cls._relativize_artifact_path(
                    base_dir,
                    scaler_path,
                )

            if getattr(model, "imputer", None) is not None:

                imputer_path = model_dir / "imputer.joblib"

                joblib.dump(model.imputer, imputer_path)

                models_saved["imputer"] = cls._relativize_artifact_path(
                    base_dir,
                    imputer_path,
                )

            models_saved["feature_cols"] = list(model.feature_cols)
            models_saved["n_states"] = int(model.n_states)
            models_saved["cluster_space"] = getattr(
                model,
                "cluster_space",
                None,
            )
            models_saved["n_pca"] = getattr(model, "n_pca", None)

        cfg["models"] = models_saved

        #
        # TRAINING ARTIFACTS
        #

        if (
            include_learned
        ):

            training_dir = base_dir / "training_artifacts"
            training_dir.mkdir(exist_ok=True)

            training_cfg: dict[str, object] = {}

            assigner_tables = {
                "train_meth": getattr(inst.assigner, "train_meth", None),
                "train_emission_df": getattr(
                    inst.assigner,
                    "train_emission_df",
                    None,
                ),
                "training_summary_df": getattr(
                    inst.assigner,
                    "training_summary_df",
                    None,
                ),
                "train_joint": getattr(inst.analyzer, "train_joint", None),
            }

            for table_name, table_value in assigner_tables.items():

                if isinstance(table_value, pd.DataFrame):

                    table_path = training_dir / f"{table_name}.feather"

                    MethylSegConfig._write_feather_table(
                        table_value,
                        table_path,
                    )

                    training_cfg[table_name] = cls._relativize_artifact_path(
                        base_dir,
                        table_path,
                    )

            train_labels = getattr(inst.assigner, "train_labels", None)

            if train_labels is not None:

                labels_path = training_dir / "train_labels.npy"

                numeric_labels = np.asarray(
                    MethylationStates.convert_to_numeric(np.asarray(train_labels)),
                    dtype=int,
                )

                np.save(labels_path, numeric_labels)

                training_cfg["train_labels"] = cls._relativize_artifact_path(
                    base_dir,
                    labels_path,
                )

            train_pca_scores = getattr(
                inst.assigner,
                "train_pca_scores",
                None,
            )

            if train_pca_scores is not None:

                pca_scores_path = training_dir / "train_pca_scores.npy"

                np.save(pca_scores_path, np.asarray(train_pca_scores))

                training_cfg["train_pca_scores"] = cls._relativize_artifact_path(
                    base_dir,
                    pca_scores_path,
                )

            if training_cfg:
                cfg["training_artifacts"] = training_cfg

        return cls(cfg)

    def get_state_cutoffs(self) -> dict | None:
        """
        Return the normalized rule-based cutoff configuration, if present.

        Returns
        -------
        dict or None
            State-cutoff mapping compatible with
            ``MethylStateAnalyzer.set_state_cutoffs`` or ``None`` when the
            serialized config does not include rule thresholds.
        """

        state_cutoffs = self.config.get("state_cutoffs", None)

        if state_cutoffs is None:
            return None

        if "cutoffs" in state_cutoffs and isinstance(
            state_cutoffs.get("cutoffs"), dict
        ):
            return state_cutoffs["cutoffs"]

        return state_cutoffs

    def to_yaml(self, yaml_path: str):
        """
        Write the serialized configuration dictionary to YAML.

        Parameters
        ----------
        yaml_path
            Output path for the YAML configuration file. Parent directories are
            created automatically.
        """

        yaml_path = Path(yaml_path)

        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as fh:
            yaml.safe_dump(self.config, fh, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MethylSegConfig":
        """
        Load a serialized methylseg configuration from YAML.

        Parameters
        ----------
        yaml_path
            Path to a YAML file previously written by ``to_yaml``.

        Returns
        -------
        MethylSegConfig
            Config wrapper with ``source_path`` set to the loaded YAML file.
        """

        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)

        config = cls(data)
        config.source_path = Path(yaml_path).resolve()

        return config

    def build_pathway(self, load_learned: bool = True):
        """
        Reconstruct a ``MethylSegPathway`` from the serialized config.

        Parameters
        ----------
        load_learned
            If ``True``, attempt to restore saved training artifacts, learned
            KMeans preprocessing components, and persisted tables referenced by
            the config.

        Returns
        -------
        MethylSegPathway
            Rehydrated pathway configured from the serialized YAML bundle.

        Raises
        ------
        ValueError
            If the loaded config is missing required sections or has invalid
            field shapes.
        FileNotFoundError
            If an expected serialized artifact referenced by the config is
            missing on disk.
        """

        from .helper_classes import KMeansMethylationModel
        from .methylseg_pathway import MethylSegPathway

        cfg = self.config

        if not isinstance(cfg, dict):
            raise ValueError("Serialized methylseg config must be a YAML mapping.")

        pathway_cfg = cfg.get("pathway", {})
        state_assigner_cfg = cfg.get("state_assigner", {})
        hmm_cfg = cfg.get("hmm", {})
        segmenter_cfg = cfg.get("segmenter", {})

        if not pathway_cfg:
            raise ValueError(
                "Serialized config is missing the required " "'pathway' section."
            )

        n_states = state_assigner_cfg.get("n_states", 4)

        int_low_cutoff = state_assigner_cfg.get("int_low_cutoff", 0.2)
        int_high_cutoff = state_assigner_cfg.get("int_high_cutoff", 0.7)
        high_cutoff = state_assigner_cfg.get("high_cutoff", 0.8)

        window_specs = self._normalize_window_specs(
            state_assigner_cfg.get(
                "window_specs",
                [(500_000, "500kb")],
            )
        )

        train_chroms = self._normalize_train_chroms(pathway_cfg.get("train_chroms"))

        max_cpg_per_chrom = pathway_cfg.get(
            "max_cpg_per_chrom",
            50_000,
        )

        out_dir = pathway_cfg.get("out_dir", ".")
        random_state = pathway_cfg.get("random_state", 42)

        cluster_space = pathway_cfg.get(
            "cluster_space",
            hmm_cfg.get("cluster_space", "pca"),
        )

        n_pca = pathway_cfg.get(
            "n_pca",
            hmm_cfg.get("n_pca", 5),
        )

        hmm_type = hmm_cfg.get("type", "continuous-time")

        min_region_length = pathway_cfg.get(
            "min_region_length",
            10_000,
        )

        min_region_cpgs = pathway_cfg.get(
            "min_region_cpgs",
            1,
        )

        merge_gap_bp = pathway_cfg.get(
            "merge_gap_bp",
            0,
        )

        state_assignment_method = segmenter_cfg.get(
            "state_assignment_method",
            MethylStateAssignmentMethod.DEFINITION.value,
        )

        hmm_observation_mode = hmm_cfg.get(
            "observation_mode",
            HMMObservationMode.DISCRETE_STATES.value,
        )

        yaml_dir = (
            self.source_path.parent
            if self.source_path is not None
            else Path(out_dir).resolve()
        )

        #
        # HMM PARAMS
        #

        hmm_params_cfg = hmm_cfg.get("params", {})

        loaded_hmm_params = {}

        for k, v in hmm_params_cfg.items():

            if isinstance(v, dict) and "__npy_path__" in v:

                npy_path = self._resolve_artifact_path(
                    yaml_dir,
                    v["__npy_path__"],
                    field_name=f"hmm.params.{k}",
                )

                loaded_hmm_params[k] = np.load(
                    npy_path,
                    allow_pickle=False,
                )

            else:
                loaded_hmm_params[k] = v

        #
        # TRAIN SAMPLE
        #

        train_sample_info = None

        if "train_sample_info" in cfg:

            sample_cfg = cfg["train_sample_info"]

            if not isinstance(sample_cfg, dict):
                raise ValueError("train_sample_info must be a dictionary.")

            if "sample_id" not in sample_cfg or "meth_data_path" not in sample_cfg:
                raise ValueError(
                    "train_sample_info must contain "
                    "'sample_id' and 'meth_data_path'."
                )

            meth_path = self._resolve_artifact_path(
                yaml_dir,
                sample_cfg["meth_data_path"],
                field_name="train_sample_info.meth_data_path",
            )

            meth_df = pd.read_feather(meth_path)

            train_sample_info = SampleInfo(
                sample_id=sample_cfg["sample_id"],
                meth_data=meth_df,
            )

        else:
            raise ValueError("Serialized config is missing " "'train_sample_info'.")

        train_sample_cfg = cfg.get("train_sample", {})

        train_sample_name = train_sample_cfg.get("name")
        train_sample_file = train_sample_cfg.get("file")

        #
        # BUILD INSTANCE
        #

        inst = MethylSegPathway(
            n_states=n_states,
            int_low_cutoff=int_low_cutoff,
            int_high_cutoff=int_high_cutoff,
            high_cutoff=high_cutoff,
            window_specs=window_specs,
            train_sample_info=train_sample_info,
            train_sample_file=train_sample_file,
            train_sample_name=train_sample_name,
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

        #
        # LOAD MODELS
        #

        if load_learned and "models" in cfg:

            model_cfg = cfg["models"]

            if not isinstance(model_cfg, dict):
                raise ValueError("models must be a dictionary.")

            kmeans = (
                joblib.load(
                    self._resolve_artifact_path(
                        yaml_dir,
                        model_cfg["kmeans"],
                        field_name="models.kmeans",
                    )
                )
                if "kmeans" in model_cfg
                else None
            )

            scaler = (
                joblib.load(
                    self._resolve_artifact_path(
                        yaml_dir,
                        model_cfg["scaler"],
                        field_name="models.scaler",
                    )
                )
                if "scaler" in model_cfg
                else None
            )

            imputer = (
                joblib.load(
                    self._resolve_artifact_path(
                        yaml_dir,
                        model_cfg["imputer"],
                        field_name="models.imputer",
                    )
                )
                if "imputer" in model_cfg
                else None
            )

            pca = (
                joblib.load(
                    self._resolve_artifact_path(
                        yaml_dir,
                        model_cfg["pca"],
                        field_name="models.pca",
                    )
                )
                if "pca" in model_cfg
                else None
            )

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

        #
        # LOAD TRAINING ARTIFACTS
        #

        training_cfg = cfg.get("training_artifacts", {})

        if isinstance(training_cfg, dict):

            table_fields = (
                "train_meth",
                "train_emission_df",
                "training_summary_df",
                "train_joint",
            )

            for field_name in table_fields:

                raw_path = training_cfg.get(field_name)


                if raw_path is None:
                    continue

                resolved_path = self._resolve_artifact_path(
                    yaml_dir,
                    raw_path,
                    field_name=f"training_artifacts.{field_name}",
                )

                loaded_table = pd.read_feather(resolved_path)

                loaded_table = self._restore_table_after_feather(
                    field_name,
                    loaded_table,
                )

                if field_name == "train_joint":
                    inst.analyzer.train_joint = loaded_table
                else:
                    setattr(inst.assigner, field_name, loaded_table)

            labels_path_raw = training_cfg.get("train_labels")

            if labels_path_raw is not None:

                labels_path = self._resolve_artifact_path(
                    yaml_dir,
                    labels_path_raw,
                    field_name="training_artifacts.train_labels",
                )

                train_labels = np.load(
                    labels_path,
                    allow_pickle=False,
                )

                inst.assigner.train_labels = np.array(
                    [self._coerce_state_value(value) for value in train_labels],
                    dtype=object,
                )

            train_pca_path_raw = training_cfg.get("train_pca_scores")

            if train_pca_path_raw is not None:

                train_pca_path = self._resolve_artifact_path(
                    yaml_dir,
                    train_pca_path_raw,
                    field_name="training_artifacts.train_pca_scores",
                )

                inst.assigner.train_pca_scores = np.load(
                    train_pca_path,
                    allow_pickle=False,
                )

        #
        # RESTORE STATE
        #

        inst.assigner.train_sample_info = inst.train_sample_info
        inst.assigner.train_sample = inst.train_sample_name

        if "state_cutoffs" in cfg:

            inst.analyzer.state_cutoffs = self._validate_state_cutoffs(
                cfg["state_cutoffs"]
            )

            inst.analyzer.cutoffs_set_manually = bool(
                cfg.get("state_cutoffs_set_manually", False)
            )

        inst._loaded_config = cfg

        return inst
