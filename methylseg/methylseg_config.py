"""Serialization helpers for saving and restoring methylseg workflow settings."""

from pathlib import Path

import joblib
import numpy as np
import yaml

from .helper_classes import HMMObservationMode, MethylStateAssignmentMethod


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
