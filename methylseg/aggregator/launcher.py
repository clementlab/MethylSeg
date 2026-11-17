"""Local multiprocessing and Slurm-array launcher for MethylSeg."""

from __future__ import annotations

import json
import hashlib
import math
import multiprocessing
import re
import shlex
import subprocess
import time as time_module
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm

from ..helper_classes import MethylationStates, SampleInfo
from ..methylseg_config import MethylSegConfig
from ..methylseg_pathway import MethylSegPathway
from .aggregator import (
    AggregationMode,
    AggregatorConfig,
    MethylSegAggregator,
)

PROCESS_POOL_CONTEXT = multiprocessing.get_context("spawn")
REQUIRED_MANIFEST_COLUMNS = {"sample_id", "meth_data_path", "resolution"}


class RunMethod(str, Enum):
    """Execution backends supported by :class:`AggregationLauncher`."""

    LOCAL = "local"
    CLUSTER = "cluster"


@dataclass(frozen=True)
class ResolutionMethylSegConfig:
    """Resolution-specific MethylSeg settings; learned-model reuse is opt-in."""

    config: MethylSegConfig
    reuse_learned: bool = False


class LauncherConfig:
    """Inputs shared by local and Slurm MethylSeg launches."""

    def __init__(
        self,
        input_manifest,
        output_root,
        methylseg_configs_by_resolution=None,
        cohort_name: str | None = None,
        region_type: MethylationStates | None = None,
        use_cleaned_regions: bool = True,
        chrom: str | None = None,
        chrom_sizes_path: str | Path | None = None,
        aggregation_mode=AggregationMode.COMMON,
        force_recreate: bool = False,
        cohort_training_sample: dict | None = None,
        cohort_training_manifest: str | Path | None = None,
        cohort_training_model: str | Path | None = None,
        cohort_training_model_dependency: str | None = None,
        cohort_model_root: str | Path | None = None,
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
        self.aggregation_mode = AggregationMode(aggregation_mode)
        self.force_recreate = bool(force_recreate)
        configured_training_sources = sum(
            value is not None
            for value in (
                cohort_training_sample,
                cohort_training_manifest,
                cohort_training_model,
            )
        )
        if configured_training_sources > 1:
            raise ValueError(
                "cohort_training_sample, cohort_training_manifest, and "
                "cohort_training_model are mutually exclusive."
            )
        self.cohort_training_sample = self._validate_training_sample(
            cohort_training_sample
        )
        self.cohort_training_manifest = self._validate_training_manifest(
            cohort_training_manifest
        )
        self.cohort_training_model_dependency = (
            None if cohort_training_model_dependency is None else str(cohort_training_model_dependency)
        )
        self.cohort_model_root = (
            None if cohort_model_root is None else Path(cohort_model_root).expanduser().resolve()
        )
        if self.cohort_training_model_dependency is not None and cohort_training_model is None:
            raise ValueError("cohort_training_model_dependency requires cohort_training_model.")
        self.cohort_training_model = (
            None
            if cohort_training_model is None
            else Path(cohort_training_model).expanduser().resolve()
        )
        if (self.cohort_training_model is not None
                and not self.cohort_training_model.is_file()
                and self.cohort_training_model_dependency is None):
            raise FileNotFoundError(
                f"Cohort training model does not exist: {self.cohort_training_model}"
            )
        self.methylseg_configs_by_resolution = {}
        for resolution, value in (methylseg_configs_by_resolution or {}).items():
            key = str(resolution).strip().lower()
            if not key:
                raise ValueError("Resolution configuration keys must be non-empty.")
            if isinstance(value, MethylSegConfig):
                value = ResolutionMethylSegConfig(value)
            if not isinstance(value, ResolutionMethylSegConfig):
                raise TypeError(
                    "Resolution configs must be MethylSegConfig or "
                    "ResolutionMethylSegConfig instances."
                )
            self.methylseg_configs_by_resolution[key] = value

    @staticmethod
    def _validate_training_sample(training_sample):
        if training_sample is None:
            return None
        if not isinstance(training_sample, dict):
            raise TypeError("cohort_training_sample must be a dictionary or None.")
        required = {"sample_id", "meth_data_path", "resolution"}
        missing = required.difference(training_sample)
        if missing:
            raise ValueError(
                "cohort_training_sample is missing keys: "
                f"{sorted(missing)}"
            )
        sample_id = training_sample["sample_id"]
        resolution = training_sample["resolution"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("cohort_training_sample.sample_id must be a non-empty string.")
        if not isinstance(resolution, str) or not resolution.strip():
            raise ValueError("cohort_training_sample.resolution must be a non-empty string.")
        meth_data_path = Path(training_sample["meth_data_path"]).expanduser().resolve()
        if not meth_data_path.is_file():
            raise FileNotFoundError(
                f"Cohort training methylation file does not exist: {meth_data_path}"
            )
        return {
            "sample_id": sample_id.strip(),
            "meth_data_path": str(meth_data_path),
            "resolution": resolution.strip().lower(),
        }

    @staticmethod
    def _validate_training_manifest(training_manifest):
        if training_manifest is None:
            return None
        path = Path(training_manifest).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cohort training manifest does not exist: {path}")
        return path


class LocalLauncherConfig(LauncherConfig):
    """Launcher settings for bounded local multiprocessing."""

    def __init__(
        self,
        input_manifest,
        output_root,
        methylseg_configs_by_resolution=None,
        n_cpus: int = 1,
        **aggregation_settings,
    ):
        super().__init__(
            input_manifest,
            output_root,
            methylseg_configs_by_resolution,
            **aggregation_settings,
        )
        if not isinstance(n_cpus, int) or isinstance(n_cpus, bool) or n_cpus < 1:
            raise ValueError("n_cpus must be a positive integer.")
        self.run_method = RunMethod.LOCAL
        self.n_cpus = n_cpus


class ClusterLauncherConfig(LauncherConfig):
    """Launcher settings for a portable Slurm array submission."""

    def __init__(
        self,
        input_manifest,
        output_root,
        partition: str,
        methylseg_configs_by_resolution=None,
        account: str | None = None,
        cpus_per_task: int = 1,
        mem: str = "4G",
        time: str = "01:00:00",
        python_executable: str = "python",
        environment_setup: str | None = None,
        pythonpath: str | None = None,
        job_name: str = "methylseg-aggregation",
        output: str = "%x-%A_%a.out",
        error: str = "%x-%A_%a.err",
        array_task_count: int | None = None,
        requeue: bool = True,
        transient_retry_limit: int = 1,
        recovery_cpus_per_task: int = 1,
        recovery_mem: str = "4G",
        recovery_time: str | None = None,
        **aggregation_settings,
    ):
        super().__init__(
            input_manifest,
            output_root,
            methylseg_configs_by_resolution,
            **aggregation_settings,
        )
        if not isinstance(partition, str) or not partition.strip():
            raise ValueError("partition must be a non-empty string.")
        if not isinstance(cpus_per_task, int) or isinstance(cpus_per_task, bool) or cpus_per_task < 1:
            raise ValueError("cpus_per_task must be a positive integer.")
        if array_task_count is not None and (
            not isinstance(array_task_count, int)
            or isinstance(array_task_count, bool)
            or array_task_count < 1
        ):
            raise ValueError("array_task_count must be a positive integer when provided.")
        self.run_method = RunMethod.CLUSTER
        self.partition = partition
        self.account = account
        self.cpus_per_task = cpus_per_task
        self.mem = mem
        self.time = time
        self.python_executable = python_executable
        self.environment_setup = environment_setup
        self.pythonpath = pythonpath
        self.job_name = job_name
        self.output = output
        self.error = error
        self.array_task_count = array_task_count
        self.requeue = bool(requeue)
        if not isinstance(transient_retry_limit, int) or transient_retry_limit < 0:
            raise ValueError("transient_retry_limit must be a non-negative integer.")
        if not isinstance(recovery_cpus_per_task, int) or recovery_cpus_per_task < 1:
            raise ValueError("recovery_cpus_per_task must be a positive integer.")
        self.transient_retry_limit = transient_retry_limit
        self.recovery_cpus_per_task = recovery_cpus_per_task
        self.recovery_mem = recovery_mem
        self.recovery_time = recovery_time or time


def _write_result(task, result):
    path = Path(task["result_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _completion_marker_path(task):
    """Return the durable completion marker for one sample output directory."""
    output_dir = Path(task["output_dir"])
    return output_dir.parent / f"{task['sample_id']}.complete"


def _cached_summary_paths(task):
    """Recover summary paths for a marked sample without running MethylSeg."""
    result_path = Path(task["result_path"])
    try:
        previous_result = json.loads(result_path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        previous_result = None

    if previous_result and previous_result.get("status") == "completed":
        paths = previous_result.get("summary_paths")
        if paths and isinstance(paths, list) and all(isinstance(path, str) for path in paths):
            resolved_paths = [str(Path(path).resolve()) for path in paths]
            if all(Path(path).is_file() for path in resolved_paths):
                return resolved_paths

    summary_dir = Path(task["output_dir"]) / "summary_files"
    return [str(path.resolve()) for path in sorted(summary_dir.glob("segments_*.bed")) if path.is_file()]


def _task_is_complete(task, use_cleaned_regions):
    marker = _completion_marker_path(task)
    if not marker.is_file():
        return False
    if task.get("fitted_model_yaml"):
        try:
            provenance = json.loads(marker.read_text())
        except (OSError, ValueError):
            return False
        if provenance != {"fitted_model_yaml": task.get("fitted_model_yaml")}:
            return False
    for input_path in (
        task.get("meth_data_path"),
        task.get("fitted_model_yaml"),
        task.get("methylseg_config"),
    ):
        if input_path and (not Path(input_path).is_file() or Path(input_path).stat().st_mtime > marker.stat().st_mtime):
            return False
    prefix = "cleaned" if use_cleaned_regions else "raw"
    region_type = task.get("region_type")
    if region_type is None:
        return True
    summary = Path(task["output_dir"]) / "summary_files" / f"segments_{prefix}_{region_type}.bed"
    return summary.is_file()


def _run_sample_task(task):
    """Run one row; module-level for both spawned workers and Slurm jobs."""
    marker_path = _completion_marker_path(task)
    force_recreate = bool(task.get("force_recreate", False))
    if not force_recreate and _task_is_complete(task, task.get("use_cleaned_regions", True)):
        result = {
            "task_index": task["task_index"],
            "sample_id": task["sample_id"],
            "resolution": task["resolution"],
            "status": "completed",
            "summary_paths": _cached_summary_paths(task),
        }
        _write_result(task, result)
        return result
    if force_recreate:
        marker_path.unlink(missing_ok=True)
    try:
        sample_info = SampleInfo(
            sample_id=task["sample_id"],
            meth_data=pd.read_csv(task["meth_data_path"]),
            resolution=task["resolution"],
        )
        if task.get("fitted_model_yaml"):
            run_config = MethylSegConfig.from_yaml(task["methylseg_config"]).config if task.get("methylseg_config") else {}
            explicit_build = run_config.get("pathway", {}).get("genome_build")
            pathway = MethylSegPathway.from_yaml(task["fitted_model_yaml"], load_learned=True,
                                                genome_build=explicit_build)
            # Cleaning is a target-run policy, not a learned clustering weight.
            # Older fitted bundles did not persist all cleaning settings.
            if task.get("methylseg_config"):
                for field in ("max_cpg_gap_bp", "merge_with_intermediate", "merge_with_intermediate_gap_bp",
                              "merge_gap_bp", "min_region_length", "min_region_cpgs",
                              "exclude_centromeres", "exclude_telomeres", "genome_build"):
                    if field in run_config.get("pathway", {}):
                        setattr(pathway, field, run_config["pathway"][field])
            pathway.set_out_dir(task["output_dir"])
            fit = False
        elif task["methylseg_config"] is None:
            pathway = MethylSegPathway(train_sample_info=sample_info, out_dir=task["output_dir"])
            fit = True
        else:
            pathway = MethylSegConfig.from_yaml(task["methylseg_config"]).build_pathway_for_sample(
                sample_info, task["output_dir"], load_learned=task["reuse_learned"]
            )
            fit = not task["reuse_learned"]
        paths = pathway.run_pathway(
            sample_info=sample_info,
            fit=fit,
        )
        paths = [str(Path(path).resolve()) for path in paths]
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise RuntimeError(f"MethylSeg did not write expected summaries: {missing}")
        result = {"task_index": task["task_index"], "sample_id": task["sample_id"], "resolution": task["resolution"], "status": "completed", "summary_paths": paths}
    except Exception as exc:
        result = {"task_index": task["task_index"], "sample_id": task["sample_id"], "resolution": task["resolution"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _write_result(task, result)
        raise
    _write_result(task, result)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    if task.get("fitted_model_yaml"):
        marker_path.write_text(json.dumps({"fitted_model_yaml": task.get("fitted_model_yaml")}, sort_keys=True) + "\n")
    else:
        marker_path.touch()
    return result


class AggregationLauncher:
    """Run samples locally or submit an array and dependent finalizer to Slurm."""

    def __init__(self, launcher_config: LauncherConfig):
        self.launcher_config = launcher_config

    @classmethod
    def from_yaml(cls, config_path):
        config_path = Path(config_path).resolve()
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Aggregation run YAML must be a mapping.")
        base = config_path.parent
        resolution_configs = {}
        for resolution, entry in raw.get("methylseg_configs_by_resolution", {}).items():
            entry = {"path": entry} if isinstance(entry, str) else entry
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError("Each methylseg_configs_by_resolution entry needs a YAML 'path'.")
            path = Path(entry["path"])
            path = (base / path).resolve() if not path.is_absolute() else path.resolve()
            resolution_configs[resolution] = ResolutionMethylSegConfig(MethylSegConfig.from_yaml(path), bool(entry.get("reuse_learned", False)))
        common = dict(
            input_manifest=_path_from(raw, "input_manifest", base),
            output_root=_path_from(raw, "output_root", base),
            methylseg_configs_by_resolution=resolution_configs,
            cohort_name=raw.get("cohort_name"),
            region_type=_region_type_from_serialized(raw.get("region_type"), "YAML"),
            use_cleaned_regions=raw.get("use_cleaned_regions", True),
            chrom=raw.get("chrom"),
            chrom_sizes_path=_optional_path_from(raw, "chrom_sizes_path", base),
            aggregation_mode=raw.get("aggregation_mode", AggregationMode.COMMON.value),
            force_recreate=raw.get("force_recreate", False),
            cohort_training_sample=raw.get("cohort_training_sample"),
            cohort_training_manifest=_optional_path_from(raw, "cohort_training_manifest", base),
            cohort_training_model=_optional_path_from(raw, "cohort_training_model", base),
            cohort_training_model_dependency=raw.get("cohort_training_model_dependency"),
            cohort_model_root=_optional_path_from(raw, "cohort_model_root", base),
        )
        if RunMethod(raw.get("mode", "local")) is RunMethod.LOCAL:
            launcher_config = LocalLauncherConfig(**common, n_cpus=raw.get("n_cpus", 1))
        else:
            slurm = raw.get("slurm", {})
            if not isinstance(slurm, dict):
                raise ValueError("slurm must be a mapping.")
            launcher_config = ClusterLauncherConfig(**common, partition=slurm["partition"], account=slurm.get("account"), cpus_per_task=slurm["cpus_per_task"], mem=slurm["mem"], time=slurm["time"], python_executable=slurm.get("python_executable", "python"), environment_setup=slurm.get("environment_setup"), pythonpath=slurm.get("pythonpath"), job_name=slurm.get("job_name", "methylseg-aggregation"), output=slurm.get("output", "%x-%A_%a.out"), error=slurm.get("error", "%x-%A_%a.err"), array_task_count=slurm.get("array_task_count"), requeue=slurm.get("requeue", True))
        return cls(launcher_config)

    @classmethod
    def from_launch_spec(cls, spec_path):
        spec = _read_spec(spec_path)
        resolution_configs = {}
        for resolution, entry in spec.get("resolution_configs", {}).items():
            resolution_configs[resolution] = ResolutionMethylSegConfig(
                MethylSegConfig.from_yaml(entry["path"]),
                bool(entry.get("reuse_learned", False)),
            )
        common = dict(
                input_manifest=spec["input_manifest"],
                output_root=spec["output_root"],
                cohort_name=spec.get("cohort_name"),
                region_type=_region_type_from_serialized(
                    spec.get("region_type"), "launch spec"
                ),
                use_cleaned_regions=spec.get("use_cleaned_regions", True),
                chrom=spec.get("chrom"),
                chrom_sizes_path=spec.get("chrom_sizes_path"),
                aggregation_mode=spec.get("aggregation_mode", AggregationMode.COMMON.value),
                force_recreate=spec.get("force_recreate", False),
                methylseg_configs_by_resolution=resolution_configs,
                cohort_training_sample=spec.get("cohort_training_sample"),
                cohort_training_manifest=spec.get("cohort_training_manifest"),
                cohort_training_model=spec.get("cohort_training_model"),
                cohort_training_model_dependency=spec.get("cohort_training_model_dependency"),
                cohort_model_root=spec.get("cohort_model_root"),
        )
        cluster = spec.get("cluster_recovery")
        if isinstance(cluster, dict):
            return cls(ClusterLauncherConfig(
                **common,
                partition=cluster["partition"], account=cluster.get("account"),
                cpus_per_task=cluster["cpus_per_task"], mem=cluster["mem"],
                time=cluster["time"], python_executable=cluster["python_executable"],
                environment_setup=cluster.get("environment_setup"), pythonpath=cluster.get("pythonpath"),
                job_name=cluster["job_name"], output=cluster["output"], error=cluster["error"],
                array_task_count=cluster.get("array_task_count"), requeue=cluster.get("requeue", True),
                transient_retry_limit=cluster.get("transient_retry_limit", 1),
                recovery_cpus_per_task=cluster.get("recovery_cpus_per_task", 1),
                recovery_mem=cluster.get("recovery_mem", "4G"), recovery_time=cluster.get("recovery_time"),
            ))
        return cls(LocalLauncherConfig(**common))

    def _manifest(self):
        config = self.launcher_config
        if not config.input_manifest.is_file():
            raise FileNotFoundError(f"Input manifest does not exist: {config.input_manifest}")
        manifest = pd.read_csv(config.input_manifest)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
        if missing:
            raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")
        if manifest.empty:
            raise ValueError("Input manifest contains no samples.")
        manifest = manifest.copy()
        manifest.sample_id = manifest.sample_id.astype(str)
        manifest.resolution = manifest.resolution.astype(str).str.lower()
        if manifest.sample_id.duplicated().any():
            raise ValueError("Input manifest has duplicate sample IDs.")
        base = config.input_manifest.parent
        manifest.meth_data_path = manifest.meth_data_path.map(lambda item: str((base / item).resolve()) if not Path(item).is_absolute() else str(Path(item).resolve()))
        absent = [path for path in manifest.meth_data_path if not Path(path).is_file()]
        if absent:
            raise FileNotFoundError(f"Manifest methylation files do not exist: {absent}")
        if "fitted_model_yaml" in manifest.columns:
            raise ValueError(
                "Input manifests must be target-only. Pass shared models through "
                "cohort_training_sample or cohort_training_model on the launcher."
            )
        unknown = [value for value in manifest.resolution.unique() if config.methylseg_configs_by_resolution and value not in config.methylseg_configs_by_resolution and "default" not in config.methylseg_configs_by_resolution]
        if unknown:
            raise ValueError(f"No MethylSeg configuration for resolutions: {unknown}")
        training_rows = self._cohort_training_rows()
        if training_rows is not None:
            resolutions = set(manifest.resolution)
            training_resolutions = set(training_rows.resolution)
            if len(training_resolutions) != 1 or resolutions != training_resolutions:
                raise ValueError(
                    "Shared cohort training requires one matching resolution for all "
                    f"targets and training samples; got targets {sorted(resolutions)} "
                    f"and training {sorted(training_resolutions)}."
                )
            resolution = next(iter(training_resolutions))
            resolution_config = config.methylseg_configs_by_resolution.get(
                resolution,
                config.methylseg_configs_by_resolution.get("default"),
            )
            if resolution_config is None or resolution_config.config.source_path is None:
                raise ValueError(
                    "Shared cohort training requires a YAML-backed MethylSeg config "
                    "for the training resolution."
                )
        return manifest

    def _cohort_training_rows(self):
        """Load the configured cohort-training source in normalized manifest form."""
        config = self.launcher_config
        if config.cohort_training_sample is not None:
            return pd.DataFrame([config.cohort_training_sample])
        path = config.cohort_training_manifest
        if path is None:
            return None
        training = pd.read_csv(path)
        if set(training.columns) != REQUIRED_MANIFEST_COLUMNS:
            raise ValueError(
                "Cohort training manifest must contain exactly "
                f"{sorted(REQUIRED_MANIFEST_COLUMNS)}; found {list(training.columns)}."
            )
        if training.empty:
            raise ValueError("Cohort training manifest contains no samples.")
        training = training.copy()
        training.sample_id = training.sample_id.astype(str)
        training.resolution = training.resolution.astype(str).str.lower()
        if training.sample_id.duplicated().any():
            raise ValueError("Cohort training manifest has duplicate sample IDs.")
        base = path.parent
        training.meth_data_path = training.meth_data_path.map(
            lambda value: str((base / value).resolve())
            if not Path(value).is_absolute()
            else str(Path(value).resolve())
        )
        absent = [item for item in training.meth_data_path if not Path(item).is_file()]
        if absent:
            raise FileNotFoundError(
                f"Cohort training methylation files do not exist: {absent}"
            )
        return training

    def _tasks(self, manifest, fitted_model_yaml=None):
        tasks = []
        for index, row in manifest.reset_index(drop=True).iterrows():
            config = self.launcher_config
            resolution_config = config.methylseg_configs_by_resolution.get(row.resolution, config.methylseg_configs_by_resolution.get("default"))
            if resolution_config is not None and resolution_config.config.source_path is None:
                raise ValueError("MethylSegConfig values must be loaded from YAML for reproducible worker execution.")
            output_dir = getattr(row, "methylseg_output_dir", None)
            if not isinstance(output_dir, str) or not output_dir:
                output_dir = config.output_root / "samples" / row.sample_id
            tasks.append({"task_index": int(index), "sample_id": row.sample_id, "meth_data_path": row.meth_data_path, "fitted_model_yaml": None if fitted_model_yaml is None else str(fitted_model_yaml), "resolution": row.resolution, "methylseg_config": str(resolution_config.config.source_path) if resolution_config else None, "reuse_learned": resolution_config.reuse_learned if resolution_config else False, "force_recreate": config.force_recreate, "use_cleaned_regions": config.use_cleaned_regions, "region_type": None if config.region_type is None else config.region_type.name, "output_dir": str(output_dir), "result_path": str(config.output_root / "task_results" / f"{index:06d}.json")})
        return tasks

    @property
    def cohort_model_yaml(self):
        """Resolved shared model path, or ``None`` for per-target fitting."""
        config = self.launcher_config
        if config.cohort_training_model is not None:
            return config.cohort_training_model
        if (
            config.cohort_training_sample is not None
            or config.cohort_training_manifest is not None
        ):
            return (config.cohort_model_root or config.output_root / "cohort_model") / "fitted_model.yaml"
        return None

    def _cohort_model_signature(self):
        """Return provenance that makes a learned cohort model reusable."""
        training_rows = self._cohort_training_rows()
        if training_rows is None:
            return None
        resolution = training_rows.iloc[0].resolution
        resolution_config = self.launcher_config.methylseg_configs_by_resolution.get(
            resolution,
            self.launcher_config.methylseg_configs_by_resolution.get("default"),
        )
        source_yaml = Path(resolution_config.config.source_path)
        return {
            "training_samples": [
                {
                    "sample_id": row.sample_id,
                    "meth_data_path": str(row.meth_data_path),
                    "meth_data_sha256": _sha256(Path(row.meth_data_path)),
                    "resolution": row.resolution,
                }
                for row in training_rows.itertuples(index=False)
            ],
            "resolution": resolution,
            "base_methylseg_yaml": str(source_yaml),
            "base_methylseg_yaml_sha256": _sha256(source_yaml),
            "project_settings": {
                "region_type": None if self.launcher_config.region_type is None else self.launcher_config.region_type.name,
                "use_cleaned_regions": self.launcher_config.use_cleaned_regions,
                "aggregation_mode": self.launcher_config.aggregation_mode.value,
            },
        }

    def _cohort_model_is_current(self):
        if (
            self.launcher_config.cohort_training_sample is None
            and self.launcher_config.cohort_training_manifest is None
        ):
            return self.cohort_model_yaml is not None and self.cohort_model_yaml.is_file()
        manifest_path = self.cohort_model_yaml.parent / "model_manifest.json"
        try:
            recorded = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            return False
        recorded_signature = recorded.get("signature")
        if isinstance(recorded_signature, dict):
            recorded_signature = {
                key: value for key, value in recorded_signature.items()
                if key != "meth_data_mtime_ns"
            }
        return (
            self.cohort_model_yaml is not None
            and self.cohort_model_yaml.is_file()
            and recorded_signature == self._cohort_model_signature()
        )

    def fit_cohort_model(self):
        """Fit and serialize the launcher-owned shared model when it is stale."""
        training_rows = self._cohort_training_rows()
        if training_rows is None:
            return self.cohort_model_yaml
        if self._cohort_model_is_current():
            return self.cohort_model_yaml
        model_root = self.cohort_model_yaml.parent
        model_root.mkdir(parents=True, exist_ok=True)
        training_samples = [
            SampleInfo(
                sample_id=row.sample_id,
                meth_data=pd.read_csv(row.meth_data_path),
                resolution=row.resolution,
            )
            for row in training_rows.itertuples(index=False)
        ]
        resolution_config = self.launcher_config.methylseg_configs_by_resolution.get(
            training_samples[0].resolution,
            self.launcher_config.methylseg_configs_by_resolution.get("default"),
        )
        pathway = MethylSegConfig.from_yaml(
            resolution_config.config.source_path
        ).build_pathway_for_samples(training_samples, model_root, load_learned=False)
        pathway.fit_pathway()
        pathway.to_yaml(
            self.cohort_model_yaml,
            include_learned=True,
            artifact_dir=model_root / "learned_artifacts",
        )
        manifest = {
            "signature": self._cohort_model_signature(),
            "fitted_model_yaml": str(self.cohort_model_yaml),
            "training_sample_ids": [sample.sample_id for sample in training_samples],
        }
        (model_root / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return self.cohort_model_yaml

    def _write_spec(self, tasks, pending=None):
        config = self.launcher_config
        config.output_root.mkdir(parents=True, exist_ok=True)
        spec = {
            "input_manifest": str(config.input_manifest),
            "output_root": str(config.output_root),
            "cohort_name": config.cohort_name,
            "region_type": (
                None if config.region_type is None else config.region_type.name
            ),
            "use_cleaned_regions": config.use_cleaned_regions,
            "chrom": config.chrom,
            "chrom_sizes_path": (
                None if config.chrom_sizes_path is None else str(config.chrom_sizes_path)
            ),
            "aggregation_mode": config.aggregation_mode.value,
            "force_recreate": config.force_recreate,
            "cohort_training_sample": config.cohort_training_sample,
            "cohort_training_manifest": (
                None
                if config.cohort_training_manifest is None
                else str(config.cohort_training_manifest)
            ),
            "cohort_training_model": (
                None if config.cohort_training_model is None else str(config.cohort_training_model)
            ),
            "cohort_training_model_dependency": config.cohort_training_model_dependency,
            "cohort_model_root": None if config.cohort_model_root is None else str(config.cohort_model_root),
            "cohort_model_yaml": (
                None if self.cohort_model_yaml is None else str(self.cohort_model_yaml)
            ),
            "resolution_configs": {
                key: {"path": str(value.config.source_path), "reuse_learned": value.reuse_learned}
                for key, value in config.methylseg_configs_by_resolution.items()
            },
            "cluster_recovery": (
                None if config.run_method is not RunMethod.CLUSTER else {
                    "partition": config.partition,
                    "account": config.account,
                    "cpus_per_task": config.cpus_per_task,
                    "mem": config.mem,
                    "time": config.time,
                    "python_executable": config.python_executable,
                    "environment_setup": config.environment_setup,
                    "pythonpath": config.pythonpath,
                    "job_name": config.job_name,
                    "output": config.output,
                    "error": config.error,
                    "array_task_count": config.array_task_count,
                    "requeue": config.requeue,
                    "transient_retry_limit": config.transient_retry_limit,
                    "recovery_cpus_per_task": config.recovery_cpus_per_task,
                    "recovery_mem": config.recovery_mem,
                    "recovery_time": config.recovery_time,
                }
            ),
            "tasks": tasks,
            "pending_task_indices": [task["task_index"] for task in (tasks if pending is None else pending)],
        }
        path = config.output_root / "launch_spec.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        return path

    def launch(self):
        manifest = self._manifest()
        # Local workers cannot share a live Python object under spawn.  Fit the
        # serializable bundle once here, then give every worker the same YAML.
        if (
            self.launcher_config.run_method is RunMethod.LOCAL
            and (
                self.launcher_config.cohort_training_sample is not None
                or self.launcher_config.cohort_training_manifest is not None
            )
        ):
            self.fit_cohort_model()
        tasks = self._tasks(manifest, self.cohort_model_yaml)
        pending = []
        for task in tasks:
            if self.launcher_config.force_recreate or not _task_is_complete(task, self.launcher_config.use_cleaned_regions):
                pending.append(task)
            else:
                _write_result(task, {"task_index": task["task_index"], "sample_id": task["sample_id"], "resolution": task["resolution"], "status": "completed", "summary_paths": _cached_summary_paths(task)})
        return self._launch_local(tasks, pending) if self.launcher_config.run_method is RunMethod.LOCAL else self._launch_cluster(tasks, pending)

    def _launch_local(self, tasks, pending):
        self._write_spec(tasks, pending)
        workers = min(self.launcher_config.n_cpus, len(pending))
        if workers <= 1:
            results = [_run_sample_task(task) for task in tqdm(pending, desc="Running MethylSeg", unit="sample")]
        else:
            results = []
            with PROCESS_POOL_CONTEXT.Pool(workers) as pool, tqdm(total=len(pending), desc="Running MethylSeg", unit="sample") as progress:
                for result in pool.imap_unordered(_run_sample_task, pending):
                    results.append(result)
                    progress.update()
        by_index = {result["task_index"]: result for result in results}
        ordered = [by_index.get(task["task_index"]) or json.loads(Path(task["result_path"]).read_text()) for task in tasks]
        return self._finalize(ordered, tasks)

    def _finalize(self, results, tasks):
        failures = [result for result in results if result["status"] != "completed"]
        if failures:
            raise RuntimeError(f"MethylSeg tasks failed: {failures}")
        regions_manifest = self.launcher_config.output_root / "regions_manifest.tsv"
        pd.DataFrame(results).sort_values("task_index").reset_index(drop=True).to_csv(
            regions_manifest, sep="\t", index=False
        )
        aggregation_manifest = self.launcher_config.output_root / "aggregation_manifest.csv"
        completed = pd.DataFrame(results).sort_values("task_index").reset_index(drop=True)
        pd.DataFrame(
            {
                "sample_id": completed["sample_id"],
                "methylseg_output_dir": [
                    str(task["output_dir"])
                    for task in sorted(tasks, key=lambda item: item["task_index"])
                ],
            }
        ).to_csv(aggregation_manifest, index=False)
        generation_path = self.launcher_config.output_root / "call_generation.json"
        generation = {task["sample_id"]: _completion_marker_path(task).stat().st_mtime_ns for task in tasks}
        try:
            previous_generation = json.loads(generation_path.read_text())
        except (OSError, ValueError):
            previous_generation = None
        aggregator_config = AggregatorConfig(
            input_manifest=aggregation_manifest,
            output_root=self.launcher_config.output_root,
            cohort_name=self.launcher_config.cohort_name,
            region_type=self.launcher_config.region_type,
            use_cleaned_regions=self.launcher_config.use_cleaned_regions,
            chrom=self.launcher_config.chrom,
            chrom_sizes_path=self.launcher_config.chrom_sizes_path,
            force_recreate=self.launcher_config.force_recreate or previous_generation != generation,
        )
        output = MethylSegAggregator(aggregator_config).aggregate(
            self.launcher_config.aggregation_mode
        )
        temporary = generation_path.with_name(f".{generation_path.name}.tmp")
        temporary.write_text(json.dumps(generation, indent=2, sort_keys=True) + "\n")
        temporary.replace(generation_path)
        return output

    @staticmethod
    def _sbatch(arguments, script):
        """Submit a shell-safe Slurm job and return its parsable job ID."""
        completed = subprocess.run(
            arguments, input=script, text=True, capture_output=True, check=True
        )
        return completed.stdout.strip().split(";", maxsplit=1)[0]

    @staticmethod
    def _write_json(path, payload):
        temporary = Path(path).with_name(f".{Path(path).name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @staticmethod
    def _slurm_states(job_id):
        """Return terminal Slurm states keyed by array task index when available."""
        try:
            completed = subprocess.run(
                ["sacct", "-n", "-P", "-j", str(job_id), "--format=JobIDRaw,State"],
                text=True, capture_output=True, check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return {}
        states = {}
        for line in completed.stdout.splitlines():
            raw, *rest = line.split("|")
            if not rest or "_" not in raw or "." in raw:
                continue
            task = raw.rsplit("_", maxsplit=1)[-1]
            if task.isdigit():
                states[int(task)] = rest[0].split()[0]
        return states

    @staticmethod
    def _wait_for_job(job_id, poll_seconds=15):
        """Wait until a Slurm job leaves squeue, without treating failure as success."""
        while True:
            try:
                completed = subprocess.run(
                    ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                    text=True, capture_output=True, check=True,
                )
            except OSError as error:
                raise RuntimeError(f"Cannot query Slurm job {job_id}.") from error
            if not completed.stdout.strip():
                return
            time_module.sleep(poll_seconds)

    @staticmethod
    def _is_transient_slurm_failure(state, stderr):
        if state in {"PREEMPTED", "NODE_FAIL"}:
            return True
        return bool(re.search(
            r"munge.*(stale file handle|authentication error)|"
            r"slurmstepd.*(connection|communication).*error|"
            r"slurm.*node.*(fail|not avail)",
            stderr or "",
            flags=re.IGNORECASE | re.DOTALL,
        ))

    def _recovery_report(self, spec_path, **values):
        path = Path(spec_path).with_name("recovery_manifest.json")
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            report = {}
        report.update(values)
        self._write_json(path, report)
        return path

    def _recovery_incomplete_tasks(self, spec, task_indices=None):
        selected = set(spec.get("pending_task_indices", ())) if task_indices is None else set(task_indices)
        return [
            task for task in spec["tasks"]
            if task["task_index"] in selected
            and not _task_is_complete(task, spec.get("use_cleaned_regions", True))
        ]

    def _submit_retry_array(self, spec_path, cluster, pending, attempt):
        self._write_json(spec_path, {
            **_read_spec(spec_path),
            "pending_task_indices": [task["task_index"] for task in pending],
            "recovery_attempt": attempt,
        })
        count = min(cluster["array_task_count"] or len(pending), len(pending))
        job_name = f"{cluster['job_name']}-retry-{attempt}"
        logs = Path(spec_path).parent / "logs"
        args = [
            "sbatch", "--parsable", f"--job-name={job_name}",
            f"--partition={cluster['partition']}",
            f"--cpus-per-task={cluster['cpus_per_task']}", f"--mem={cluster['mem']}",
            f"--time={cluster['time']}", f"--array=1-{count}",
            f"--output={logs / cluster['output']}", f"--error={logs / cluster['error']}",
        ]
        if cluster.get("account"):
            args.append(f"--account={cluster['account']}")
        if cluster.get("requeue"):
            args.append("--requeue")
        script = self._script(
            f"{cluster['python_executable']} -m methylseg.cli aggregate-worker "
            f"--launch-spec {shlex.quote(str(spec_path))} "
            f"--array-index ${{SLURM_ARRAY_TASK_ID}} --array-count {count}"
        )
        return self._sbatch(args, script), count

    def _submit_finalizer_from_spec(self, spec_path, cluster):
        logs = Path(spec_path).parent / "logs"
        args = [
            "sbatch", "--parsable", f"--job-name={cluster['job_name']}-finalize",
            f"--partition={cluster['partition']}",
            f"--cpus-per-task={cluster['cpus_per_task']}", f"--mem={cluster['mem']}",
            f"--time={cluster['time']}", f"--output={logs / '%x-%j.out'}",
            f"--error={logs / '%x-%j.err'}",
        ]
        if cluster.get("account"):
            args.append(f"--account={cluster['account']}")
        if cluster.get("requeue"):
            args.append("--requeue")
        return self._sbatch(args, self._script(
            f"{cluster['python_executable']} -m methylseg.cli aggregate-finalize "
            f"--launch-spec {shlex.quote(str(spec_path))}"
        ))

    def recover_from_spec(self, spec_path, array_job_id, array_count):
        """Recover transiently interrupted arrays, then run their finalizer."""
        spec_path = Path(spec_path)
        spec = _read_spec(spec_path)
        cluster = spec.get("cluster_recovery")
        if not isinstance(cluster, dict):
            raise RuntimeError("Launch specification has no cluster recovery settings.")
        attempt = int(spec.get("recovery_attempt", 0))
        original_pending = spec.get("pending_task_indices", list(range(len(spec["tasks"]))))
        incomplete = self._recovery_incomplete_tasks(spec, original_pending)
        failed_results = []
        for task in incomplete:
            result_path = Path(task["result_path"])
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result.get("status") == "failed":
                    failed_results.append({"sample_id": task["sample_id"], "error": result.get("error")})
        if failed_results:
            report = self._recovery_report(spec_path, array_job_id=str(array_job_id), attempt=attempt,
                                           outcome="non_transient_task_failure", failures=failed_results)
            raise RuntimeError(f"PMD recovery stopped; see {report}")
        if incomplete:
            states = self._slurm_states(array_job_id)
            positions = {index: position for position, index in enumerate(original_pending)}
            non_transient = []
            for task in incomplete:
                slot = math.floor(positions[task["task_index"]] * array_count / len(original_pending)) + 1
                error_name = cluster["error"].replace("%x", cluster["job_name"]).replace("%A", str(array_job_id)).replace("%a", str(slot))
                stderr_path = spec_path.parent / "logs" / error_name
                stderr = stderr_path.read_text(errors="replace") if stderr_path.is_file() else ""
                if not self._is_transient_slurm_failure(states.get(slot), stderr):
                    non_transient.append({"sample_id": task["sample_id"], "array_slot": slot,
                                          "state": states.get(slot), "stderr": str(stderr_path)})
            if non_transient or attempt >= int(cluster["transient_retry_limit"]):
                report = self._recovery_report(
                    spec_path, array_job_id=str(array_job_id), attempt=attempt,
                    outcome="retry_exhausted" if not non_transient else "non_transient_slurm_failure",
                    failures=non_transient,
                )
                raise RuntimeError(f"PMD recovery stopped; see {report}")
            retry_id, retry_count = self._submit_retry_array(spec_path, cluster, incomplete, attempt + 1)
            self._recovery_report(spec_path, array_job_id=str(array_job_id), retry_job_id=retry_id,
                                  retry_array_count=retry_count, attempt=attempt + 1, outcome="retry_submitted")
            self._wait_for_job(retry_id)
            spec = _read_spec(spec_path)
            incomplete = self._recovery_incomplete_tasks(spec, [task["task_index"] for task in pending])
            if incomplete:
                report = self._recovery_report(
                    spec_path, array_job_id=str(array_job_id), retry_job_id=retry_id,
                    attempt=attempt + 1, outcome="retry_exhausted",
                    failures=[{"sample_id": task["sample_id"]} for task in incomplete],
                )
                raise RuntimeError(f"PMD recovery stopped; see {report}")
        finalizer_id = self._submit_finalizer_from_spec(spec_path, cluster)
        self._recovery_report(spec_path, array_job_id=str(array_job_id), attempt=attempt,
                              finalizer_job_id=finalizer_id, outcome="finalizer_submitted")
        self._wait_for_job(finalizer_id)
        states = self._slurm_states(finalizer_id)
        if states or not (spec_path.parent / "aggregation_manifest.csv").is_file():
            report = self._recovery_report(spec_path, finalizer_job_id=finalizer_id, outcome="finalizer_failed")
            raise RuntimeError(f"PMD finalizer did not publish aggregation artifacts; see {report}")
        self._recovery_report(spec_path, finalizer_job_id=finalizer_id, outcome="completed")
        return finalizer_id

    def _launch_cluster(self, tasks, pending):
        spec_path = self._write_spec(tasks, pending)
        config, logs = self.launcher_config, self.launcher_config.output_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        args = ["sbatch", "--parsable", f"--partition={config.partition}", f"--cpus-per-task={config.cpus_per_task}", f"--mem={config.mem}", f"--time={config.time}"]
        if config.account:
            args.append(f"--account={config.account}")
        if config.requeue:
            args.append("--requeue")
        fit_job = None
        if (
            (
                config.cohort_training_sample is not None
                or config.cohort_training_manifest is not None
            )
            and not self._cohort_model_is_current()
        ):
            fit = self._script(
                f"{config.python_executable} -m methylseg.cli aggregate-fit-cohort-model "
                f"--launch-spec {shlex.quote(str(spec_path))}"
            )
            fit_job = self._sbatch(
                args + [
                    f"--job-name={config.job_name}-fit",
                    f"--output={logs / '%x-%j.out'}",
                    f"--error={logs / '%x-%j.err'}",
                ], fit,
            )
        array = None
        array_task_count = min(config.array_task_count or len(pending), len(pending))
        if pending:
            worker = self._script(f"{config.python_executable} -m methylseg.cli aggregate-worker --launch-spec {shlex.quote(str(spec_path))} --array-index ${{SLURM_ARRAY_TASK_ID}} --array-count {array_task_count}")
            array_args = args + [f"--job-name={config.job_name}", f"--array=1-{array_task_count}", f"--output={logs / config.output}", f"--error={logs / config.error}"]
            dependencies = [job for job in (
                fit_job, config.cohort_training_model_dependency
            ) if job is not None]
            if dependencies:
                array_args.append(f"--dependency=afterok:{':'.join(dependencies)}")
            array = self._sbatch(array_args, worker)
        finalizer = self._script(f"{config.python_executable} -m methylseg.cli aggregate-finalize --launch-spec {shlex.quote(str(spec_path))}")
        final_args = args + [f"--job-name={config.job_name}-finalize", f"--output={logs / '%x-%j.out'}", f"--error={logs / '%x-%j.err'}"]
        if array is not None:
            recovery = self._script(
                f"{config.python_executable} -m methylseg.cli aggregate-recover "
                f"--launch-spec {shlex.quote(str(spec_path))} --array-job-id {array} "
                f"--array-count {array_task_count}"
            )
            recovery_args = [
                "sbatch", "--parsable", f"--job-name={config.job_name}-recover",
                f"--partition={config.partition}",
                f"--cpus-per-task={config.recovery_cpus_per_task}",
                f"--mem={config.recovery_mem}", f"--time={config.recovery_time}",
                f"--output={logs / '%x-%j.out'}", f"--error={logs / '%x-%j.err'}",
                f"--dependency=afterany:{array}",
            ]
            if config.account:
                recovery_args.append(f"--account={config.account}")
            if config.requeue:
                recovery_args.append("--requeue")
            final = self._sbatch(recovery_args, recovery)
            return {"fit_job_id": fit_job, "model_dependency_job_id": config.cohort_training_model_dependency, "array_job_id": array, "finalizer_job_id": final, "array_task_count": array_task_count, "launch_spec": str(spec_path), "cohort_model_yaml": None if self.cohort_model_yaml is None else str(self.cohort_model_yaml)}
        elif fit_job is not None:
            final_args.append(f"--dependency=afterok:{fit_job}")
        final = self._sbatch(final_args, finalizer)
        return {"fit_job_id": fit_job, "model_dependency_job_id": config.cohort_training_model_dependency, "array_job_id": array, "finalizer_job_id": final, "array_task_count": array_task_count, "launch_spec": str(spec_path), "cohort_model_yaml": None if self.cohort_model_yaml is None else str(self.cohort_model_yaml)}

    def _script(self, command):
        lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
        if self.launcher_config.environment_setup:
            lines.append(self.launcher_config.environment_setup)
        if self.launcher_config.pythonpath:
            lines.append(f"export PYTHONPATH={shlex.quote(self.launcher_config.pythonpath)}:${{PYTHONPATH:-}}")
        return "\n".join(lines + [command, ""])

    def run_task_from_spec(self, spec_path, task_index):
        return _run_sample_task(_read_spec(spec_path)["tasks"][task_index])

    def run_task_batch_from_spec(self, spec_path, array_index, array_count):
        """Run the 1-based contiguous task batch assigned to one array slot."""
        spec = _read_spec(spec_path)
        tasks = [spec["tasks"][index] for index in spec.get("pending_task_indices", range(len(spec["tasks"])))]
        if array_count < 1 or array_index < 1 or array_index > array_count:
            raise ValueError(
                f"array_index must be between 1 and {array_count}; got {array_index}."
            )
        start = math.floor((array_index - 1) * len(tasks) / array_count)
        end = math.floor(array_index * len(tasks) / array_count)
        return [_run_sample_task(task) for task in tasks[start:end]]

    def finalize_from_spec(self, spec_path):
        tasks = _read_spec(spec_path)["tasks"]
        missing = [task["result_path"] for task in tasks if not Path(task["result_path"]).is_file()]
        if missing:
            raise RuntimeError(f"Missing Slurm task result files: {missing}")
        return self._finalize([json.loads(Path(task["result_path"]).read_text()) for task in tasks], tasks)

    def fit_cohort_model_from_spec(self, spec_path):
        """CLI boundary for the one Slurm model-fit job."""
        return self.fit_cohort_model()


def _region_type_from_serialized(value, source: str) -> MethylationStates | None:
    """Deserialize an enum name at a YAML or JSON boundary."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"region_type in {source} must be a state name or null.")
    try:
        return MethylationStates.from_string(value)
    except ValueError as error:
        raise ValueError(f"Unknown region_type in {source}: {value!r}.") from error


def _path_from(raw, key, base):
    if key not in raw:
        raise ValueError(f"Aggregation run YAML is missing {key!r}.")
    path = Path(raw[key])
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_path_from(raw, key, base):
    """Resolve an optional YAML path relative to its configuration file."""
    if raw.get(key) is None:
        return None
    path = Path(raw[key])
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _read_spec(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("tasks"), list):
        raise ValueError("Launch specification must contain a task list.")
    return spec
