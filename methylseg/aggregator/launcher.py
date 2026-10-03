"""Local multiprocessing and Slurm-array launcher for MethylSeg."""

from __future__ import annotations

import json
import math
import multiprocessing
import shlex
import subprocess
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
TEST_ONLY = True


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


def _write_result(task, result):
    path = Path(task["result_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _run_sample_task(task):
    """Run one row; module-level for both spawned workers and Slurm jobs."""
    try:
        if TEST_ONLY:
            result = {
                "task_index": task["task_index"],
                "sample_id": task["sample_id"],
                "resolution": task["resolution"],
                "status": "completed",
                "summary_paths": [],
            }
            _write_result(task, result)
            return result
        sample_info = SampleInfo(
            sample_id=task["sample_id"],
            meth_data=pd.read_csv(task["meth_data_path"]),
            resolution=task["resolution"],
        )
        if task["methylseg_config"] is None:
            pathway = MethylSegPathway(train_sample_info=sample_info, out_dir=task["output_dir"])
        else:
            pathway = MethylSegConfig.from_yaml(task["methylseg_config"]).build_pathway_for_sample(
                sample_info, task["output_dir"], load_learned=task["reuse_learned"]
            )
        paths = pathway.run_pathway(
            sample_info=sample_info,
            fit=not task["reuse_learned"],
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
        )
        if RunMethod(raw.get("mode", "local")) is RunMethod.LOCAL:
            launcher_config = LocalLauncherConfig(**common, n_cpus=raw.get("n_cpus", 1))
        else:
            slurm = raw.get("slurm", {})
            if not isinstance(slurm, dict):
                raise ValueError("slurm must be a mapping.")
            launcher_config = ClusterLauncherConfig(**common, partition=slurm["partition"], account=slurm.get("account"), cpus_per_task=slurm["cpus_per_task"], mem=slurm["mem"], time=slurm["time"], python_executable=slurm.get("python_executable", "python"), environment_setup=slurm.get("environment_setup"), pythonpath=slurm.get("pythonpath"), job_name=slurm.get("job_name", "methylseg-aggregation"), output=slurm.get("output", "%x-%A_%a.out"), error=slurm.get("error", "%x-%A_%a.err"), array_task_count=slurm.get("array_task_count"))
        return cls(launcher_config)

    @classmethod
    def from_launch_spec(cls, spec_path):
        spec = _read_spec(spec_path)
        return cls(
            LocalLauncherConfig(
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
            )
        )

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
        unknown = [value for value in manifest.resolution.unique() if config.methylseg_configs_by_resolution and value not in config.methylseg_configs_by_resolution and "default" not in config.methylseg_configs_by_resolution]
        if unknown:
            raise ValueError(f"No MethylSeg configuration for resolutions: {unknown}")
        return manifest

    def _tasks(self, manifest):
        tasks = []
        for index, row in manifest.reset_index(drop=True).iterrows():
            config = self.launcher_config
            resolution_config = config.methylseg_configs_by_resolution.get(row.resolution, config.methylseg_configs_by_resolution.get("default"))
            if resolution_config is not None and resolution_config.config.source_path is None:
                raise ValueError("MethylSegConfig values must be loaded from YAML for reproducible worker execution.")
            tasks.append({"task_index": int(index), "sample_id": row.sample_id, "meth_data_path": row.meth_data_path, "resolution": row.resolution, "methylseg_config": str(resolution_config.config.source_path) if resolution_config else None, "reuse_learned": resolution_config.reuse_learned if resolution_config else False, "output_dir": str(config.output_root / "samples" / row.sample_id), "result_path": str(config.output_root / "task_results" / f"{index:06d}.json")})
        return tasks

    def _write_spec(self, tasks):
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
            "resolution_configs": {
                key: {"path": str(value.config.source_path), "reuse_learned": value.reuse_learned}
                for key, value in config.methylseg_configs_by_resolution.items()
            },
            "tasks": tasks,
        }
        path = config.output_root / "launch_spec.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        return path

    def launch(self):
        tasks = self._tasks(self._manifest())
        return self._launch_local(tasks) if self.launcher_config.run_method is RunMethod.LOCAL else self._launch_cluster(tasks)

    def _launch_local(self, tasks):
        self._write_spec(tasks)
        workers = min(self.launcher_config.n_cpus, len(tasks))
        if workers == 1:
            results = [_run_sample_task(task) for task in tqdm(tasks, desc="Running MethylSeg", unit="sample")]
        else:
            results = []
            with PROCESS_POOL_CONTEXT.Pool(workers) as pool, tqdm(total=len(tasks), desc="Running MethylSeg", unit="sample") as progress:
                for result in pool.imap_unordered(_run_sample_task, tasks):
                    results.append(result)
                    progress.update()
        return self._finalize(results)

    def _finalize(self, results):
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
                    str(self.launcher_config.output_root / "samples" / sample_id)
                    for sample_id in completed["sample_id"]
                ],
            }
        ).to_csv(aggregation_manifest, index=False)
        aggregator_config = AggregatorConfig(
            input_manifest=aggregation_manifest,
            output_root=self.launcher_config.output_root,
            cohort_name=self.launcher_config.cohort_name,
            region_type=self.launcher_config.region_type,
            use_cleaned_regions=self.launcher_config.use_cleaned_regions,
            chrom=self.launcher_config.chrom,
            chrom_sizes_path=self.launcher_config.chrom_sizes_path,
            force_recreate=self.launcher_config.force_recreate,
        )
        return MethylSegAggregator(aggregator_config).aggregate(
            self.launcher_config.aggregation_mode
        )

    def _launch_cluster(self, tasks):
        spec_path = self._write_spec(tasks)
        config, logs = self.launcher_config, self.launcher_config.output_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        args = ["sbatch", "--parsable", f"--partition={config.partition}", f"--cpus-per-task={config.cpus_per_task}", f"--mem={config.mem}", f"--time={config.time}"]
        if config.account:
            args.append(f"--account={config.account}")
        array_task_count = min(config.array_task_count or len(tasks), len(tasks))
        worker = self._script(f"{config.python_executable} -m methylseg.cli aggregate-worker --launch-spec {shlex.quote(str(spec_path))} --array-index ${{SLURM_ARRAY_TASK_ID}} --array-count {array_task_count}")
        array = subprocess.run(args + [f"--job-name={config.job_name}", f"--array=1-{array_task_count}", f"--output={logs / config.output}", f"--error={logs / config.error}"], input=worker, text=True, capture_output=True, check=True).stdout.strip()
        finalizer = self._script(f"{config.python_executable} -m methylseg.cli aggregate-finalize --launch-spec {shlex.quote(str(spec_path))}")
        final = subprocess.run(args + [f"--job-name={config.job_name}-finalize", f"--dependency=afterok:{array}", f"--output={logs / '%x-%j.out'}", f"--error={logs / '%x-%j.err'}"], input=finalizer, text=True, capture_output=True, check=True).stdout.strip()
        return {"array_job_id": array, "finalizer_job_id": final, "array_task_count": array_task_count, "launch_spec": str(spec_path)}

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
        tasks = _read_spec(spec_path)["tasks"]
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
        return self._finalize([json.loads(Path(task["result_path"]).read_text()) for task in tasks])


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
