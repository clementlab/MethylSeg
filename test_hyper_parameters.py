from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pybedtools
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterSampler
from tqdm.auto import tqdm

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from methyl_seg import (
    MethylDataPrep,
    MethylSegPathway,
    MethylStateAssignmentMethod,
    MethylationStates,
    SampleInfo,
)

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CODE_DIR / "data" / "test_hyper_parameters_config.yaml"
DEFAULT_WGBS_WINDOW_SPECS = [
    # (25_000, "25kb"),
    (500_000, "500kb"),
    (1_000_000, "1Mb"),
]
STATE_F1_ORDER = [
    MethylationStates.LOW,
    MethylationStates.PMR,
    MethylationStates.INTERMEDIATE,
    MethylationStates.HIGH,
]
BASE_RANKING_COLUMNS = {
    "pmr_only": [
        "pmr_probe_recall",
        "pmr_false_discovery_rate",
        "pmr_jaccard_bp",
        "pmr_probe_f1",
        "pmr_bp_coverage",
        "candidate_id",
    ],
    "pmr_weighted_states": [
        "pmr_probe_recall",
        "pmr_false_discovery_rate",
        "pmr_jaccard_bp",
        "pmr_probe_f1",
        "weighted_state_f1",
        "pmr_bp_coverage",
        "candidate_id",
    ],
}
_WORKER_CONTEXT: dict[str, Any] = {}


def _format_window_label(size_bp: int) -> str:
    if size_bp % 1_000_000 == 0:
        return f"{size_bp // 1_000_000}Mb"
    if size_bp % 1_000 == 0:
        return f"{size_bp // 1_000}kb"
    return f"{size_bp}bp"


def _state_assignment_method_from_value(value: str | MethylStateAssignmentMethod):
    if isinstance(value, MethylStateAssignmentMethod):
        return value
    normalized = str(value).strip().lower()
    try:
        return MethylStateAssignmentMethod(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported state assignment method: {value}") from exc


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config_dir = config_path.parent
    paths_cfg = config["paths"]
    samples_cfg = config["samples"]
    reference_cfg = config["reference_model"]
    search_cfg = config["search"]
    scoring_cfg = config["scoring"]
    parallel_cfg = config["parallel"]
    runtime_cfg = config.get("runtime", {})

    for key in ("train_sample_file", "hm450K_file", "wgbs_file"):
        samples_cfg[key] = _resolve_path(config_dir, samples_cfg[key])
    paths_cfg["output_dir"] = _resolve_path(config_dir, paths_cfg["output_dir"])
    paths_cfg["clear_output_dir_on_start"] = bool(
        paths_cfg.get("clear_output_dir_on_start", False)
    )

    reference_cfg["state_assignment_method"] = _state_assignment_method_from_value(
        reference_cfg["state_assignment_method"]
    )
    reference_cfg["window_specs"] = [
        (int(size), str(label)) for size, label in reference_cfg["window_specs"]
    ]
    search_cfg["window_sizes_bp"] = [int(size) for size in search_cfg["window_sizes_bp"]]
    search_cfg["ct_holding_time_guesses"] = [
        int(value) for value in search_cfg["ct_holding_time_guesses"]
    ]
    search_cfg["min_windows_per_candidate"] = int(
        search_cfg["min_windows_per_candidate"]
    )
    search_cfg["max_windows_per_candidate"] = int(
        search_cfg["max_windows_per_candidate"]
    )
    search_cfg["n_iter"] = int(search_cfg["n_iter"])
    search_cfg["random_state"] = int(search_cfg["random_state"])
    if search_cfg["min_windows_per_candidate"] < 2:
        raise ValueError(
            "min_windows_per_candidate must be at least 2 for this search."
        )
    if search_cfg["max_windows_per_candidate"] < search_cfg["min_windows_per_candidate"]:
        raise ValueError(
            "max_windows_per_candidate must be greater than or equal to "
            "min_windows_per_candidate."
        )
    scoring_cfg["score_mode"] = str(scoring_cfg["score_mode"]).strip().lower()
    if scoring_cfg["score_mode"] not in BASE_RANKING_COLUMNS:
        raise ValueError(
            f"Unsupported score_mode: {scoring_cfg['score_mode']}. "
            f"Expected one of {sorted(BASE_RANKING_COLUMNS)}."
        )

    normalized_weights = {}
    for state in MethylationStates:
        normalized_weights[state.name] = float(scoring_cfg["state_weights"][state.name])
    scoring_cfg["state_weights"] = normalized_weights
    region_penalty_cfg = scoring_cfg.get("region_length_penalty", {})
    scoring_cfg["region_length_penalty"] = {
        "enabled": bool(region_penalty_cfg.get("enabled", False)),
        "min_region_length_bp": int(
            region_penalty_cfg.get("min_region_length_bp", 200_000)
        ),
    }

    if parallel_cfg.get("max_workers") is not None:
        parallel_cfg["max_workers"] = int(parallel_cfg["max_workers"])
    runtime_cfg["min_probes_per_region"] = int(runtime_cfg["min_probes_per_region"])
    runtime_cfg["show_progress"] = bool(runtime_cfg.get("show_progress", True))
    runtime_cfg["chromosomes"] = runtime_cfg.get("chromosomes")

    config["_config_path"] = config_path
    return config


def _prepare_output_dir(output_dir: Path, clear_output_dir_on_start: bool) -> None:
    output_dir = output_dir.resolve()
    if not clear_output_dir_on_start:
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    protected_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        CODE_DIR.resolve(),
        CODE_DIR.parent.resolve(),
    }
    if output_dir in protected_paths:
        raise ValueError(
            f"Refusing to clear protected path: {output_dir}. "
            "Choose a dedicated run output directory instead."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_sample_info(config: dict[str, Any]) -> dict[str, SampleInfo]:
    samples_cfg = config["samples"]
    reference_cfg = config["reference_model"]
    min_coverage = int(reference_cfg["min_coverage"])

    train_sample_info, _ = MethylDataPrep(
        meth_file=samples_cfg["train_sample_file"],
        sample_id=samples_cfg["train_sample_name"],
        resolution="450k",
        min_coverage=min_coverage,
        remove_low_coverage_like_cpgs=False,
    ).prepare()

    hm450k_sample_info, _ = MethylDataPrep(
        meth_file=samples_cfg["hm450K_file"],
        sample_id=f"{samples_cfg['test_sample_name']}_hm450k",
        resolution="450k",
        min_coverage=min_coverage,
        remove_low_coverage_like_cpgs=True,
    ).prepare()

    wgbs_sample_info, _ = MethylDataPrep(
        meth_file=samples_cfg["wgbs_file"],
        sample_id=f"{samples_cfg['test_sample_name']}_wgbs",
        resolution="wgbs",
        min_coverage=min_coverage,
        remove_low_coverage_like_cpgs=True,
    ).prepare()

    return {
        "train_sample_info": train_sample_info,
        "hm450k_sample_info": hm450k_sample_info,
        "wgbs_sample_info": wgbs_sample_info,
    }


def _shared_chromosomes(
    hm450k_sample_info: SampleInfo,
    wgbs_sample_info: SampleInfo,
    configured_chromosomes: list[str] | None = None,
) -> list[str]:
    hm450k_chroms = set(hm450k_sample_info.meth_data["CpG_chrm"].dropna()) - {"chrM"}
    wgbs_chroms = set(wgbs_sample_info.meth_data["CpG_chrm"].dropna()) - {"chrM"}
    shared = sorted(hm450k_chroms & wgbs_chroms)
    if configured_chromosomes:
        configured = [chrom for chrom in configured_chromosomes if chrom in shared]
        if not configured:
            raise ValueError(
                "None of the requested chromosomes are present in both HM450K and WGBS data."
            )
        return configured
    return shared


def _probe_label_frame(meth_df: pd.DataFrame) -> pd.DataFrame:
    frame = meth_df.loc[:, ["CpG_chrm", "CpG_beg", "hmm_state_readable"]].copy()
    frame["state"] = frame["hmm_state_readable"].map(
        lambda value: value.name if isinstance(value, MethylationStates) else str(value)
    )
    return (
        frame.drop(columns=["hmm_state_readable"])
        .rename(columns={"CpG_beg": "probe_pos"})
        .sort_values(["CpG_chrm", "probe_pos"])
        .drop_duplicates(subset=["CpG_chrm", "probe_pos"], keep="first")
        .reset_index(drop=True)
    )


def _run_methylseg_on_each_chromosome(
    methylseg: MethylSegPathway,
    sample_info: SampleInfo,
    chromosomes: list[str],
    min_probes: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    iterator = chromosomes
    if show_progress:
        iterator = tqdm(chromosomes, desc=f"Segmenting {sample_info.sample_id}")

    methylseg.fit_pathway()
    if methylseg.state_assignment_method == MethylStateAssignmentMethod.DEFINITION:
        methylseg.analyzer.pretty_print_rules()

    probe_frames: list[pd.DataFrame] = []
    for chrom in iterator:
        methylseg.generate_regions(
            sample_info=sample_info,
            chrom=chrom,
            min_probes=min_probes,
        )
        probe_frames.append(_probe_label_frame(methylseg.segmentor.meth_data))

    if not probe_frames:
        return pd.DataFrame(columns=["CpG_chrm", "probe_pos", "state"])
    return pd.concat(probe_frames, ignore_index=True)


def _create_summary_files(output_dir: Path, sample_id: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    summary_dir = output_dir / "summary_files"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_paths: dict[str, Path] = {}
    for state in MethylationStates:
        state_paths = sorted(output_dir.glob(f"segments_*_{sample_id}_{state.name}.bed"))
        out_path = summary_dir / f"segments_{state.name}.bed"
        frames = []
        for state_path in state_paths:
            if state_path.stat().st_size == 0:
                continue
            frames.append(
                pd.read_csv(
                    state_path,
                    sep="\t",
                    header=None,
                    names=["chr", "start", "end", "state"],
                )
            )

        if frames:
            state_df = (
                pd.concat(frames, ignore_index=True)
                .sort_values(["chr", "start", "end"])
                .reset_index(drop=True)
            )
            state_df.to_csv(out_path, sep="\t", header=False, index=False)
        else:
            out_path.write_text("", encoding="utf-8")

        summary_paths[state.name] = out_path

    return summary_paths


def _window_specs_from_sizes(window_sizes_bp: tuple[int, ...]) -> list[tuple[int, str]]:
    return [(int(size), _format_window_label(int(size))) for size in window_sizes_bp]


def _candidate_id(candidate: dict[str, Any]) -> str:
    label_key = "-".join(candidate["window_labels"])
    if candidate["hmm_type"] == "ct":
        return f"ct_ht{candidate['holding_time_guess']}_{label_key}"
    return f"sticky_{label_key}"


def _build_candidate_pool(config: dict[str, Any]) -> list[dict[str, Any]]:
    search_cfg = config["search"]
    window_sizes = sorted(set(search_cfg["window_sizes_bp"]))
    combos = []
    for n_windows in range(
        search_cfg["min_windows_per_candidate"],
        search_cfg["max_windows_per_candidate"] + 1,
    ):
        combos.extend(itertools.combinations(window_sizes, n_windows))

    pool: list[dict[str, Any]] = []
    for combo in combos:
        window_specs = _window_specs_from_sizes(combo)
        window_labels = [label for _, label in window_specs]
        if "ct" in search_cfg["hmm_types"]:
            for holding_time in search_cfg["ct_holding_time_guesses"]:
                hmm_params = dict(search_cfg["ct_hmm_params"])
                hmm_params["holding_time_guess"] = int(holding_time)
                candidate = {
                    "hmm_type": "ct",
                    "holding_time_guess": int(holding_time),
                    "window_sizes_bp": [int(size) for size in combo],
                    "window_labels": window_labels,
                    "window_specs": window_specs,
                    "hmm_params": hmm_params,
                }
                candidate["candidate_id"] = _candidate_id(candidate)
                pool.append(candidate)

        if "sticky" in search_cfg["hmm_types"]:
            candidate = {
                "hmm_type": "sticky",
                "holding_time_guess": None,
                "window_sizes_bp": [int(size) for size in combo],
                "window_labels": window_labels,
                "window_specs": window_specs,
                "hmm_params": dict(search_cfg["sticky_hmm_params"]),
            }
            candidate["candidate_id"] = _candidate_id(candidate)
            pool.append(candidate)

    return pool


def _sample_candidates(
    candidate_pool: list[dict[str, Any]],
    n_iter: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if not candidate_pool:
        raise ValueError("Candidate pool is empty.")

    n_iter = min(int(n_iter), len(candidate_pool))
    available_hmm_types = sorted({candidate["hmm_type"] for candidate in candidate_pool})
    coverage_indices: list[int] = []
    if n_iter >= len(available_hmm_types):
        for hmm_type in available_hmm_types:
            hmm_type_indices = [
                idx for idx, candidate in enumerate(candidate_pool)
                if candidate["hmm_type"] == hmm_type
            ]
            sampled_index = next(
                iter(
                    sample["candidate_index"]
                    for sample in ParameterSampler(
                        {"candidate_index": hmm_type_indices},
                        n_iter=1,
                        random_state=random_state + len(coverage_indices),
                    )
                )
            )
            coverage_indices.append(sampled_index)

    remaining_n_iter = n_iter - len(coverage_indices)
    remaining_indices = [
        idx for idx in range(len(candidate_pool))
        if idx not in set(coverage_indices)
    ]
    sampled_indices = coverage_indices.copy()
    if remaining_n_iter > 0:
        sampled_indices.extend(
            sample["candidate_index"]
            for sample in ParameterSampler(
                {"candidate_index": remaining_indices},
                n_iter=remaining_n_iter,
                random_state=random_state,
            )
        )
    return [candidate_pool[idx] for idx in sampled_indices]


def _bedtool_or_none(path: str | Path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pybedtools.BedTool(str(path))


def _total_bp(path: str | Path) -> int:
    bedtool = _bedtool_or_none(path)
    if bedtool is None:
        return 0
    total = 0
    for interval in bedtool.sort().merge():
        total += int(interval.end) - int(interval.start)
    return total


def _region_lengths_bp(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.array([], dtype=np.int64)

    region_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "state"],
    )
    if region_df.empty:
        return np.array([], dtype=np.int64)

    lengths = (
        region_df["end"].astype(np.int64) - region_df["start"].astype(np.int64)
    ).to_numpy(dtype=np.int64)
    return lengths[lengths >= 0]


def _intersection_bp(a_path: str | Path, b_path: str | Path) -> int:
    a_bt = _bedtool_or_none(a_path)
    b_bt = _bedtool_or_none(b_path)
    if a_bt is None or b_bt is None:
        return 0

    total_overlap = 0
    intersection = a_bt.sort().merge().intersect(b_bt.sort().merge(), wao=True)
    for row in intersection:
        try:
            total_overlap += int(row.fields[-1])
        except (TypeError, ValueError):
            continue
    return total_overlap


def _fit_wgbs_model(
    wgbs_sample_info: SampleInfo,
    hm450k_sample_info: SampleInfo,
    config: dict[str, Any],
    chromosomes: list[str],
) -> dict[str, Any]:
    reference_cfg = config["reference_model"]
    runtime_cfg = config["runtime"]
    reference_out_dir = config["paths"]["output_dir"] / "wgbs_reference"
    reference_out_dir.mkdir(parents=True, exist_ok=True)

    wgbs_methylseg = MethylSegPathway(
        train_sample_info=wgbs_sample_info,
        window_specs=reference_cfg["window_specs"],
        int_low_cutoff=reference_cfg["int_low_cutoff"],
        int_high_cutoff=reference_cfg["int_high_cutoff"],
        high_cutoff=reference_cfg["high_cutoff"],
        n_states=reference_cfg["n_states"],
        out_dir=str(reference_out_dir),
        random_state=reference_cfg["random_state"],
        hmm_type=reference_cfg["hmm_type"],
        hmm_params=dict(reference_cfg["hmm_params"]),
        state_assignment_method=reference_cfg["state_assignment_method"],
    )

    wgbs_probe_labels = _run_methylseg_on_each_chromosome(
        methylseg=wgbs_methylseg,
        sample_info=wgbs_sample_info,
        chromosomes=chromosomes,
        min_probes=runtime_cfg["min_probes_per_region"],
        show_progress=runtime_cfg["show_progress"],
    )
    summary_paths = _create_summary_files(reference_out_dir, wgbs_sample_info.sample_id)

    hm450k_positions = (
        hm450k_sample_info.meth_data.loc[:, ["CpG_chrm", "CpG_beg"]]
        .drop_duplicates()
        .rename(columns={"CpG_beg": "probe_pos"})
    )
    shared_probe_labels = (
        hm450k_positions.merge(
            wgbs_probe_labels,
            on=["CpG_chrm", "probe_pos"],
            how="inner",
        )
        .rename(columns={"state": "reference_state"})
        .sort_values(["CpG_chrm", "probe_pos"])
        .reset_index(drop=True)
    )

    shared_probe_labels_path = reference_out_dir / "shared_probe_labels.tsv.gz"
    shared_probe_labels.to_csv(
        shared_probe_labels_path,
        sep="\t",
        index=False,
        compression="gzip",
    )

    return {
        "output_dir": reference_out_dir,
        "summary_paths": summary_paths,
        "pmr_bed_path": summary_paths["PMR"],
        "shared_probe_labels": shared_probe_labels,
        "shared_probe_labels_path": shared_probe_labels_path,
    }


def _fit_hm450k_model(candidate: dict[str, Any]) -> dict[str, Any]:
    context = _WORKER_CONTEXT
    reference_cfg = context["reference_model"]
    runtime_cfg = context["runtime"]
    candidate_out_dir = context["output_dir"] / "candidates" / candidate["candidate_id"]
    candidate_out_dir.mkdir(parents=True, exist_ok=True)

    hm450k_methylseg = MethylSegPathway(
        train_sample_info=context["hm450k_sample_info"],
        window_specs=candidate["window_specs"],
        int_low_cutoff=reference_cfg["int_low_cutoff"],
        int_high_cutoff=reference_cfg["int_high_cutoff"],
        high_cutoff=reference_cfg["high_cutoff"],
        n_states=reference_cfg["n_states"],
        out_dir=str(candidate_out_dir),
        random_state=reference_cfg["random_state"],
        hmm_type=candidate["hmm_type"],
        hmm_params=dict(candidate["hmm_params"]),
        state_assignment_method=reference_cfg["state_assignment_method"],
    )

    hm450k_probe_labels = _run_methylseg_on_each_chromosome(
        methylseg=hm450k_methylseg,
        sample_info=context["hm450k_sample_info"],
        chromosomes=context["chromosomes"],
        min_probes=runtime_cfg["min_probes_per_region"],
        show_progress=False,
    )
    probe_labels_path = candidate_out_dir / "probe_labels.tsv.gz"
    hm450k_probe_labels.to_csv(
        probe_labels_path,
        sep="\t",
        index=False,
        compression="gzip",
    )
    summary_paths = _create_summary_files(
        candidate_out_dir,
        context["hm450k_sample_info"].sample_id,
    )

    return {
        "output_dir": candidate_out_dir,
        "summary_paths": summary_paths,
        "probe_labels": hm450k_probe_labels,
        "probe_labels_path": probe_labels_path,
    }


def compare_models(
    reference_artifacts: dict[str, Any],
    candidate_probe_labels: pd.DataFrame,
    candidate_summary_paths: dict[str, Path],
    scoring_cfg: dict[str, Any],
) -> dict[str, Any]:
    shared = reference_artifacts["shared_probe_labels"].merge(
        candidate_probe_labels.rename(columns={"state": "candidate_state"}),
        on=["CpG_chrm", "probe_pos"],
        how="inner",
    )

    pmr_intersection_bp = _intersection_bp(
        reference_artifacts["pmr_bed_path"],
        candidate_summary_paths["PMR"],
    )
    reference_pmr_bp = _total_bp(reference_artifacts["pmr_bed_path"])
    candidate_pmr_bp = _total_bp(candidate_summary_paths["PMR"])
    union_bp = reference_pmr_bp + candidate_pmr_bp - pmr_intersection_bp
    pmr_jaccard_bp = (
        pmr_intersection_bp / union_bp if union_bp > 0 else math.nan
    )
    pmr_bp_coverage = (
        pmr_intersection_bp / reference_pmr_bp if reference_pmr_bp > 0 else math.nan
    )
    region_lengths = _region_lengths_bp(candidate_summary_paths["PMR"])
    short_region_threshold_bp = scoring_cfg["region_length_penalty"][
        "min_region_length_bp"
    ]
    if len(region_lengths) == 0:
        pmr_region_count = 0
        pmr_short_region_count = 0
        pmr_mean_region_bp = math.nan
        pmr_median_region_bp = math.nan
        pmr_short_region_fraction = math.nan
    else:
        pmr_region_count = int(len(region_lengths))
        pmr_short_region_count = int(np.sum(region_lengths < short_region_threshold_bp))
        pmr_mean_region_bp = float(np.mean(region_lengths))
        pmr_median_region_bp = float(np.median(region_lengths))
        pmr_short_region_fraction = pmr_short_region_count / pmr_region_count

    metrics = {
        "n_shared_probes": int(len(shared)),
        "reference_pmr_bp": int(reference_pmr_bp),
        "candidate_pmr_bp": int(candidate_pmr_bp),
        "pmr_intersection_bp": int(pmr_intersection_bp),
        "pmr_jaccard_bp": pmr_jaccard_bp,
        "pmr_bp_coverage": pmr_bp_coverage,
        "pmr_region_count": pmr_region_count,
        "pmr_mean_region_bp": pmr_mean_region_bp,
        "pmr_median_region_bp": pmr_median_region_bp,
        "pmr_short_region_count": pmr_short_region_count,
        "pmr_short_region_fraction": pmr_short_region_fraction,
        "pmr_true_positive_count": 0,
        "pmr_false_positive_count": 0,
        "pmr_false_negative_count": 0,
        "pmr_true_negative_count": 0,
        "pmr_false_positive_rate": math.nan,
        "pmr_false_discovery_rate": math.nan,
        "pmr_probe_f1": math.nan,
        "pmr_probe_precision": math.nan,
        "pmr_probe_recall": math.nan,
        "weighted_state_f1": math.nan,
    }

    for state in MethylationStates:
        metrics[f"{state.name.lower()}_f1"] = math.nan

    if shared.empty:
        return metrics

    y_true_pmr = (shared["reference_state"] == "PMR").astype(int)
    y_pred_pmr = (shared["candidate_state"] == "PMR").astype(int)
    tp = int(((y_true_pmr == 1) & (y_pred_pmr == 1)).sum())
    fp = int(((y_true_pmr == 0) & (y_pred_pmr == 1)).sum())
    fn = int(((y_true_pmr == 1) & (y_pred_pmr == 0)).sum())
    tn = int(((y_true_pmr == 0) & (y_pred_pmr == 0)).sum())
    metrics["pmr_true_positive_count"] = tp
    metrics["pmr_false_positive_count"] = fp
    metrics["pmr_false_negative_count"] = fn
    metrics["pmr_true_negative_count"] = tn
    metrics["pmr_false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["pmr_false_discovery_rate"] = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["pmr_probe_f1"] = f1_score(y_true_pmr, y_pred_pmr, zero_division=0)
    metrics["pmr_probe_precision"] = precision_score(
        y_true_pmr,
        y_pred_pmr,
        zero_division=0,
    )
    metrics["pmr_probe_recall"] = recall_score(
        y_true_pmr,
        y_pred_pmr,
        zero_division=0,
    )

    state_weights = scoring_cfg["state_weights"]
    weighted_total = 0.0
    weight_sum = 0.0
    for state in STATE_F1_ORDER:
        state_true = (shared["reference_state"] == state.name).astype(int)
        state_pred = (shared["candidate_state"] == state.name).astype(int)
        state_f1 = f1_score(state_true, state_pred, zero_division=0)
        metrics[f"{state.name.lower()}_f1"] = state_f1
        weight = float(state_weights[state.name])
        weighted_total += weight * state_f1
        weight_sum += weight

    if weight_sum > 0:
        metrics["weighted_state_f1"] = weighted_total / weight_sum

    return metrics


def _candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "hmm_type": candidate["hmm_type"],
        "holding_time_guess": (
            candidate["holding_time_guess"]
            if candidate["holding_time_guess"] is not None
            else math.nan
        ),
        "window_count": len(candidate["window_sizes_bp"]),
        "window_sizes_bp": ",".join(str(size) for size in candidate["window_sizes_bp"]),
        "window_labels": ",".join(candidate["window_labels"]),
        "window_specs": repr(candidate["window_specs"]),
    }


def _serializable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _candidate_summary_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": record["candidate_id"],
        "hmm_type": "cthmm" if record["hmm_type"] == "ct" else record["hmm_type"],
        "hmm_type_internal": record["hmm_type"],
        "holding_time_guess": _serializable_scalar(record["holding_time_guess"]),
        "window_count": int(record["window_count"]),
        "window_sizes_bp": [int(value) for value in str(record["window_sizes_bp"]).split(",")],
        "window_labels": str(record["window_labels"]).split(","),
        "status": record["status"],
        "error": record.get("error", ""),
        "metrics": {
            key: _serializable_scalar(record.get(key, math.nan))
            for key in (
                "n_shared_probes",
                "reference_pmr_bp",
                "candidate_pmr_bp",
                "pmr_intersection_bp",
                "pmr_jaccard_bp",
                "pmr_bp_coverage",
                "pmr_region_count",
                "pmr_mean_region_bp",
                "pmr_median_region_bp",
                "pmr_short_region_count",
                "pmr_short_region_fraction",
                "pmr_true_positive_count",
                "pmr_false_positive_count",
                "pmr_false_negative_count",
                "pmr_true_negative_count",
                "pmr_false_positive_rate",
                "pmr_false_discovery_rate",
                "pmr_probe_f1",
                "pmr_probe_precision",
                "pmr_probe_recall",
                "weighted_state_f1",
                "low_f1",
                "pmr_f1",
                "intermediate_f1",
                "high_f1",
            )
        },
        "artifacts": {
            "probe_labels_path": record.get("probe_labels_path"),
            "output_dir": record.get("output_dir"),
        },
    }


def _evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    record = _candidate_record(candidate)
    try:
        hm450k_artifacts = _fit_hm450k_model(candidate)
        metrics = compare_models(
            reference_artifacts=_WORKER_CONTEXT["reference_artifacts"],
            candidate_probe_labels=hm450k_artifacts["probe_labels"],
            candidate_summary_paths=hm450k_artifacts["summary_paths"],
            scoring_cfg=_WORKER_CONTEXT["scoring"],
        )
        record.update(metrics)
        record["output_dir"] = str(hm450k_artifacts["output_dir"])
        record["probe_labels_path"] = str(hm450k_artifacts["probe_labels_path"])
        record["status"] = "success"
        record["error"] = ""
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["output_dir"] = str(
            _WORKER_CONTEXT["output_dir"] / "candidates" / candidate["candidate_id"]
        )
        record["probe_labels_path"] = str(
            _WORKER_CONTEXT["output_dir"]
            / "candidates"
            / candidate["candidate_id"]
            / "probe_labels.tsv.gz"
        )
        record.setdefault("n_shared_probes", 0)
        for metric_key in (
            "reference_pmr_bp",
            "candidate_pmr_bp",
            "pmr_intersection_bp",
            "pmr_jaccard_bp",
            "pmr_bp_coverage",
            "pmr_region_count",
            "pmr_mean_region_bp",
            "pmr_median_region_bp",
            "pmr_short_region_count",
            "pmr_short_region_fraction",
            "pmr_true_positive_count",
            "pmr_false_positive_count",
            "pmr_false_negative_count",
            "pmr_true_negative_count",
            "pmr_false_positive_rate",
            "pmr_false_discovery_rate",
            "pmr_probe_f1",
            "pmr_probe_precision",
            "pmr_probe_recall",
            "weighted_state_f1",
        ):
            record.setdefault(metric_key, math.nan)
        for state in MethylationStates:
            record.setdefault(f"{state.name.lower()}_f1", math.nan)
    artifact_paths = _write_candidate_artifacts(
        record=record,
        reference_artifacts=_WORKER_CONTEXT["reference_artifacts"],
    )
    record.update(artifact_paths)
    return record


def _init_worker(worker_context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = worker_context


def _resolve_max_workers(requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    cpu_count = os.cpu_count() or 1
    return min(8, max(1, cpu_count - 1))


def _ranking_spec(scoring_cfg: dict[str, Any]) -> tuple[list[str], list[bool]]:
    sort_columns = list(BASE_RANKING_COLUMNS[scoring_cfg["score_mode"]])
    ascending = []
    lower_is_better = {
        "pmr_false_positive_rate",
        "pmr_false_discovery_rate",
        "pmr_short_region_fraction",
        "candidate_id",
    }
    for column in sort_columns:
        ascending.append(column in lower_is_better)

    if scoring_cfg["region_length_penalty"]["enabled"]:
        insert_at = sort_columns.index("pmr_bp_coverage")
        sort_columns.insert(insert_at, "pmr_short_region_fraction")
        ascending.insert(insert_at, True)

    return sort_columns, ascending


def _rank_results(results_df: pd.DataFrame, scoring_cfg: dict[str, Any]) -> pd.DataFrame:
    ranked = results_df.copy()
    ranked["rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")

    successful = ranked["status"] == "success"
    if successful.any():
        sort_columns, ascending = _ranking_spec(scoring_cfg)
        ordered = ranked.loc[successful].sort_values(
            by=sort_columns,
            ascending=ascending,
            na_position="last",
        )
        ranked.loc[ordered.index, "rank"] = np.arange(1, len(ordered) + 1)
        ranked = pd.concat(
            [
                ranked.loc[ordered.index],
                ranked.loc[~successful].sort_values("candidate_id"),
            ]
        )

    return ranked.reset_index(drop=True)


def _merge_reference_and_candidate_labels(
    reference_shared_probe_labels: pd.DataFrame,
    candidate_probe_labels_path: str | Path,
) -> pd.DataFrame:
    candidate_path = Path(candidate_probe_labels_path)
    if not candidate_path.exists():
        return pd.DataFrame(
            columns=["CpG_chrm", "probe_pos", "reference_state", "candidate_state"]
        )

    candidate_probe_labels = pd.read_csv(candidate_path, sep="\t")
    if candidate_probe_labels.empty:
        return pd.DataFrame(
            columns=["CpG_chrm", "probe_pos", "reference_state", "candidate_state"]
        )

    return reference_shared_probe_labels.merge(
        candidate_probe_labels.rename(columns={"state": "candidate_state"}),
        on=["CpG_chrm", "probe_pos"],
        how="inner",
    )


def _plot_top_candidate_pmr_f1(successful_results: pd.DataFrame, plots_dir: Path) -> Path:
    top_n = min(15, len(successful_results))
    top_results = successful_results.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, max(4, top_n * 0.45)))
    ax.barh(top_results["candidate_id"], top_results["pmr_probe_f1"], color="#2a9d8f")
    ax.set_xlabel("PMR Probe F1")
    ax.set_ylabel("Candidate")
    ax.set_title(f"Top {top_n} Candidates Ranked Best to Worst")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path = plots_dir / "top_candidates_pmr_f1.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_candidate_f1_scores(
    candidate_row: pd.Series,
    plots_dir: Path,
    file_name: str,
    title_prefix: str,
) -> Path:
    score_items = [
        ("LOW", float(candidate_row["low_f1"])),
        ("PMR", float(candidate_row["pmr_f1"])),
        ("INTERMEDIATE", float(candidate_row["intermediate_f1"])),
        ("HIGH", float(candidate_row["high_f1"])),
        ("PMR Probe", float(candidate_row["pmr_probe_f1"])),
        ("Weighted", float(candidate_row["weighted_state_f1"])),
    ]
    labels = [label for label, _ in score_items]
    values = [value for _, value in score_items]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color=["#577590", "#e76f51", "#43aa8b", "#f9c74f", "#264653", "#8d99ae"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{title_prefix}: {candidate_row['candidate_id']}")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = plots_dir / file_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    fmt: str = "d",
    cmap: str = "Blues",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_confusion_matrix_csv(
    matrix: np.ndarray,
    labels: list[str],
    out_path: Path,
) -> Path:
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(out_path)
    return out_path


def _write_top_level_candidate_plots(
    candidate_row: pd.Series,
    candidate_kind: str,
    title_prefix: str,
    reference_artifacts: dict[str, Any],
    plots_dir: Path,
) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    artifact_paths[f"{candidate_kind}_f1_scores"] = str(
        _plot_candidate_f1_scores(
            candidate_row=candidate_row,
            plots_dir=plots_dir,
            file_name=f"{candidate_kind}_f1_scores.png",
            title_prefix=f"{title_prefix} F1 Scores",
        )
    )

    merged_labels = _merge_reference_and_candidate_labels(
        reference_shared_probe_labels=reference_artifacts["shared_probe_labels"],
        candidate_probe_labels_path=candidate_row["probe_labels_path"],
    )
    if merged_labels.empty:
        return artifact_paths

    pmr_true = (merged_labels["reference_state"] == "PMR").astype(int)
    pmr_pred = (merged_labels["candidate_state"] == "PMR").astype(int)
    pmr_cm = confusion_matrix(pmr_true, pmr_pred, labels=[0, 1])
    pmr_cm_path = plots_dir / f"{candidate_kind}_pmr_confusion_matrix.png"
    _plot_confusion_matrix(
        matrix=pmr_cm,
        labels=["non-PMR", "PMR"],
        title=f"{title_prefix} PMR Confusion Matrix: {candidate_row['candidate_id']}",
        out_path=pmr_cm_path,
    )
    artifact_paths[f"{candidate_kind}_pmr_confusion_matrix"] = str(pmr_cm_path)
    artifact_paths[f"{candidate_kind}_pmr_confusion_matrix_csv"] = str(
        _write_confusion_matrix_csv(
            matrix=pmr_cm,
            labels=["non-PMR", "PMR"],
            out_path=plots_dir / f"{candidate_kind}_pmr_confusion_matrix.csv",
        )
    )

    state_labels = [state.name for state in STATE_F1_ORDER]
    state_cm = confusion_matrix(
        merged_labels["reference_state"],
        merged_labels["candidate_state"],
        labels=state_labels,
    )
    state_cm_path = plots_dir / f"{candidate_kind}_state_confusion_matrix.png"
    _plot_confusion_matrix(
        matrix=state_cm,
        labels=state_labels,
        title=f"{title_prefix} State Confusion Matrix: {candidate_row['candidate_id']}",
        out_path=state_cm_path,
    )
    artifact_paths[f"{candidate_kind}_state_confusion_matrix"] = str(state_cm_path)
    artifact_paths[f"{candidate_kind}_state_confusion_matrix_csv"] = str(
        _write_confusion_matrix_csv(
            matrix=state_cm,
            labels=state_labels,
            out_path=plots_dir / f"{candidate_kind}_state_confusion_matrix.csv",
        )
    )

    return artifact_paths


def _write_candidate_artifacts(
    record: dict[str, Any],
    reference_artifacts: dict[str, Any],
) -> dict[str, str]:
    candidate_out_dir = Path(record["output_dir"])
    candidate_out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = candidate_out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    payload = _candidate_summary_payload(record)
    artifact_paths: dict[str, str] = {}

    if record["status"] == "success":
        merged_labels = _merge_reference_and_candidate_labels(
            reference_shared_probe_labels=reference_artifacts["shared_probe_labels"],
            candidate_probe_labels_path=record["probe_labels_path"],
        )

        artifact_paths["f1_scores_plot"] = str(
            _plot_candidate_f1_scores(
                candidate_row=pd.Series(record),
                plots_dir=plots_dir,
                file_name="f1_scores.png",
                title_prefix="Candidate F1 Scores",
            )
        )

    else:
        merged_labels = pd.DataFrame()

    if not merged_labels.empty:
        pmr_true = (merged_labels["reference_state"] == "PMR").astype(int)
        pmr_pred = (merged_labels["candidate_state"] == "PMR").astype(int)
        pmr_cm = confusion_matrix(pmr_true, pmr_pred, labels=[0, 1])
        pmr_cm_path = plots_dir / "pmr_confusion_matrix.png"
        _plot_confusion_matrix(
            matrix=pmr_cm,
            labels=["non-PMR", "PMR"],
            title=f"PMR Confusion Matrix: {record['candidate_id']}",
            out_path=pmr_cm_path,
        )
        artifact_paths["pmr_confusion_matrix_plot"] = str(pmr_cm_path)
        artifact_paths["pmr_confusion_matrix_csv"] = str(
            _write_confusion_matrix_csv(
                matrix=pmr_cm,
                labels=["non-PMR", "PMR"],
                out_path=plots_dir / "pmr_confusion_matrix.csv",
            )
        )

        state_labels = [state.name for state in STATE_F1_ORDER]
        state_cm = confusion_matrix(
            merged_labels["reference_state"],
            merged_labels["candidate_state"],
            labels=state_labels,
        )
        state_cm_path = plots_dir / "state_confusion_matrix.png"
        _plot_confusion_matrix(
            matrix=state_cm,
            labels=state_labels,
            title=f"State Confusion Matrix: {record['candidate_id']}",
            out_path=state_cm_path,
        )
        artifact_paths["state_confusion_matrix_plot"] = str(state_cm_path)
        artifact_paths["state_confusion_matrix_csv"] = str(
            _write_confusion_matrix_csv(
                matrix=state_cm,
                labels=state_labels,
                out_path=plots_dir / "state_confusion_matrix.csv",
            )
        )

    payload["artifacts"].update(artifact_paths)
    hyperparameters_path = candidate_out_dir / "hyperparameters.yaml"
    with open(hyperparameters_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    artifact_paths["hyperparameters_yaml"] = str(hyperparameters_path)
    return artifact_paths


def _write_result_plots(
    ranked_results: pd.DataFrame,
    reference_artifacts: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    successful = ranked_results[ranked_results["status"] == "success"].copy()
    if successful.empty:
        return {}

    successful = successful.sort_values("rank", na_position="last").reset_index(drop=True)
    plot_paths: dict[str, str] = {}
    plot_paths["top_candidates_pmr_f1"] = str(
        _plot_top_candidate_pmr_f1(successful, plots_dir)
    )
    best_ct = successful[successful["hmm_type"] == "ct"]
    best_sticky = successful[successful["hmm_type"] == "sticky"]
    summary_candidates: list[tuple[str, str, pd.Series | None]] = [
        ("best_overall", "Best Overall Candidate", successful.iloc[0]),
        ("best_cthmm", "Best CTHMM Candidate", None if best_ct.empty else best_ct.iloc[0]),
        ("best_sticky", "Best Sticky Candidate", None if best_sticky.empty else best_sticky.iloc[0]),
    ]
    for candidate_kind, title_prefix, candidate_row in summary_candidates:
        if candidate_row is None:
            continue
        plot_paths.update(
            _write_top_level_candidate_plots(
                candidate_row=candidate_row,
                candidate_kind=candidate_kind,
                title_prefix=title_prefix,
                reference_artifacts=reference_artifacts,
                plots_dir=plots_dir,
            )
        )

    return plot_paths


def _best_candidate_payload(candidate_row: pd.Series | None) -> dict[str, Any] | None:
    if candidate_row is None:
        return None
    return {
        "candidate_id": candidate_row["candidate_id"],
        "hmm_type": "cthmm" if candidate_row["hmm_type"] == "ct" else candidate_row["hmm_type"],
        "hmm_type_internal": candidate_row["hmm_type"],
        "holding_time_guess": (
            None
            if pd.isna(candidate_row["holding_time_guess"])
            else int(candidate_row["holding_time_guess"])
        ),
        "window_sizes_bp": [
            int(value) for value in str(candidate_row["window_sizes_bp"]).split(",")
        ],
        "window_specs": [
            [int(size), _format_window_label(int(size))]
            for size in str(candidate_row["window_sizes_bp"]).split(",")
        ],
        "metrics": {
            key: _serializable_scalar(candidate_row[key])
            for key in (
                "pmr_probe_recall",
                "pmr_false_discovery_rate",
                "pmr_false_positive_rate",
                "pmr_false_positive_count",
                "pmr_false_negative_count",
                "pmr_true_positive_count",
                "pmr_true_negative_count",
                "pmr_jaccard_bp",
                "pmr_probe_f1",
                "pmr_probe_precision",
                "pmr_bp_coverage",
                "pmr_region_count",
                "pmr_mean_region_bp",
                "pmr_median_region_bp",
                "pmr_short_region_count",
                "pmr_short_region_fraction",
                "weighted_state_f1",
                "low_f1",
                "pmr_f1",
                "intermediate_f1",
                "high_f1",
                "n_shared_probes",
            )
        },
        "output_dir": candidate_row["output_dir"],
        "probe_labels_path": candidate_row.get("probe_labels_path"),
    }


def _write_best_hyperparameters(
    ranked_results: pd.DataFrame,
    config: dict[str, Any],
    sampled_candidates: list[dict[str, Any]],
    candidate_pool_size: int,
    plot_paths: dict[str, str] | None = None,
) -> Path:
    best_path = config["paths"]["output_dir"] / "best_hyperparameters.yaml"
    successful = ranked_results[ranked_results["status"] == "success"].copy()
    if successful.empty:
        payload = {
            "best_candidate": None,
            "best_candidates_by_hmm_type": {
                "cthmm": None,
                "sticky": None,
            },
            "reference_model": {
                "window_specs": [
                    [int(size), label]
                    for size, label in config["reference_model"]["window_specs"]
                ],
                "hmm_type": config["reference_model"]["hmm_type"],
                "hmm_params": config["reference_model"]["hmm_params"],
                "min_coverage": config["reference_model"]["min_coverage"],
                "state_assignment_method": config["reference_model"][
                    "state_assignment_method"
                ].value,
            },
            "search_summary": {
                "candidate_pool_size": candidate_pool_size,
                "sampled_candidates": len(sampled_candidates),
                "score_mode": config["scoring"]["score_mode"],
                "region_length_penalty": config["scoring"]["region_length_penalty"],
            },
        }
    else:
        successful = successful.sort_values("rank", na_position="last").reset_index(drop=True)
        best = successful.iloc[0]
        best_ct = successful[successful["hmm_type"] == "ct"]
        best_sticky = successful[successful["hmm_type"] == "sticky"]
        payload = {
            "best_candidate": _best_candidate_payload(best),
            "best_candidates_by_hmm_type": {
                "cthmm": _best_candidate_payload(best_ct.iloc[0]) if not best_ct.empty else None,
                "sticky": _best_candidate_payload(best_sticky.iloc[0]) if not best_sticky.empty else None,
            },
            "reference_model": {
                "window_specs": [
                    [int(size), label]
                    for size, label in config["reference_model"]["window_specs"]
                ],
                "hmm_type": config["reference_model"]["hmm_type"],
                "hmm_params": config["reference_model"]["hmm_params"],
                "min_coverage": config["reference_model"]["min_coverage"],
                "state_assignment_method": config["reference_model"][
                    "state_assignment_method"
                ].value,
            },
            "search_summary": {
                "candidate_pool_size": candidate_pool_size,
                "sampled_candidates": len(sampled_candidates),
                "score_mode": config["scoring"]["score_mode"],
                "random_state": config["search"]["random_state"],
                "n_iter": config["search"]["n_iter"],
                "region_length_penalty": config["scoring"]["region_length_penalty"],
            },
        }

    if plot_paths:
        payload["artifacts"] = plot_paths

    with open(best_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return best_path


def _evaluate_candidates(
    sampled_candidates: list[dict[str, Any]],
    worker_context: dict[str, Any],
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers == 1:
        _init_worker(worker_context)
        return [
            _evaluate_candidate(candidate)
            for candidate in tqdm(sampled_candidates, desc="Scoring candidates")
        ]

    results: list[dict[str, Any]] = []
    mp_context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(worker_context,),
    ) as executor:
        future_map = {
            executor.submit(_evaluate_candidate, candidate): candidate
            for candidate in sampled_candidates
        }
        progress = tqdm(total=len(future_map), desc="Scoring candidates")
        try:
            for future in as_completed(future_map):
                results.append(future.result())
                progress.update(1)
        finally:
            progress.close()
    return results


def run_hyperparameter_search(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = config["paths"]["output_dir"]
    _prepare_output_dir(
        output_dir=output_dir,
        clear_output_dir_on_start=config["paths"]["clear_output_dir_on_start"],
    )

    sample_infos = _load_sample_info(config)
    chromosomes = _shared_chromosomes(
        sample_infos["hm450k_sample_info"],
        sample_infos["wgbs_sample_info"],
        config["runtime"]["chromosomes"],
    )
    print(f"Using chromosomes: {', '.join(chromosomes)}")

    reference_artifacts = _fit_wgbs_model(
        wgbs_sample_info=sample_infos["wgbs_sample_info"],
        hm450k_sample_info=sample_infos["hm450k_sample_info"],
        config=config,
        chromosomes=chromosomes,
    )

    candidate_pool = _build_candidate_pool(config)
    sampled_candidates = _sample_candidates(
        candidate_pool=candidate_pool,
        n_iter=config["search"]["n_iter"],
        random_state=config["search"]["random_state"],
    )
    print(
        "Candidate pool size: "
        f"{len(candidate_pool)} | sampled candidates: {len(sampled_candidates)}"
    )

    worker_context = {
        "output_dir": output_dir,
        "chromosomes": chromosomes,
        "reference_model": config["reference_model"],
        "runtime": config["runtime"],
        "scoring": config["scoring"],
        "hm450k_sample_info": sample_infos["hm450k_sample_info"],
        "reference_artifacts": reference_artifacts,
    }
    max_workers = _resolve_max_workers(config["parallel"]["max_workers"])
    results = _evaluate_candidates(
        sampled_candidates=sampled_candidates,
        worker_context=worker_context,
        max_workers=max_workers,
    )

    results_df = pd.DataFrame(results)
    results_df = _rank_results(results_df, config["scoring"])
    results_path = output_dir / "search_results.csv"
    results_df.to_csv(results_path, index=False)
    plot_paths = _write_result_plots(
        ranked_results=results_df,
        reference_artifacts=reference_artifacts,
        output_dir=output_dir,
    )
    best_path = _write_best_hyperparameters(
        ranked_results=results_df,
        config=config,
        sampled_candidates=sampled_candidates,
        candidate_pool_size=len(candidate_pool),
        plot_paths=plot_paths,
    )

    successful = results_df[results_df["status"] == "success"]
    best_candidate_id = (
        successful.iloc[0]["candidate_id"] if not successful.empty else None
    )
    return {
        "search_results": results_path,
        "best_hyperparameters": best_path,
        "best_candidate_id": best_candidate_id,
        "candidate_pool_size": len(candidate_pool),
        "sampled_candidates": len(sampled_candidates),
        "shared_probes": int(len(reference_artifacts["shared_probe_labels"])),
        "plots": plot_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomized PMR-first hyperparameter search for HM450K methyl-seg.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    results = run_hyperparameter_search(config)
    print(f"Search results: {results['search_results']}")
    print(f"Best hyperparameters: {results['best_hyperparameters']}")
    print(f"Best candidate: {results['best_candidate_id']}")
    if results["plots"]:
        print("Saved plots:")
        for label, path in results["plots"].items():
            print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
