from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CODE_DIR / "test_hyperparameter_runs_dnmtools"
DEFAULT_DNMTOOLS_BED = Path(
    "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/"
    "20230828_tcga_methylation/analysis/11_benchmark_segmentation/"
    "08_run_all_tools/dnmtools/WGBS_colon-primary-tumor_1_meth/out/"
    "dnmtools_PMDs.bed"
)
DEFAULT_HM450K_FILE = CODE_DIR / "reference_files" / "WGBS_colon-primary-tumor_1_450k.beta"
DEFAULT_WGBS_FILE = CODE_DIR / "reference_files" / "WGBS_colon-primary-tumor_1_wgbs.tsv"
DEFAULT_WINDOW_SIZES_BP = [
    500,
    1_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
]
DEFAULT_CT_HOLDING_TIMES = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
    10_000_000,
]
DEFAULT_MIN_WINDOWS = 2
DEFAULT_MAX_WINDOWS = 4
DEFAULT_N_ITER = 64
DEFAULT_MIN_COVERAGE = 15
DEFAULT_MIN_PROBES_PER_REGION = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_FIXED_PARAMS = {
    "n_states": 4,
    "int_low_cutoff": 0.2,
    "int_high_cutoff": 0.7,
    "high_cutoff": 0.7,
    "min_probes_per_region": DEFAULT_MIN_PROBES_PER_REGION,
    "random_state": DEFAULT_RANDOM_STATE,
}
DEFAULT_CT_HMM_BASE_PARAMS = {
    "n_emissions": 4,
    "algorithm": "forward-backward",
    "max_iter": 20,
    "tol": 0.01,
}
MODE_SPECS = {
    "hm450k": {
        "sample_file": DEFAULT_HM450K_FILE,
        "resolution": "450k",
        "sample_id": "WGBS_colon-primary-tumor_1_hm450k",
        "hmm_types": ["ct", "sticky"],
    },
    "wgbs": {
        "sample_file": DEFAULT_WGBS_FILE,
        "resolution": "wgbs",
        "sample_id": "WGBS_colon-primary-tumor_1_wgbs",
        "hmm_types": ["sticky"],
    },
}
RANK_COLUMNS = [
    "bp_precision",
    "bp_fdr",
    "fragmentation_mean",
    "absorption_mean",
    "bp_recall",
    "bp_jaccard",
    "candidate_id",
]
RANK_ASCENDING = [False, True, True, True, False, False, True]
WORKER_CONTEXT: dict[str, Any] = {}


def _lazy_import_methyl_seg():
    from methyl_seg import MethylDataPrep, MethylSegPathway, MethylStateAssignmentMethod

    return MethylDataPrep, MethylSegPathway, MethylStateAssignmentMethod


def _format_window_label(size_bp: int) -> str:
    if size_bp % 1_000_000 == 0:
        return f"{size_bp // 1_000_000}Mb"
    if size_bp % 1_000 == 0:
        return f"{size_bp // 1_000}kb"
    return f"{size_bp}bp"


def _serializable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _clean_candidate_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_max_workers(requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    cpu_count = os.cpu_count() or 1
    return min(8, max(1, cpu_count - 1))


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    if not ordered:
        return []

    merged: list[tuple[int, int]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _interval_frame_to_dict(interval_df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    if interval_df.empty:
        return {}

    intervals_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, chrom_df in interval_df.groupby("chr", sort=True):
        intervals_by_chrom[str(chrom)] = _merge_intervals(
            list(chrom_df.loc[:, ["start", "end"]].itertuples(index=False, name=None))
        )
    return intervals_by_chrom


def _interval_dict_to_frame(intervals_by_chrom: dict[str, list[tuple[int, int]]]) -> pd.DataFrame:
    rows: list[tuple[str, int, int]] = []
    for chrom in sorted(intervals_by_chrom):
        for start, end in intervals_by_chrom[chrom]:
            rows.append((chrom, int(start), int(end)))
    return pd.DataFrame(rows, columns=["chr", "start", "end"])


def _total_bp(intervals_by_chrom: dict[str, list[tuple[int, int]]]) -> int:
    return int(
        sum(end - start for chrom_intervals in intervals_by_chrom.values() for start, end in chrom_intervals)
    )


def _intersection_bp(
    truth_by_chrom: dict[str, list[tuple[int, int]]],
    pred_by_chrom: dict[str, list[tuple[int, int]]],
) -> int:
    total = 0
    for chrom in sorted(set(truth_by_chrom) & set(pred_by_chrom)):
        truth_intervals = truth_by_chrom[chrom]
        pred_intervals = pred_by_chrom[chrom]
        i = 0
        j = 0
        while i < len(truth_intervals) and j < len(pred_intervals):
            truth_start, truth_end = truth_intervals[i]
            pred_start, pred_end = pred_intervals[j]
            overlap = min(truth_end, pred_end) - max(truth_start, pred_start)
            if overlap > 0:
                total += overlap
            if truth_end <= pred_end:
                i += 1
            else:
                j += 1
    return int(total)


def _overlap_counts(
    source_by_chrom: dict[str, list[tuple[int, int]]],
    target_by_chrom: dict[str, list[tuple[int, int]]],
) -> list[int]:
    counts: list[int] = []
    for chrom in sorted(source_by_chrom):
        source_intervals = source_by_chrom[chrom]
        target_intervals = target_by_chrom.get(chrom, [])
        target_index = 0
        for source_start, source_end in source_intervals:
            while target_index < len(target_intervals) and target_intervals[target_index][1] <= source_start:
                target_index += 1

            check_index = target_index
            overlap_count = 0
            while check_index < len(target_intervals) and target_intervals[check_index][0] < source_end:
                target_start, target_end = target_intervals[check_index]
                if min(source_end, target_end) > max(source_start, target_start):
                    overlap_count += 1
                check_index += 1
            counts.append(overlap_count)
    return counts


def _mean_or_nan(values: list[int]) -> float:
    return float(np.mean(values)) if values else math.nan


def _median_or_nan(values: list[int]) -> float:
    return float(np.median(values)) if values else math.nan


def _max_or_nan(values: list[int]) -> float:
    return float(np.max(values)) if values else math.nan


def _fraction_greater_than_one(values: list[int]) -> float:
    if not values:
        return math.nan
    return float(np.mean(np.array(values, dtype=np.int64) > 1))


def _score_predictions(
    truth_by_chrom: dict[str, list[tuple[int, int]]],
    pred_by_chrom: dict[str, list[tuple[int, int]]],
) -> dict[str, Any]:
    truth_bp = _total_bp(truth_by_chrom)
    pred_bp = _total_bp(pred_by_chrom)
    tp_bp = _intersection_bp(truth_by_chrom, pred_by_chrom)
    fp_bp = pred_bp - tp_bp
    fn_bp = truth_bp - tp_bp

    denom_pred = tp_bp + fp_bp
    denom_recall = tp_bp + fn_bp
    denom_jaccard = tp_bp + fp_bp + fn_bp

    truth_overlap_counts = _overlap_counts(truth_by_chrom, pred_by_chrom)
    pred_overlap_counts = _overlap_counts(pred_by_chrom, truth_by_chrom)
    recovered_truth_counts = [count for count in truth_overlap_counts if count > 0]
    matched_prediction_counts = [count for count in pred_overlap_counts if count > 0]

    metrics = {
        "truth_region_count": int(sum(len(intervals) for intervals in truth_by_chrom.values())),
        "pred_region_count": int(sum(len(intervals) for intervals in pred_by_chrom.values())),
        "truth_bp": int(truth_bp),
        "pred_bp": int(pred_bp),
        "tp_bp": int(tp_bp),
        "fp_bp": int(fp_bp),
        "fn_bp": int(fn_bp),
        "tn_bp": None,
        "bp_precision": float(tp_bp / denom_pred) if denom_pred > 0 else 0.0,
        "bp_recall": float(tp_bp / denom_recall) if denom_recall > 0 else 0.0,
        "bp_fdr": float(fp_bp / denom_pred) if denom_pred > 0 else 1.0,
        "bp_jaccard": float(tp_bp / denom_jaccard) if denom_jaccard > 0 else 0.0,
        "recovered_truth_region_count": int(len(recovered_truth_counts)),
        "matched_prediction_region_count": int(len(matched_prediction_counts)),
        "fragmentation_mean": _mean_or_nan(recovered_truth_counts),
        "fragmentation_median": _median_or_nan(recovered_truth_counts),
        "fragmentation_max": _max_or_nan(recovered_truth_counts),
        "fragmented_truth_fraction": _fraction_greater_than_one(recovered_truth_counts),
        "absorption_mean": _mean_or_nan(matched_prediction_counts),
        "absorption_median": _median_or_nan(matched_prediction_counts),
        "absorption_max": _max_or_nan(matched_prediction_counts),
        "absorbing_prediction_fraction": _fraction_greater_than_one(matched_prediction_counts),
    }
    return metrics


def _validate_metric_examples() -> None:
    fragmentation_truth = {"chr1": _merge_intervals([(1_000, 3_000)])}
    fragmentation_pred = {"chr1": _merge_intervals([(500, 1_500), (2_000, 3_500)])}
    frag_metrics = _score_predictions(fragmentation_truth, fragmentation_pred)
    assert frag_metrics["fragmentation_mean"] == 2.0
    assert frag_metrics["tp_bp"] == 1_500
    assert frag_metrics["fp_bp"] == 1_000
    assert frag_metrics["fn_bp"] == 500

    absorption_truth = {"chr1": _merge_intervals([(1_000, 3_000), (5_000, 7_000)])}
    absorption_pred = {"chr1": _merge_intervals([(500, 7_500)])}
    absorption_metrics = _score_predictions(absorption_truth, absorption_pred)
    assert absorption_metrics["absorption_mean"] == 2.0
    assert absorption_metrics["tp_bp"] == 4_000
    assert absorption_metrics["fp_bp"] == 3_000
    assert absorption_metrics["fn_bp"] == 0


def _load_truth_bed(path: Path) -> pd.DataFrame:
    truth_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chr", "start", "end"],
    )
    truth_df["chr"] = truth_df["chr"].astype(str)
    truth_df["start"] = pd.to_numeric(truth_df["start"], errors="raise").astype(np.int64)
    truth_df["end"] = pd.to_numeric(truth_df["end"], errors="raise").astype(np.int64)
    truth_df = truth_df.loc[truth_df["end"] > truth_df["start"]].copy()
    return truth_df.sort_values(["chr", "start", "end"]).reset_index(drop=True)


def _load_sample_info(sample_file: Path, sample_id: str, resolution: str):
    MethylDataPrep, _, _ = _lazy_import_methyl_seg()
    sample_info, removed = MethylDataPrep(
        meth_file=sample_file,
        sample_id=sample_id,
        resolution=resolution,
        min_coverage=DEFAULT_MIN_COVERAGE,
        remove_low_coverage_like_cpgs=True,
    ).prepare()
    return sample_info, removed


def _shared_chromosomes(
    truth_df: pd.DataFrame,
    hm450k_sample_info,
    wgbs_sample_info,
) -> list[str]:
    truth_chroms = set(truth_df["chr"].dropna().astype(str)) - {"chrM"}
    hm450k_chroms = set(hm450k_sample_info.meth_data["CpG_chrm"].dropna().astype(str)) - {"chrM"}
    wgbs_chroms = set(wgbs_sample_info.meth_data["CpG_chrm"].dropna().astype(str)) - {"chrM"}
    common = sorted(truth_chroms & hm450k_chroms & wgbs_chroms)
    if not common:
        raise ValueError("No chromosomes are shared by DNMTools truth, HM450K, and WGBS.")
    return common


def _filter_intervals_to_chromosomes(
    interval_df: pd.DataFrame,
    chromosomes: list[str],
) -> dict[str, list[tuple[int, int]]]:
    filtered = interval_df.loc[interval_df["chr"].isin(chromosomes)].copy()
    return _interval_frame_to_dict(filtered)


def _create_summary_file(
    candidate_out_dir: Path,
    sample_id: str,
    state_name: str = "PMR",
) -> Path:
    summary_dir = candidate_out_dir / "summary_files"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"segments_{state_name}.bed"

    state_paths = sorted(candidate_out_dir.glob(f"segments_*_{sample_id}_{state_name}.bed"))
    frames: list[pd.DataFrame] = []
    for state_path in state_paths:
        if not state_path.exists() or state_path.stat().st_size == 0:
            continue
        state_df = pd.read_csv(
            state_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chr", "start", "end"],
        )
        if not state_df.empty:
            frames.append(state_df)

    if frames:
        merged_df = pd.concat(frames, ignore_index=True)
        merged_df["chr"] = merged_df["chr"].astype(str)
        merged_df["start"] = pd.to_numeric(merged_df["start"], errors="raise").astype(np.int64)
        merged_df["end"] = pd.to_numeric(merged_df["end"], errors="raise").astype(np.int64)
        merged_dict = _interval_frame_to_dict(merged_df)
        merged_frame = _interval_dict_to_frame(merged_dict)
        merged_frame.to_csv(summary_path, sep="\t", header=False, index=False)
    else:
        summary_path.write_text("", encoding="utf-8")

    return summary_path


def _candidate_id(candidate: dict[str, Any]) -> str:
    label_key = "-".join(candidate["window_labels"])
    if candidate["hmm_type"] == "ct":
        return f"ct_ht{candidate['holding_time_guess']}_{label_key}"
    return f"sticky_{label_key}"


def _build_candidate_pool(mode_name: str) -> list[dict[str, Any]]:
    hmm_types = MODE_SPECS[mode_name]["hmm_types"]
    combos = []
    for n_windows in range(DEFAULT_MIN_WINDOWS, DEFAULT_MAX_WINDOWS + 1):
        combos.extend(itertools.combinations(DEFAULT_WINDOW_SIZES_BP, n_windows))

    pool: list[dict[str, Any]] = []
    for combo in combos:
        window_specs = [(int(size), _format_window_label(int(size))) for size in combo]
        window_labels = [label for _, label in window_specs]
        if "ct" in hmm_types:
            for holding_time_guess in DEFAULT_CT_HOLDING_TIMES:
                hmm_params = dict(DEFAULT_CT_HMM_BASE_PARAMS)
                hmm_params["holding_time_guess"] = int(holding_time_guess)
                candidate = {
                    "mode": mode_name,
                    "hmm_type": "ct",
                    "holding_time_guess": int(holding_time_guess),
                    "window_sizes_bp": [int(size) for size in combo],
                    "window_specs": window_specs,
                    "window_labels": window_labels,
                    "hmm_params": hmm_params,
                }
                candidate["candidate_id"] = _candidate_id(candidate)
                pool.append(candidate)

        if "sticky" in hmm_types:
            candidate = {
                "mode": mode_name,
                "hmm_type": "sticky",
                "holding_time_guess": None,
                "window_sizes_bp": [int(size) for size in combo],
                "window_specs": window_specs,
                "window_labels": window_labels,
                "hmm_params": {},
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
    rng = random.Random(int(random_state))

    sampled: list[dict[str, Any]] = []
    hmm_types = sorted({candidate["hmm_type"] for candidate in candidate_pool})
    if n_iter >= len(hmm_types):
        for hmm_type in hmm_types:
            hmm_candidates = [candidate for candidate in candidate_pool if candidate["hmm_type"] == hmm_type]
            sampled.append(rng.choice(hmm_candidates))

    remaining = [
        candidate
        for candidate in candidate_pool
        if candidate["candidate_id"] not in {item["candidate_id"] for item in sampled}
    ]
    if len(sampled) < n_iter:
        sampled.extend(rng.sample(remaining, k=n_iter - len(sampled)))
    return sampled


def _candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": candidate["mode"],
        "candidate_id": candidate["candidate_id"],
        "hmm_type": candidate["hmm_type"],
        "holding_time_guess": (
            candidate["holding_time_guess"] if candidate["holding_time_guess"] is not None else math.nan
        ),
        "window_count": len(candidate["window_sizes_bp"]),
        "window_sizes_bp": ",".join(str(size) for size in candidate["window_sizes_bp"]),
        "window_labels": ",".join(candidate["window_labels"]),
        "status": "failed",
        "error": "",
    }


def _write_candidate_payload(record: dict[str, Any]) -> Path:
    candidate_dir = Path(record["output_dir"])
    candidate_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "mode": record["mode"],
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
                "truth_region_count",
                "pred_region_count",
                "truth_bp",
                "pred_bp",
                "tp_bp",
                "fp_bp",
                "fn_bp",
                "tn_bp",
                "bp_precision",
                "bp_recall",
                "bp_fdr",
                "bp_jaccard",
                "recovered_truth_region_count",
                "matched_prediction_region_count",
                "fragmentation_mean",
                "fragmentation_median",
                "fragmentation_max",
                "fragmented_truth_fraction",
                "absorption_mean",
                "absorption_median",
                "absorption_max",
                "absorbing_prediction_fraction",
            )
        },
        "artifacts": {
            "output_dir": record["output_dir"],
            "pmr_bed_path": record.get("pmr_bed_path"),
        },
    }
    out_path = candidate_dir / "hyperparameters.yaml"
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return out_path


def _evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    record = _candidate_record(candidate)
    try:
        _, MethylSegPathway, MethylStateAssignmentMethod = _lazy_import_methyl_seg()
        context = WORKER_CONTEXT
        candidate_dir = _clean_candidate_dir(context["candidates_dir"] / candidate["candidate_id"])

        methyl_seg = MethylSegPathway(
            train_sample_info=context["sample_info"],
            window_specs=candidate["window_specs"],
            n_states=DEFAULT_FIXED_PARAMS["n_states"],
            int_low_cutoff=DEFAULT_FIXED_PARAMS["int_low_cutoff"],
            int_high_cutoff=DEFAULT_FIXED_PARAMS["int_high_cutoff"],
            high_cutoff=DEFAULT_FIXED_PARAMS["high_cutoff"],
            out_dir=str(candidate_dir),
            random_state=context["random_state"],
            hmm_type=candidate["hmm_type"],
            hmm_params=dict(candidate["hmm_params"]),
            state_assignment_method=MethylStateAssignmentMethod.KMEANS,
        )
        methyl_seg.fit_pathway()
        methyl_seg.run_on_all_chroms(
            sample_info=context["sample_info"],
            chroms=context["chromosomes"],
            min_probes=DEFAULT_FIXED_PARAMS["min_probes_per_region"],
            force_resegment=True,
        )
        pmr_bed_path = _create_summary_file(
            candidate_out_dir=candidate_dir,
            sample_id=context["sample_info"].sample_id,
            state_name="PMR",
        )

        if pmr_bed_path.exists() and pmr_bed_path.stat().st_size > 0:
            pred_df = pd.read_csv(
                pmr_bed_path,
                sep="\t",
                header=None,
                usecols=[0, 1, 2],
                names=["chr", "start", "end"],
            )
        else:
            pred_df = pd.DataFrame(columns=["chr", "start", "end"])

        pred_by_chrom = _interval_frame_to_dict(pred_df)
        metrics = _score_predictions(
            truth_by_chrom=context["truth_by_chrom"],
            pred_by_chrom=pred_by_chrom,
        )
        record.update(metrics)
        record["status"] = "success"
        record["output_dir"] = str(candidate_dir)
        record["pmr_bed_path"] = str(pmr_bed_path)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["output_dir"] = str(WORKER_CONTEXT["candidates_dir"] / candidate["candidate_id"])
        record["pmr_bed_path"] = str(
            WORKER_CONTEXT["candidates_dir"] / candidate["candidate_id"] / "summary_files" / "segments_PMR.bed"
        )
        for metric_name in (
            "truth_region_count",
            "pred_region_count",
            "truth_bp",
            "pred_bp",
            "tp_bp",
            "fp_bp",
            "fn_bp",
            "tn_bp",
            "bp_precision",
            "bp_recall",
            "bp_fdr",
            "bp_jaccard",
            "recovered_truth_region_count",
            "matched_prediction_region_count",
            "fragmentation_mean",
            "fragmentation_median",
            "fragmentation_max",
            "fragmented_truth_fraction",
            "absorption_mean",
            "absorption_median",
            "absorption_max",
            "absorbing_prediction_fraction",
        ):
            record.setdefault(metric_name, math.nan)

    _write_candidate_payload(record)
    return record


def _init_worker(worker_context: dict[str, Any]) -> None:
    global WORKER_CONTEXT
    WORKER_CONTEXT = worker_context


def _evaluate_candidates(
    sampled_candidates: list[dict[str, Any]],
    worker_context: dict[str, Any],
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers == 1:
        _init_worker(worker_context)
        return [
            _evaluate_candidate(candidate)
            for candidate in tqdm(sampled_candidates, desc=f"Scoring {worker_context['mode_name']}")
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
            executor.submit(_evaluate_candidate, candidate): candidate for candidate in sampled_candidates
        }
        progress = tqdm(total=len(future_map), desc=f"Scoring {worker_context['mode_name']}")
        try:
            for future in as_completed(future_map):
                results.append(future.result())
                progress.update(1)
        finally:
            progress.close()
    return results


def _rank_results(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = results_df.copy()
    ranked["rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")

    successful = ranked["status"] == "success"
    if successful.any():
        ordered = ranked.loc[successful].sort_values(
            by=RANK_COLUMNS,
            ascending=RANK_ASCENDING,
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


def _best_candidate_payload(candidate_row: pd.Series | None) -> dict[str, Any] | None:
    if candidate_row is None:
        return None
    return {
        "candidate_id": candidate_row["candidate_id"],
        "hmm_type": "cthmm" if candidate_row["hmm_type"] == "ct" else candidate_row["hmm_type"],
        "hmm_type_internal": candidate_row["hmm_type"],
        "holding_time_guess": (
            None if pd.isna(candidate_row["holding_time_guess"]) else int(candidate_row["holding_time_guess"])
        ),
        "window_sizes_bp": [int(value) for value in str(candidate_row["window_sizes_bp"]).split(",")],
        "window_specs": [
            [int(size), _format_window_label(int(size))]
            for size in str(candidate_row["window_sizes_bp"]).split(",")
        ],
        "metrics": {
            key: _serializable_scalar(candidate_row[key])
            for key in (
                "truth_region_count",
                "pred_region_count",
                "truth_bp",
                "pred_bp",
                "tp_bp",
                "fp_bp",
                "fn_bp",
                "tn_bp",
                "bp_precision",
                "bp_recall",
                "bp_fdr",
                "bp_jaccard",
                "recovered_truth_region_count",
                "matched_prediction_region_count",
                "fragmentation_mean",
                "fragmentation_median",
                "fragmentation_max",
                "fragmented_truth_fraction",
                "absorption_mean",
                "absorption_median",
                "absorption_max",
                "absorbing_prediction_fraction",
            )
        },
        "output_dir": candidate_row["output_dir"],
        "pmr_bed_path": candidate_row["pmr_bed_path"],
    }


def _write_best_hyperparameters(
    output_dir: Path,
    n_iter: int,
    random_state: int,
    common_chromosomes: list[str],
    candidate_pool_sizes: dict[str, int],
    sampled_candidate_sizes: dict[str, int],
    ranked_results_by_mode: dict[str, pd.DataFrame],
) -> Path:
    out_path = output_dir / "best_hyperparameters.yaml"
    payload = {
        "input_paths": {
            "dnmtools_truth_bed": str(DEFAULT_DNMTOOLS_BED),
            "hm450k_file": str(DEFAULT_HM450K_FILE),
            "wgbs_file": str(DEFAULT_WGBS_FILE),
        },
        "shared_chromosomes": common_chromosomes,
        "fixed_defaults": {
            "state_assignment_method": "kmeans",
            "n_states": DEFAULT_FIXED_PARAMS["n_states"],
            "int_low_cutoff": DEFAULT_FIXED_PARAMS["int_low_cutoff"],
            "int_high_cutoff": DEFAULT_FIXED_PARAMS["int_high_cutoff"],
            "high_cutoff": DEFAULT_FIXED_PARAMS["high_cutoff"],
            "min_probes_per_region": DEFAULT_FIXED_PARAMS["min_probes_per_region"],
            "min_coverage": DEFAULT_MIN_COVERAGE,
            "remove_low_coverage_like_cpgs": True,
            "random_state": random_state,
        },
        "search_spaces": {
            "hm450k": {
                "window_sizes_bp": DEFAULT_WINDOW_SIZES_BP,
                "min_windows_per_candidate": DEFAULT_MIN_WINDOWS,
                "max_windows_per_candidate": DEFAULT_MAX_WINDOWS,
                "hmm_types": ["ct", "sticky"],
                "ct_holding_time_guesses": DEFAULT_CT_HOLDING_TIMES,
                "n_iter": n_iter,
            },
            "wgbs": {
                "window_sizes_bp": DEFAULT_WINDOW_SIZES_BP,
                "min_windows_per_candidate": DEFAULT_MIN_WINDOWS,
                "max_windows_per_candidate": DEFAULT_MAX_WINDOWS,
                "hmm_types": ["sticky"],
                "ct_holding_time_guesses": [],
                "n_iter": n_iter,
            },
        },
        "metric_definitions": {
            "tp_bp": "Basepairs overlapping between DNMTools PMDs and MethylSeg PMRs.",
            "fp_bp": "Predicted PMR basepairs that do not overlap DNMTools PMDs.",
            "fn_bp": "DNMTools PMD basepairs missed by predicted PMRs.",
            "fragmentation": "For each true PMD, number of overlapping predicted PMRs.",
            "absorption": "For each predicted PMR, number of overlapping true PMDs.",
        },
        "search_summary": {
            "candidate_pool_sizes": candidate_pool_sizes,
            "sampled_candidate_sizes": sampled_candidate_sizes,
            "random_state": random_state,
        },
        "best_candidates": {},
        "artifacts": {
            "hm450k_search_results": str(output_dir / "hm450k_search_results.csv"),
            "wgbs_search_results": str(output_dir / "wgbs_search_results.csv"),
        },
    }

    for mode_name in ("hm450k", "wgbs"):
        ranked_results = ranked_results_by_mode[mode_name]
        successful = ranked_results.loc[ranked_results["status"] == "success"].copy()
        best_row = None if successful.empty else successful.sort_values("rank").iloc[0]
        payload["best_candidates"][mode_name] = _best_candidate_payload(best_row)

    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return out_path


def _run_mode_search(
    mode_name: str,
    sample_info,
    truth_by_chrom: dict[str, list[tuple[int, int]]],
    chromosomes: list[str],
    output_dir: Path,
    n_iter: int,
    max_workers: int,
    random_state: int,
) -> tuple[pd.DataFrame, int, int]:
    mode_dir = output_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = mode_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    candidate_pool = _build_candidate_pool(mode_name)
    sampled_candidates = _sample_candidates(
        candidate_pool=candidate_pool,
        n_iter=n_iter,
        random_state=random_state,
    )
    worker_context = {
        "mode_name": mode_name,
        "sample_info": sample_info,
        "truth_by_chrom": truth_by_chrom,
        "chromosomes": chromosomes,
        "candidates_dir": candidates_dir,
        "random_state": random_state,
    }
    results = _evaluate_candidates(
        sampled_candidates=sampled_candidates,
        worker_context=worker_context,
        max_workers=max_workers,
    )
    ranked_results = _rank_results(pd.DataFrame(results))
    results_path = output_dir / f"{mode_name}_search_results.csv"
    ranked_results.to_csv(results_path, index=False)
    return ranked_results, len(candidate_pool), len(sampled_candidates)


def run_search(
    output_dir: Path,
    n_iter: int,
    max_workers: int,
    random_state: int,
) -> dict[str, Any]:
    truth_df = _load_truth_bed(DEFAULT_DNMTOOLS_BED)
    hm450k_sample_info, _ = _load_sample_info(
        sample_file=DEFAULT_HM450K_FILE,
        sample_id=MODE_SPECS["hm450k"]["sample_id"],
        resolution=MODE_SPECS["hm450k"]["resolution"],
    )
    wgbs_sample_info, _ = _load_sample_info(
        sample_file=DEFAULT_WGBS_FILE,
        sample_id=MODE_SPECS["wgbs"]["sample_id"],
        resolution=MODE_SPECS["wgbs"]["resolution"],
    )
    common_chromosomes = _shared_chromosomes(
        truth_df=truth_df,
        hm450k_sample_info=hm450k_sample_info,
        wgbs_sample_info=wgbs_sample_info,
    )
    truth_by_chrom = _filter_intervals_to_chromosomes(truth_df, common_chromosomes)

    ranked_results_by_mode: dict[str, pd.DataFrame] = {}
    candidate_pool_sizes: dict[str, int] = {}
    sampled_candidate_sizes: dict[str, int] = {}
    mode_sample_infos = {
        "hm450k": hm450k_sample_info,
        "wgbs": wgbs_sample_info,
    }

    for mode_name in ("hm450k", "wgbs"):
        ranked_results, pool_size, sampled_size = _run_mode_search(
            mode_name=mode_name,
            sample_info=mode_sample_infos[mode_name],
            truth_by_chrom=truth_by_chrom,
            chromosomes=common_chromosomes,
            output_dir=output_dir,
            n_iter=n_iter,
            max_workers=max_workers,
            random_state=random_state,
        )
        ranked_results_by_mode[mode_name] = ranked_results
        candidate_pool_sizes[mode_name] = pool_size
        sampled_candidate_sizes[mode_name] = sampled_size

    best_path = _write_best_hyperparameters(
        output_dir=output_dir,
        n_iter=n_iter,
        random_state=random_state,
        common_chromosomes=common_chromosomes,
        candidate_pool_sizes=candidate_pool_sizes,
        sampled_candidate_sizes=sampled_candidate_sizes,
        ranked_results_by_mode=ranked_results_by_mode,
    )
    return {
        "output_dir": output_dir,
        "common_chromosomes": common_chromosomes,
        "best_hyperparameters": best_path,
        "hm450k_results": output_dir / "hm450k_search_results.csv",
        "wgbs_results": output_dir / "wgbs_search_results.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search MethylSeg HM450K and WGBS hyperparameters against DNMTools PMDs."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write DNMTools-based search results.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help="Number of random candidates to evaluate per mode.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers to use.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for candidate sampling.",
    )
    return parser.parse_args()


def main() -> None:
    _validate_metric_examples()
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_search(
        output_dir=output_dir,
        n_iter=int(args.n_iter),
        max_workers=_resolve_max_workers(args.max_workers),
        random_state=int(args.random_state),
    )
    print(f"Shared chromosomes: {', '.join(result['common_chromosomes'])}")
    print(f"HM450K results: {result['hm450k_results']}")
    print(f"WGBS results: {result['wgbs_results']}")
    print(f"Best hyperparameters: {result['best_hyperparameters']}")


if __name__ == "__main__":
    main()
