# MethylSeg multisample aggregation

`methylseg.aggregator` runs one MethylSeg workflow per row in a manifest and
then calls an aggregator after every sample has finished. It supports local
spawn-based multiprocessing and a Slurm array followed by an `afterok`
finalizer.

## Inputs and outputs

The input is the CSV produced by `DataPrep.create_aggregation_manifest()`. It
must have `sample_id`, `meth_data_path`, and `resolution` columns. Paths may be
absolute or relative to the manifest. Each sample is written to
`<output_root>/samples/<sample_id>/`; worker status JSON files are written to
`<output_root>/task_results/`; successful completion produces
`<output_root>/regions_manifest.tsv` and an internal
`<output_root>/aggregation_manifest.csv` before peak aggregation writes its
cohort tracks.

Omit `methylseg_configs_by_resolution` entirely to construct each
`MethylSegPathway` with its built-in defaults for that sample's resolution.
Alternatively, provide a saved `MethylSegConfig` YAML for every used
resolution (or a `default` entry). The launcher uses its parameters (HMM,
windows, cutoffs, and region filters), but replaces the serialized training
sample with the current manifest sample. Every sample is fitted independently
by default. Set `reuse_learned: true` only for a YAML that contains learned
artifacts you want to apply without refitting. This uses
`run_pathway(..., fit=False)`; `force_resegment` remains separate cache
control for segmentation outputs.

## CLI

Run locally with a single YAML file:

```yaml
# aggregation-local.yaml
mode: local
input_manifest: aggregation/aggregation_manifest.csv
output_root: aggregation/out
cohort_name: tcga-colon
region_type: PMD
use_cleaned_regions: true
aggregation_mode: common
force_recreate: false  # rebuild cached signal and region artifacts
n_cpus: 4
```

Peak aggregation builds cohort-level state regions and a contributor track
after all samples complete:

```yaml
cohort_name: tcga-colon
region_type: PMD
use_cleaned_regions: true
chrom: null
aggregation_mode: common  # common, rare, shared, never, or signal_only
force_recreate: false
```

Peak aggregation writes `{cohort_name}.{region_type}.signal.bedgraph`, a
`{cohort_name}.{region_type}.signal_normalized.bedgraph` with each count
divided by the manifest sample count, and
`{cohort_name}.{region_type}.signal_metadata.tsv` to `output_root`. The
metadata table records the contributing sample IDs for every signal interval.
An existing count signal is reused; a missing normalized signal or metadata
file is added without replacing it. Set `force_recreate: true` to rebuild all
signal artifacts. Region BED files are similarly retained unless that flag is
set.
For all modes other than `signal_only`, it also writes
`{cohort_name}.{region_type}.{aggregation_mode}.bed`.
`never` requires `chrom_sizes_path`, a two-column whitespace-delimited file
with chromosome names and positive chromosome lengths; it writes the
chromosome-wide complement of the selected region type.

```bash
methylseg aggregate --config aggregation-local.yaml
```

For Slurm, replace the execution section with:

```yaml
mode: cluster
input_manifest: aggregation/aggregation_manifest.csv
output_root: /scratch/user/methylseg-aggregation
cohort_name: tcga-colon
region_type: PMD
aggregation_mode: common
methylseg_configs_by_resolution:
  "450k": {path: configs/hm450k.yaml}
  wgbs: {path: configs/wgbs.yaml, reuse_learned: false}
slurm:
  partition: notchpeak-guest
  account: owner-guest
  cpus_per_task: 4
  mem: 32G
  time: "12:00:00"
  python_executable: /path/to/python
  environment_setup: "source /path/to/conda.sh && conda activate methylseg"
  pythonpath: /path/to/MethylSeg
  array_task_count: 100
```

The command writes an immutable `launch_spec.json`, submits a configurable
number of indexed array tasks, and returns the array and finalizer job IDs.
Each array task processes one contiguous manifest batch; omit
`array_task_count` to retain one task per manifest row. The finalizer is
submitted with `afterok:<array-job-id>`, reads every task result, writes
`regions_manifest.tsv`, and invokes `MethylSegAggregator.aggregate()`. Peak
finalization returns `(aggregate_region_path_or_none, signal_path,
normalized_signal_path, metadata_path)`.

## Python API

Load the resolution configs from YAML so worker processes and Slurm jobs can
reconstruct them reproducibly:

```python
from methylseg import MethylSegConfig
from methylseg.aggregator import AggregationLauncher, LocalLauncherConfig

launcher_config = LocalLauncherConfig(
    input_manifest="aggregation/aggregation_manifest.csv",
    output_root="aggregation/out",
    methylseg_configs_by_resolution={
        "450k": MethylSegConfig.from_yaml("configs/hm450k.yaml"),
        "wgbs": MethylSegConfig.from_yaml("configs/wgbs.yaml"),
    },
    n_cpus=4,
)
aggregation_result = AggregationLauncher(
    launcher_config,
).launch()
```

For Slurm, use `ClusterLauncherConfig` with the same launcher inputs plus
`partition`, `account`, `cpus_per_task`, `mem`, `time`, and optional
`python_executable`, `environment_setup`, or `pythonpath`. The aggregator is
constructed only after the launcher has written its completed task manifests.
