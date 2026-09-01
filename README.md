# MethylSeg

`methylseg` is a methylation segmentation toolkit for preparing WGBS and
HM450K-style methylation tables, training methylation-state models, and
exporting raw and cleaned genomic regions.

The README covers the standard workflow. The [Sphinx documentation](https://clementlab.github.io/MethylSeg/)
contains API reference material and all example notebooks.

## Install

Install the current package from GitHub:

```bash
python -m pip install git+https://github.com/clementlab/MethylSeg.git
```

The repository also includes `environment.yml` for conda-based local setup.
Download the example reference data after installation:

```bash
methylseg download_data_files
```

## Valid Input

MethylSeg reads tab-delimited `.tsv` or `.tsv.gz` files. Use canonical column
names where possible.

WGBS input supplies methylated-read counts and total coverage; beta values are
calculated internally.

```text
CpG_chrm	CpG_beg	CpG_end	meth	coverage
chr1	    10468	10469	7       12
chr1	    10470	10471	3	    10
chr1	    10483	10484	15	    18
```

TCGA/HM450K input supplies beta values directly. The optional `probe` column is
kept as metadata.

```text
CpG_chrm	CpG_beg	CpG_end	beta	probe
chr1	    10468	10469	0.583	cg00000029
chr1	    10470	10471	0.271	cg00000108
chr1	    10483	10484	0.842	cg00000109
```

The full inputs used below are installed at `data/reference_files/` by
`methylseg download_data_files`.

## Quickstart

The default workflow trains a model on your input, segments one chromosome,
cleans the calls, and draws a cleaned PMD overlay. Use `resolution="wgbs"` for
WGBS count tables or `resolution="450k"` for TCGA/HM450K beta tables.

```python
from pathlib import Path

from methylseg import MethylSegPathway, MethylationStates
from methylseg.helper_classes import DATA_DIR

reference_dir = DATA_DIR / "reference_files"

# WGBS: replace these with your own sample name and count table.
sample_name = "WGBS_colon-primary-tumor_1"
sample_file = reference_dir / "WGBS_colon-primary-tumor_1_wgbs.tsv.gz"
resolution = "wgbs"

# TCGA/HM450K alternative:
# sample_name = "TCGA-BD-A3EP-01A"
# sample_file = reference_dir / "TCGA-BD-A3EP-01A_450k.tsv.gz"
# resolution = "450k"

sample_info, removed_df = MethylSegPathway.prepare_sample_info(
    sample_name=sample_name,
    sample_file=sample_file,
    resolution=resolution,
    min_coverage=10,
)

pathway = MethylSegPathway(
    train_sample_info=sample_info,
    out_dir=Path("methylseg_output") / sample_name,
)
pathway.fit_pathway()

regions = pathway.generate_regions(sample_info=sample_info, chrom="chr1")
_, clean_dir = pathway.get_clean_regions(
    regions_df=regions,
    sample_id=sample_info.sample_id,
    chrom="chr1",
)
fig = pathway.plot_labels(
    sample_info=sample_info,
    sample_info_removed=removed_df,
    chrom="chr1",
    overlay_state=MethylationStates.PMD,
    use_cleaned_regions=True,
    label_title="Cleaned PMD overlay",
)
```

`run_pathway(sample_info=sample_info, chroms=["chr1"])` performs fitting,
segmentation, cleaning, and summary-file writing in one call. See
`examples/run_full_pipeline.ipynb` for the full end-to-end workflow.

For fast reruns with a previously saved model, use
`MethylSegPathway.get_pretrained_model(out_dir, resolution=...)`; the detailed
workflow is in `examples/use_preloaded_model.ipynb`.

## Outputs

`generate_regions` returns one chromosome's raw region table and writes
state-specific BED files. `get_clean_regions` merges and filters calls, writes
cleaned metadata under `clean_regions/`, and returns the cleaned summary paths.
`plot_labels(use_cleaned_regions=True)` reads those cleaned artifacts, so run
the cleaning step before requesting a cleaned overlay.

## Development Checklist

- [ ] Add documentation and regression tests for public documentation discovery.
- [x] Document the full workflow, inputs, outputs, and cleaned-region behavior.
- [x] Include all maintained example notebooks in the documentation website.
- [x] Add user-facing docstrings and CLI help for currently supported commands.
- [ ] Continue legacy cleanup as separately scoped maintenance work.
- [ ] Upgrade scikitlearn from 1.8.0 to 1.9.0 and then repackage sample data
