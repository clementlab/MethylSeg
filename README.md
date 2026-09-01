# MethylSeg

`methylseg` is a methylation segmentation toolkit in this repository for:

- preparing WGBS and HM450K-style methylation tables
- learning methylation state assignments from local window summaries
- segmenting samples with several HMM backends
- exporting raw and cleaned genomic regions for downstream analysis

## Install

Install the current public version directly from GitHub:

```bash
python -m pip install git+https://github.com/clementlab/MethylSeg.git
```

or from TestPyPi

```bash
pip install -i https://test.pypi.org/simple/ methylseg==0.1.0
```

PyPI installation will be added here once the package is published there. 

The
repository also includes a conda environment file at [environment.yml](./environment.yml)
for local environment management.

## Quickstart

```python
from pathlib import Path

from methylseg import MethylDataPrep, MethylSegPathway, MethylationStates

reference_dir = Path("data/reference_files")
sample_name = "TCGA-BD-A3EP-01A"
sample_file = reference_dir / "TCGA-BD-A3EP-01A_450k.tsv.gz"

sample_info, removed_df = MethylDataPrep(
    meth_file=sample_file,
    sample_id=sample_name,
    resolution="450k",
).prepare()

pathway = MethylSegPathway.get_pretrained_model(out_dir="out", hmm_type="ct")
regions = pathway.generate_regions(
    sample_info=sample_info,
    chrom="chr1",
    force_resegment=True,
)
_, clean_dir = pathway.get_clean_regions(
    regions_df=regions,
    sample_id=sample_info.sample_id,
    chrom="chr1",
)

fig = pathway.plot_labels(
    sample_info=sample_info,
    chrom="chr1",
    overlay_state=MethylationStates.PMD,
    use_cleaned_regions=True,
    label_title="Cleaned PMD overlay",
)
```

## Package Layout

- `methylseg/`: installable Python package
- `data/`: small config and reference assets used by examples and pretrained configs
- `examples/`: notebook examples
- `docs/`: Sphinx documentation source

## Notes

- The example notebook in `examples/example.ipynb` is included in the docs as a rendered tutorial.
- API docs are generated from package docstrings with Sphinx `autodoc` and `autosummary`, so docstring updates appear after each rebuild.

## TODO

[ ] Finish all TODOs in repo

[ ] Move away from glfs so it doesn't break for users

[ ] Think about what should be saved and what should be returned

[ ] Add usage text and docstrings to all public functions and classes
clean up plotting functions to make them easier for public use

[ ] Add a function that runs full pipeline from raw input to figures and saves the outputs including running the cleaning step
