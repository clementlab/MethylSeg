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

PyPI installation can be added here once the package is published there. The
repository also includes a conda environment file at [environment.yml](./environment.yml)
for local environment management.

## Quickstart

```python
from pathlib import Path

from methylseg import MethylDataPrep, MethylSegPathway

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
clean_pmds = pathway.get_clean_regions(regions_df=regions)
```

## Package Layout

- `methylseg/`: installable Python package
- `data/`: small config and reference assets used by examples and pretrained configs
- `examples/`: notebook examples
- `docs/`: Sphinx documentation source

## Notes

- The example notebook in `examples/example.ipynb` is included in the docs as a rendered tutorial.
- API docs are generated from package docstrings with Sphinx `autodoc` and `autosummary`, so docstring updates appear after each rebuild.
