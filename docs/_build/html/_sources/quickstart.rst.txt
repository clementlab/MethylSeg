Quickstart
==========

Core workflow
-------------

The most common flow is:

1. Prepare an input methylation table with ``MethylDataPrep``.
2. Load or train a ``MethylSegPathway``.
3. Segment a sample into regions.
4. Post-process those regions with ``get_clean_regions``.

Example
-------

.. code-block:: python

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

   pathway = MethylSegPathway.get_pretrained_model(
       out_dir="out",
       hmm_type="ct",
   )

   regions = pathway.generate_regions(
       sample_info=sample_info,
       chrom="chr1",
       force_resegment=True,
   )

   clean_pmds = pathway.get_clean_regions(regions_df=regions)

Important entrypoints
---------------------

- ``MethylDataPrep`` standardizes WGBS and 450k-style input tables.
- ``MethylSegPathway`` is the top-level workflow object for training and
  segmentation.
- ``MethylStateAssigner`` builds window-based emissions and KMeans state labels.
- ``MethylSegmentor`` applies an HMM backend to convert state labels into
  genomic segments.
- ``MethylSegConfig`` serializes a trained pathway to YAML plus sidecar tables.

Notebook tutorial
-----------------

The packaged example notebook is included under :doc:`tutorials` and is rendered
without execution during docs builds.
