Quickstart
==========

The standard MethylSeg workflow trains a model on a prepared sample, generates
regions, cleans them, and optionally plots the cleaned calls. See
:doc:`input_format` before adapting this example to your own files.

Train A WGBS Model
------------------

.. code-block:: python

   from pathlib import Path

   from methylseg import MethylSegPathway, MethylationStates
   from methylseg.helper_classes import DATA_DIR

   reference_dir = DATA_DIR / "reference_files"
   sample_info, removed_df = MethylSegPathway.prepare_sample_info(
       sample_name="WGBS_colon-primary-tumor_1",
       sample_file=reference_dir / "WGBS_colon-primary-tumor_1_wgbs.tsv.gz",
       resolution="wgbs",
       min_coverage=10,
   )
   pathway = MethylSegPathway(
       train_sample_info=sample_info,
       out_dir=Path("methylseg_output") / sample_info.sample_id,
   )
   pathway.fit_pathway()
   regions = pathway.generate_regions(sample_info=sample_info, chrom="chr1")
   pathway.get_clean_regions(
       regions_df=regions,
       sample_id=sample_info.sample_id,
       chrom="chr1",
   )
   pathway.plot_labels(
       sample_info=sample_info,
       sample_info_removed=removed_df,
       chrom="chr1",
       overlay_state=MethylationStates.PMD,
       use_cleaned_regions=True,
   )

Train A TCGA HM450K Model
-------------------------

Use the same workflow with a beta-value table and ``resolution="450k"``:

.. code-block:: python

   sample_info, removed_df = MethylSegPathway.prepare_sample_info(
       sample_name="TCGA-BD-A3EP-01A",
       sample_file=reference_dir / "TCGA-BD-A3EP-01A_450k.tsv.gz",
       resolution="450k",
   )
   pathway = MethylSegPathway(
       train_sample_info=sample_info,
       out_dir=Path("methylseg_output") / sample_info.sample_id,
   )
   summary_paths = pathway.run_pathway(
       sample_info=sample_info,
       chroms=["chr1"],
   )

The defaults select a Sticky HMM for WGBS and a continuous-time HMM for 450K
data. ``summary_paths`` lists the raw and cleaned summary BED files.

Saved Models
------------

After completing a training run, use a saved model for fast downstream
experiments. The public advanced notebooks demonstrate this path, and
``examples/use_preloaded_model.ipynb`` is the focused saved-model tutorial.
