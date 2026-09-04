Methylome Segmentation Workflow
===============================

This implementation-level companion to :doc:`MethylSeg methodology <methylseg_methodology>` maps the public workflow methods to the components that perform the work. For a runnable end-to-end example, see :doc:`the full pipeline notebook </tutorials/generated/02_run_full_pipeline>`.

Workflow layers
---------------

1. Prepare an input table as ``SampleInfo`` with :meth:`methylseg.MethylSegPathway.prepare_sample_info`.
2. Fit state-assignment components with :meth:`methylseg.MethylSegPathway.fit_pathway`.
3. Segment one chromosome with :meth:`methylseg.MethylSegPathway.generate_regions` or many with :meth:`methylseg.MethylSegPathway.run_on_all_chroms`.
4. Clean raw regions with :meth:`methylseg.MethylSegPathway.get_clean_regions`.
5. Use :meth:`methylseg.MethylSegPathway.run_pathway` to perform steps 2 through 4 in one call.

Sample preparation
------------------

:meth:`~methylseg.MethylSegPathway.prepare_sample_info` delegates input parsing and filtering to :class:`methylseg.MethylDataPrep`. It returns the canonical ``sample_info`` object plus ``removed_df``, a table of CpGs removed during preprocessing.

.. code-block:: python

   from pathlib import Path
   from methylseg import MethylSegPathway

   sample_info, removed_df = MethylSegPathway.prepare_sample_info(
       sample_name="sample-a",
       sample_file=Path("sample-a.tsv.gz"),
       resolution="wgbs",  # or "450k"
       remove_low_coverage_like_cpgs=True,
   )

Keep ``removed_df`` if it should appear as a background layer in label plots.

Fitting state assignment
------------------------

Create a pathway with a training sample, then call :meth:`~methylseg.MethylSegPathway.fit_pathway`. This trains the ``assigner`` through :meth:`methylseg.MethylStateAssigner.train_kmeans_for_sample`. For a rule-based configuration, it also optimizes analyzer cutoffs through :meth:`methylseg.MethylStateAnalyzer.optimize_rule_params_random`.

.. code-block:: python

   pathway = MethylSegPathway(
       train_sample_info=sample_info,
       out_dir=Path("out") / sample_info.sample_id,
   )
   pathway.fit_pathway()

Use ``train_chroms`` and ``max_cpg_per_chrom`` in the constructor to limit fitting data. See :doc:`the KMeans notebook </tutorials/generated/03_kmeans_based_model>` and :doc:`the rule-based notebook </tutorials/generated/04_rule_based_model>` for model-specific configuration.

Segmentation and cleaning
-------------------------

:meth:`~methylseg.MethylSegPathway.generate_regions` segments one chromosome, creates contiguous regions, writes raw BED files, and applies the configured minimum-length filter to the returned table.

.. code-block:: python

   chr1_regions = pathway.generate_regions(
       sample_info=sample_info,
       chrom="chr1",
       min_probes=3,
   )

:meth:`~methylseg.MethylSegPathway.run_on_all_chroms` is the genome-scale counterpart. It accepts a chromosome subset, writes raw summaries, and by default cleans the regions and writes cleaned summaries.

.. code-block:: python

   summary_paths = pathway.run_on_all_chroms(
       sample_info=sample_info,
       chroms=["chr1", "chr2"],
       clean_regions=True,
   )

For the Sticky HMM, selected chromosomes are segmented jointly. Other backends call ``generate_regions`` once per chromosome. Both paths use :meth:`methylseg.MethylSegmentor.segment_sample` and :meth:`methylseg.MethylSegmentor.create_regions`.

:meth:`~methylseg.MethylSegPathway.get_clean_regions` merges and filters raw regions using pathway settings. It writes chromosome-local artifacts in ``clean_regions/``. When called genome-wide, it can also write cleaned summaries; its metadata TSV files are used for cleaned-region plotting.

.. code-block:: python

   state_paths, metadata_path = pathway.get_clean_regions(
       regions_df=chr1_regions,
       sample_id=sample_info.sample_id,
       chrom="chr1",
   )

What ``run_pathway()`` calls
----------------------------

:meth:`~methylseg.MethylSegPathway.run_pathway` is the convenience entry point. The table shows its orchestration path; summary writing is an implementation detail included to explain the resulting artifacts.

.. list-table:: End-to-end call chain
   :header-rows: 1
   :widths: 28 35 37

   * - Entry point
     - Delegates to
     - Result
   * - ``run_pathway()``
     - ``fit_pathway()``
     - Trains KMeans and optionally optimizes rule cutoffs.
   * - ``run_pathway()``
     - ``run_on_all_chroms()``
     - Resolves selected chromosomes and coordinates segmentation.
   * - ``run_on_all_chroms()``
     - ``generate_regions()`` per chromosome, or joint Sticky-HMM processing
     - Produces raw region tables and chromosome-level BED files.
   * - Region generation
     - ``segmentor.segment_sample()`` then ``segmentor.create_regions()``
     - Smooths CpG labels and groups contiguous states into regions.
   * - ``run_on_all_chroms()`` with ``clean_regions=True``
     - ``get_clean_regions()``
     - Writes cleaned BED and metadata artifacts.
   * - ``run_on_all_chroms()``
     - Summary-file writing
     - Writes genome-wide raw summaries and, when requested, cleaned summaries.

Alternative usage patterns
--------------------------

One-call workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   summary_paths = pathway.run_pathway(
       sample_info=sample_info,
       chroms=["chr1", "chr2"],
       clean_regions=True,
   )

Explicit fitting and per-chromosome segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pathway.fit_pathway()
   regions_by_chrom = {
       chrom: pathway.generate_regions(sample_info=sample_info, chrom=chrom)
       for chrom in ["chr1", "chr2"]
   }

Manual postprocessing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   raw_regions = pathway.generate_regions(sample_info=sample_info, chrom="chr1")
   pathway.get_clean_regions(
       regions_df=raw_regions,
       sample_id=sample_info.sample_id,
       chrom="chr1",
       merge_gap_bp=50_000,
       min_region_length=10_000,
   )

Component-oriented usage
~~~~~~~~~~~~~~~~~~~~~~~~

The pathway exposes ``assigner``, ``analyzer``, and ``segmentor`` for custom analysis after fitting.

.. code-block:: python

   pathway.fit_pathway()
   meth_data, emissions = pathway.assigner.prepare_sample_for_clustering(
       sample_info=sample_info, chrom="chr1",
   )
   kmeans_result = pathway.assigner.apply_kmeans_to_sample(
       sample_info=sample_info, chrom="chr1",
   )
   rule_labels = pathway.analyzer.define_states_by_rules(
       sample_info=sample_info, chrom="chr1",
   )
   segmented_data, hmm_model = pathway.segmentor.segment_sample(
       sample_info=sample_info, chrom="chr1",
   )
   regions = pathway.segmentor.create_regions(region_min_probes=3)

For a complete component walkthrough, see :doc:`the components example </tutorials/generated/05_methylseg_components_example>`. Component calls have more state and ordering requirements than pathway methods.

Related guides
--------------

* :doc:`MethylSeg methodology <methylseg_methodology>` explains the biological and statistical rationale.
* :doc:`Plotting guide <plotting>` explains how to inspect labels, embeddings, and cleaned region overlays.
