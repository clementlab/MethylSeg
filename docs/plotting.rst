Plotting MethylSeg Results
==========================

MethylSeg plotting is available at three levels: pathway methods for typical workflows, component methods for diagnostics, and utility functions for custom renderers. For runnable demonstrations, see :doc:`the plotting examples </tutorials/generated/07_plotting_examples>` and :doc:`the components example </tutorials/generated/05_methylseg_components_example>`.

Pathway-level plotting
----------------------

:meth:`methylseg.MethylSegPathway.plot_labels` is the standard genomic beta-value scatter plot. Choose the label family with ``label_source``:

* ``"hmm"`` plots final HMM-smoothed labels. It requires ``chrom`` because HMM labels are chromosome-specific.
* ``"kmeans"`` plots KMeans-derived labels before HMM smoothing.
* ``"rule_based"`` plots labels generated from analyzer cutoffs.

``chrom`` may also restrict KMeans and rule-based sample plots, but it is required only for pathway-level HMM label plots. Pass ``sample_info_removed`` from preprocessing to show removed CpGs as a background layer.

.. code-block:: python

   fig = pathway.plot_labels(
       label_source="hmm",
       sample_info=sample_info,
       sample_info_removed=removed_df,
       chrom="chr1",
       use_cleaned_regions=True,
       region_start=2_200_000,
       region_end=3_700_000,
   )

Set ``use_cleaned_regions=True`` after cleaning to overlay metadata written by :meth:`methylseg.MethylSegPathway.get_clean_regions`. Pass ``overlay_regions_df`` for an explicit region table, optionally with ``overlay_state`` to select a state from cleaned outputs. ``region_start`` and ``region_end`` zoom the viewport only; they do not create an overlay.

:meth:`methylseg.MethylSegPathway.plot_embedding` provides PCA or UMAP views of KMeans, rule-based, or HMM labels. It uses the same ``label_source`` concept, but visualizes feature space instead of genomic coordinates.

.. code-block:: python

   embedding = pathway.plot_embedding(
       label_source="kmeans",
       sample_info=sample_info,
       method="pca",
       n_components=2,
   )

Assigner diagnostics
--------------------

:class:`methylseg.MethylStateAssigner` owns emission construction and KMeans visualization. Access it as ``pathway.assigner`` after fitting.

* :meth:`methylseg.MethylStateAssigner.plot_embedding` is the flexible sample-level embedding interface.
* :meth:`methylseg.MethylStateAssigner.plot_training_embedding` and :meth:`methylseg.MethylStateAssigner.plot_train_pca_clusters` inspect the training embedding saved during fitting.
* :meth:`methylseg.MethylStateAssigner.plot_kmeans_clusters`, :meth:`methylseg.MethylStateAssigner.plot_pca_clusters`, and :meth:`methylseg.MethylStateAssigner.plot_umap_clusters` provide direct plotting control when emissions and labels are already available.
* :meth:`methylseg.MethylStateAssigner.plot_labels` renders KMeans labels along genomic coordinates.
* :meth:`methylseg.MethylStateAssigner.plot_feature_distributions_by_kmeans_state` shows training-emission distributions by learned state.

Use these methods to diagnose feature separation and KMeans behavior. Fit first so the assigner has a trained model and cached training data.

Analyzer and segmentor diagnostics
----------------------------------

:class:`methylseg.MethylStateAnalyzer` compares and applies biological label definitions. Use :meth:`methylseg.MethylStateAnalyzer.plot_labels` with ``label_source="kmeans"`` or ``"rule_based"`` to inspect pre-HMM labels. Use :meth:`methylseg.MethylStateAnalyzer.evaluate_clustering_concordance` after rule cutoffs are available to return and display a KMeans-versus-rule confusion matrix.

:class:`methylseg.MethylSegmentor` owns final HMM labels. Its :meth:`methylseg.MethylSegmentor.plot_labels` is the focused counterpart to ``pathway.plot_labels(label_source="hmm", ...)``. Run segmentation first and provide ``chrom`` for an HMM genomic view.

Custom colors and overlays
--------------------------

The default biological palette is :data:`methylseg.utils.DEFAULT_BIOLOGICAL_STATE_COLORS`. It is keyed by canonical state names: ``LOW``, ``PMD``, ``INTERMEDIATE``, and ``HIGH``. The constant remains under :mod:`methylseg.utils` rather than becoming a top-level package import.

.. code-block:: python

   from methylseg.utils import (
       DEFAULT_BIOLOGICAL_STATE_COLORS,
       build_region_overlay_df,
       get_biological_state_colors,
   )

   state_colors = {
       **DEFAULT_BIOLOGICAL_STATE_COLORS,
       "PMD": "#b2182b",
   }
   overlay = build_region_overlay_df(
       region_chrom="chr1",
       region_start=2_200_000,
       region_end=3_700_000,
       label="Candidate PMD",
   )
   fig = pathway.plot_labels(
       label_source="hmm",
       chrom="chr1",
       overlay_regions_df=overlay,
       state_colors=state_colors,
   )

:func:`methylseg.utils.get_biological_state_colors` returns Matplotlib and Plotly-compatible forms of the canonical palette. :func:`methylseg.utils.get_cluster_colors` creates a discrete palette for raw integer cluster labels. For custom renderers, :func:`methylseg.utils.plot_state_labels` and :func:`methylseg.utils.plot_interactive_beta_scatter` provide the shared plotting implementations.

The advanced overlay helpers :func:`methylseg.utils.build_region_overlay_df`, :func:`methylseg.utils.resolve_region_overlay_df`, :func:`methylseg.utils.resolve_overlay_plot_args`, and :func:`methylseg.utils.annotate_plot_df_with_regions` support explicit region tables and custom integration work.

Choosing a plotting layer
-------------------------

Use pathway methods for ordinary analysis because they dispatch to the correct component and can load cleaned-region metadata. Use assigner and analyzer methods for state-assignment diagnostics before HMM smoothing. Use segmentor methods to inspect final HMM labels alone, and utility functions only when constructing a figure from a prepared plotting table.

See :doc:`Methylome segmentation workflow <methylome_segmentation>` for the workflow that creates the raw and cleaned artifacts used by these plots.
