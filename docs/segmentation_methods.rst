Segmentation Methods
====================

MethylSeg supports two ways to assign coarse biological states before Hidden
Markov Model (HMM) smoothing: a learned KMeans-based method and a rule-based
method. In both cases, the assigned per-CpG labels are passed into
``MethylSegmentor`` and then smoothed into genomic segments.

If you want to choose between them, the short version is:

- Use KMeans when you want the state definitions to be learned from training data
  and then reused across samples.
- Use the rule-based method when you want an interpretable single-sample
  analysis and want to explain why a locus was separated into a given state.

Core Classes
------------

The main classes involved in both workflows are:

- :class:`methylseg.MethylSegPathway`: high-level entry point that prepares data,
  trains the model components, runs HMM segmentation, and writes outputs.
- :class:`methylseg.MethylStateAssigner`: builds emission features and trains or
  applies the KMeans model.
- :class:`methylseg.MethylStateAnalyzer`: interprets emission features and
  applies rule-based state definitions.
- :class:`methylseg.MethylSegmentor`: converts the chosen per-CpG state labels
  into HMM-smoothed segments.
- :class:`methylseg.MethylStateAssignmentMethod`: enum used to choose
  ``KMEANS`` or ``DEFINITION``.
- :class:`methylseg.MethylationStates`: biological state labels such as
  ``LOW``, ``PMD``, ``INTERMEDIATE``, and ``HIGH``.

Internally, the learned KMeans model and its preprocessing objects are stored in
``methylseg.helper_classes.KMeansMethylationModel``.

How The Two Methods Fit Into The Pipeline
-----------------------------------------

``MethylSegPathway.fit_pathway()`` always trains the KMeans side first through
``MethylStateAssigner.train_kmeans_for_sample()``. When the selected assignment
method is rule-based, the pathway also optimizes rule cutoffs through
``MethylStateAnalyzer.optimize_rule_params_random()``.

When you later call ``generate_regions()`` or ``run_pathway()``,
``MethylSegmentor.assign_states()`` chooses one of two branches:

- ``MethylStateAssignmentMethod.KMEANS`` uses the trained KMeans model to label
  each emission profile.
- ``MethylStateAssignmentMethod.DEFINITION`` uses the analyzer's rule cutoffs to
  assign biological states directly from the emission features.

After that branch point, both methods continue through the same HMM and region
generation code.

KMeans-Based Segmentation
-------------------------

The KMeans workflow is the default public pathway. It learns state groupings
from the training sample's emission features and then reuses those learned
patterns when scoring the same sample or a new sample prepared the same way.

This method is a good fit when:

- you want a data-driven definition of methylation states,
- you expect to apply the same trained model across multiple samples, or
- you want the HMM to start from clusters learned from multifeature emission
  profiles instead of explicit hand-defined cutoffs.

Relevant classes and methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :class:`methylseg.MethylSegPathway`
- :class:`methylseg.MethylStateAssigner`
- :class:`methylseg.MethylSegmentor`
- :attr:`methylseg.MethylStateAssignmentMethod.KMEANS`
- ``MethylStateAssigner.train_kmeans_for_sample()``
- ``MethylStateAssigner.apply_kmeans_to_sample()``

Typical usage
^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path

   from methylseg import MethylSegPathway, MethylStateAssignmentMethod

   sample_info, removed_df = MethylSegPathway.prepare_sample_info(
       sample_name="sample_1",
       sample_file="sample_1.tsv.gz",
       resolution="wgbs",
       min_coverage=10,
   )

   pathway = MethylSegPathway(
       train_sample_info=sample_info,
       state_assignment_method=MethylStateAssignmentMethod.KMEANS,
       out_dir=Path("out") / sample_info.sample_id,
   )
   pathway.fit_pathway()
   regions = pathway.generate_regions(sample_info=sample_info, chrom="chr1")
   pathway.plot_labels(
       label_source="kmeans",
       sample_info=sample_info,
       sample_info_removed=removed_df,
       chrom="chr1",
   )

For a complete runnable example, see ``examples/03_kmeans_based_model.ipynb``.

Rule-Based Segmentation
-----------------------

The rule-based method is especially useful for a single-sample analysis when
your goal is to understand what is causing the separation between states and to
characterize those states directly. Instead of relying on learned cluster
boundaries alone, it applies explicit cutoffs to the emission features.

This method is a good fit when:

- you want an interpretable explanation of each state's definition,
- you are studying one sample closely and want to inspect why loci separate,
  or
- you want to describe the biological states with explicit feature thresholds.

What causes the separation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rule-based path uses ``MethylStateAnalyzer.define_states_by_rules()`` to
label each emission row. The separation is driven by:

- the locus beta value,
- the fraction of intermediate probes in regional windows,
- the within-window beta standard deviation,
- the fraction of highly methylated probes in each window, and
- the fraction of low-methylated probes in each window.

In the default implementation, a locus is labeled:

- ``LOW`` when beta is below ``beta_low_max`` and the PMD rule does not apply,
- ``HIGH`` when beta is above ``beta_high_min`` and the PMD rule does not
  apply,
- ``PMD`` when beta is in the intermediate range and at least one regional
  window matches the PMD cutoffs, and
- ``INTERMEDIATE`` otherwise.

That makes the rule-based method useful when you need to say not just that two
regions were separated, but also whether the separation came from global beta
level, local heterogeneity, or the surrounding PMD-like neighborhood profile.

Relevant classes and methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :class:`methylseg.MethylSegPathway`
- :class:`methylseg.MethylStateAnalyzer`
- :class:`methylseg.MethylSegmentor`
- :attr:`methylseg.MethylStateAssignmentMethod.DEFINITION`
- ``MethylStateAnalyzer.optimize_rule_params_random()``
- ``MethylStateAnalyzer.define_states_by_rules()``

Typical usage
^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path

   from methylseg import MethylSegPathway, MethylStateAssignmentMethod

   sample_info, removed_df = MethylSegPathway.prepare_sample_info(
       sample_name="sample_1",
       sample_file="sample_1.tsv.gz",
       resolution="wgbs",
       min_coverage=10,
   )

   pathway = MethylSegPathway(
       train_sample_info=sample_info,
       state_assignment_method=MethylStateAssignmentMethod.DEFINITION,
       out_dir=Path("out") / sample_info.sample_id,
   )
   pathway.fit_pathway()
   regions = pathway.generate_regions(sample_info=sample_info, chrom="chr1")
   pathway.plot_labels(
       label_source="rule_based",
       sample_info=sample_info,
       sample_info_removed=removed_df,
       chrom="chr1",
   )

For a complete runnable example, see ``examples/04_rule_based_model.ipynb``.

Comparing The Two
-----------------

KMeans is the better default when you want a reusable trained model for broader
sample processing. The rule-based method is often the better choice when you are
working through one sample and want to explain the biological meaning of the
separation between LOW, PMD, INTERMEDIATE, and HIGH states.

Both methods can still be visualized through
``MethylSegPathway.plot_labels(label_source="kmeans"|"rule_based"|"hmm")`` so
you can compare the initial biological labels with the final HMM-smoothed
segments.
