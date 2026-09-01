Full Pipeline And Outputs
=========================

``MethylSegPathway.run_pathway`` fits a new model, segments the requested
chromosomes, cleans the resulting calls, and writes summary BED files. Use it
when the training sample is also the sample you want to segment.

The pathway writes per-chromosome raw calls, cleaned calls, and genome-wide
summary BED files under its ``out_dir``. ``generate_regions`` returns the raw
region table for one chromosome. ``get_clean_regions`` merges and filters those
calls, writes cleaned metadata under ``clean_regions/``, and must be run before
``plot_labels(use_cleaned_regions=True)`` can read cleaned overlays.

For a complete runnable implementation, see ``examples/run_full_pipeline.ipynb``.
