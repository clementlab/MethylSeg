Input Format
============

MethylSeg reads tab-delimited methylation tables. Column names may use the
aliases accepted by ``MethylDataPrep``; the canonical names below are the most
portable choice. Plain ``.tsv`` and gzip-compressed ``.tsv.gz`` files are both
supported.

WGBS
----

WGBS input supplies methylated-read and total-coverage counts. MethylSeg
calculates ``beta`` internally and can filter low-coverage rows with
``min_coverage``.

.. code-block:: text

   CpG_chrm	CpG_beg	CpG_end	meth	coverage
   chr1	10468	10469	7	12
   chr1	10470	10471	3	10
   chr1	10483	10484	15	18

HM450K / TCGA
-------------

Array input provides beta values directly. A fifth probe identifier column is
optional.

.. code-block:: text

   CpG_chrm	CpG_beg	CpG_end	beta	probe
   chr1	10468	10469	0.583	cg00000029
   chr1	10470	10471	0.271	cg00000108
   chr1	10483	10484	0.842	cg00000109

The complete example inputs are placed in ``data/reference_files/`` by
``methylseg download_data_files``. The example notebooks use those files
directly.
