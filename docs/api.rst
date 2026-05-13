API Reference
=============

The API reference is generated from the package docstrings. Rebuilding the docs
is enough to pick up docstring updates.

Workflow objects
----------------

.. currentmodule:: methylseg

.. autosummary::
   :toctree: generated
   :nosignatures:

   MethylSegPathway
   MethylDataPrep
   SampleInfo
   MethylSegConfig
   MethylStateAssigner
   MethylStateAnalyzer
   MethylSegmentor

Enums and labels
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MethylStateAssignmentMethod
   HMMObservationMode
   MethylationStates

HMM backends
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MethylSegHMM
   DAMethylSegHMM
   MultinomialSegHMM
   StickyCategoricalMethylSegHMM
   GaussianMethylSegHMM
   CTMethylSegHMM

Utility helpers
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_biological_state_colors
   get_cluster_colors

.. toctree::
   :hidden:

   generated/methylseg.CTMethylSegHMM
   generated/methylseg.DAMethylSegHMM
   generated/methylseg.GaussianMethylSegHMM
   generated/methylseg.HMMObservationMode
   generated/methylseg.MethylDataPrep
   generated/methylseg.MethylSegConfig
   generated/methylseg.MethylSegHMM
   generated/methylseg.MethylSegPathway
   generated/methylseg.MethylSegmentor
   generated/methylseg.MethylStateAnalyzer
   generated/methylseg.MethylStateAssigner
   generated/methylseg.MethylStateAssignmentMethod
   generated/methylseg.MethylationStates
   generated/methylseg.MultinomialSegHMM
   generated/methylseg.SampleInfo
   generated/methylseg.StickyCategoricalMethylSegHMM
   generated/methylseg.get_biological_state_colors
   generated/methylseg.get_cluster_colors
