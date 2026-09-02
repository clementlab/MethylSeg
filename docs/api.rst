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
   MethylationStates
   HMMType

HMM backends
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MethylSegHMM
   StickyCategoricalMethylSegHMM
   CTMethylSegHMM

Utility helpers
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_biological_state_colors
   get_cluster_colors
