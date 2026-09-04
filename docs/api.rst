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

Advanced utilities and constants
--------------------------------

The following helpers are available from :mod:`methylseg.utils`. They are
useful when building custom plots or integrating MethylSeg results into another
visualization workflow. They are documented here without adding them to the
top-level :mod:`methylseg` import surface.

.. currentmodule:: methylseg.utils

.. autodata:: DEFAULT_BIOLOGICAL_STATE_COLORS

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_biological_state_colors
   get_cluster_colors
   resolve_state_color_lookup
   build_region_overlay_df
   resolve_region_overlay_df
   resolve_overlay_plot_args
   annotate_plot_df_with_regions
   plot_state_labels
   plot_interactive_beta_scatter
