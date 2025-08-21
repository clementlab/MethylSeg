"""
This module initializes the `MethylSeg` package by explicitly importing key classes
from various submodules. It also defines the `__all__` list to specify what is
exposed when using `from MethylSeg import *`.
"""

from .segmentation_steps.data_prep import DataPrep
from .segmentation_steps.meth_seg import MethSegMethod
from .segmentation_steps.region_identifier import GenerateMethylationRegions
from .segmentation_steps.preprocess_window_data import WindowPreprocessor
from .segmentation_steps.windowed_methylation_preprocessor import (
    WindowedMethylationPreProcessor,
)
from .segmentation_steps.data_preprocessor import DataPreprocessor  # <-- added

__all__ = [
    "DataPrep",
    "MethSegMethod",
    "GenerateMethylationRegions",
    "WindowPreprocessor",
    "WindowedMethylationPreProcessor",
    "DataPreprocessor",
]
