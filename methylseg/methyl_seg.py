import os
import textwrap
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cthmm
import joblib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import umap
import yaml
from hmmlearn import hmm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numba import njit
from panel import GridSpec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore", message="divide by zero encountered in log", module="cthmm"
)
