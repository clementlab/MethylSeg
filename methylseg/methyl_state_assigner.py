"""Window-based emission feature engineering and KMeans state assignment."""

from enum import Enum
import textwrap
import warnings

from matplotlib import pyplot as plt
from panel import GridSpec
import plotly.express as px
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import umap


from .helper_classes import (
    KMeansMethylationModel,
    MethylationStates,
    SampleInfo,
    CANONICAL_AUTOSOMES,
)
from .utils import (
    build_emission_matrix_numba,
    get_biological_state_colors,
    get_present_biological_states,
    relabel_by_mean_emission,
)


class MethylStateAssigner:
    """Create per-CpG window summaries and assign coarse methylation states."""

    def __init__(
        self,
        window_specs: List[Tuple[int, str]] = [
            (40_000, "40kb"),
            (450_000, "450kb"),
        ],
        n_states: int = 4,
        int_low_cutoff: float = 0.2,
        int_high_cutoff: float = 0.7,
        high_cutoff: float = 0.7,
        out_dir=".",
        random_state: Optional[int] = 42,
        cluster_space: str = "pca",
        n_pca: Optional[int] = 5,
    ):
        """
        Parameters
        ----------
        window_specs
            List of ``(window_size_bp, label)`` tuples used to summarize local
            methylation context around each CpG.
        n_states
            Number of coarse methylation states to learn during clustering.
        int_low_cutoff
            Lower cutoff for intermediate methylation state.
        int_high_cutoff
            Upper cutoff for intermediate methylation state.
        high_cutoff
            Cutoff for high methylation state.
        out_dir
            Directory to save output files.
        random_state
            Random state for reproducibility.
        cluster_space
            Space in which to perform k-means clustering ('pca' or 'raw').
        n_pca
            Number of principal components to use if cluster_space is 'pca'.
        """
        self.window_specs = window_specs
        self.n_states = n_states
        self.int_low_cutoff = int_low_cutoff
        self.int_high_cutoff = int_high_cutoff
        self.high_cutoff = high_cutoff
        self.out_dir = out_dir
        self.random_state = random_state
        self.cluster_space = self._validate_cluster_space(cluster_space)
        self.n_pca = n_pca

    @staticmethod
    def _validate_cluster_space(cluster_space: str) -> str:
        normalized_cluster_space = str(cluster_space).lower()
        if normalized_cluster_space not in {"pca", "raw"}:
            raise ValueError(
                "cluster_space must be either 'pca' or 'raw'. "
                f"Received: {cluster_space!r}"
            )
        return normalized_cluster_space

    def build_emission_matrix(
        self,
        positions,
        betas,
        window_specs,
        int_low_cutoff,
        int_high_cutoff,
        high_cutoff,
    ):
        """
        Build emission features for one ordered probe sequence.

        Parameters
        ----------
        positions
            Genomic positions for each CpG.
        betas
            Beta values aligned to ``positions``.
        window_specs
            ``(window_size_bp, label)`` pairs describing the local summary
            windows to compute.
        int_low_cutoff, int_high_cutoff, high_cutoff
            Thresholds used to derive low/intermediate/high proportions inside
            each window.

        Returns
        -------
        tuple
            ``(X, feature_names)`` where ``X`` is the numeric emission matrix
            and ``feature_names`` are the corresponding column labels.
        """

        window_sizes = np.array([w[0] for w in window_specs], dtype=np.int64)

        X = build_emission_matrix_numba(
            positions,
            betas,
            window_sizes,
            int_low_cutoff,
            int_high_cutoff,
            high_cutoff,
        )

        # Feature names (Python side)
        feature_names = ["beta"]
        for _, label in window_specs:
            feature_names.extend(
                [
                    f"{label}_avg_meth",
                    f"{label}_std",
                    f"{label}_high_pct",
                    f"{label}_int_pct",
                    f"{label}_low_pct",
                    f"{label}_n_cpg",
                ]
            )

        return X, feature_names

    def _transform_emission_features(
        self,
        feature_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_matrix = feature_matrix.copy()
        count_cols = [col for col in feature_matrix.columns if col.endswith("_n_cpg")]
        for col in count_cols:
            col_values = pd.to_numeric(feature_matrix[col], errors="raise")
            if (col_values.dropna() < 0).any():
                raise ValueError(
                    f"Emission count feature '{col}' contains negative values, "
                    "so log1p preprocessing cannot be applied."
                )
            feature_matrix[col] = np.log1p(col_values.astype(np.float64))
        return feature_matrix

    def preprocess_emission_features(
        self,
        emission_df: pd.DataFrame,
        feature_cols: List[str],
        fit: bool = False,
    ):
        """
        Impute and scale emission features for clustering or inference.

        Parameters
        ----------
        emission_df
            Emission-feature table.
        feature_cols
            Ordered columns to extract and preprocess.
        fit
            If ``True``, fit a new imputer/scaler pair and return them alongside
            the transformed matrix. Otherwise, reuse the trained model's
            preprocessing objects.

        Returns
        -------
        numpy.ndarray or tuple
            Scaled feature matrix, or ``(scaled_values, imputer, scaler)`` when
            ``fit=True``.

        Raises
        ------
        ValueError
            If preprocessing cannot produce a finite feature matrix or a
            required trained preprocessing component is missing.
        """
        feature_matrix = emission_df[feature_cols].copy()
        feature_matrix = self._transform_emission_features(feature_matrix)

        all_nan_cols = feature_matrix.columns[feature_matrix.isna().all()].tolist()
        if fit and all_nan_cols:
            raise ValueError(
                "Emission features are entirely missing for columns: "
                f"{all_nan_cols}. Median imputation cannot fit these features."
            )

        feature_values = feature_matrix.to_numpy(dtype=np.float64, copy=True)
        if np.isinf(feature_values).any():
            raise ValueError(
                "Emission features contain infinite values before imputation."
            )
        has_missing = np.isnan(feature_values).any()

        if fit:
            if has_missing:
                imputer = SimpleImputer(strategy="median")
                imputed_values = imputer.fit_transform(feature_values)
            else:
                imputer = None
                imputed_values = feature_values
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(imputed_values)
            if not np.isfinite(scaled_values).all():
                raise ValueError(
                    "Emission preprocessing produced non-finite values during fit."
                )
            return scaled_values, imputer, scaler

        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        if getattr(self.model, "scaler", None) is None:
            raise ValueError(
                "The trained KMeans model is missing a scaler. Refit the model "
                "with the updated preprocessing pipeline."
            )

        if getattr(self.model, "imputer", None) is None:
            if has_missing:
                raise ValueError(
                    "Emission features contain missing values, but the trained "
                    "KMeans model was fit without an imputer."
                )
            imputed_values = feature_values
        elif has_missing:
            imputed_values = self.model.imputer.transform(feature_values)
        else:
            imputed_values = feature_values
        if not np.isfinite(imputed_values).all():
            raise ValueError(
                "Emission preprocessing produced non-finite values after imputation."
            )
        scaled_values = self.model.scaler.transform(imputed_values)
        if not np.isfinite(scaled_values).all():
            raise ValueError(
                "Emission preprocessing produced non-finite values after scaling."
            )
        return scaled_values

    def resolve_feature_cols(
        self,
        emission_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Resolve the emission columns to use for clustering features.

        Parameters
        ----------
        emission_df
            Emission-feature table whose columns should be filtered.
        feature_cols
            Optional explicit column list. When omitted, uses all non-count
            features.

        Returns
        -------
        list of str
            Ordered feature-column names passed to preprocessing and clustering.
        """
        if feature_cols is None:
            feature_cols = [
                col
                for col in emission_df.columns.tolist()
                if not col.endswith("_n_cpg")
            ]

        if not feature_cols:
            raise ValueError(
                "No emission feature columns were selected. Pass feature_cols "
                "explicitly if you want to include only count-based features."
            )

        return feature_cols

    def fit_kmeans_on_emissions(
        self,
        emission_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[KMeansMethylationModel, Optional[np.ndarray], np.ndarray]:
        """
        Fit KMeans on emission features using the assigner's configured cluster space.

        Parameters
        ----------
        emission_df
            Emission-feature table used for clustering.
        feature_cols
            Optional explicit feature-column list. When omitted, uses
            ``resolve_feature_cols``.

        Returns
        -------
        tuple
            ``(model, pca_scores, relabeled_labels)`` where ``pca_scores`` is
            ``None`` when clustering in raw feature space.
        """
        feature_cols = self.resolve_feature_cols(
            emission_df=emission_df,
            feature_cols=feature_cols,
        )

        X_scaled, imputer, scaler = self.preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=feature_cols,
            fit=True,
        )

        if self.cluster_space == "pca":
            if self.n_pca is None or self.n_pca <= 0:
                raise ValueError(
                    "n_pca must be a positive integer when cluster_space='pca'."
                )
            n_components = min(self.n_pca, X_scaled.shape[0], X_scaled.shape[1])
            if n_components <= 0:
                raise ValueError(
                    "Cannot fit PCA for clustering because the emission matrix is empty."
                )
            pca = PCA(n_components=n_components, random_state=self.random_state)
            kmeans_input = pca.fit_transform(X_scaled)
            pca_scores = kmeans_input
        else:
            pca = None
            kmeans_input = X_scaled
            pca_scores = None

        kmeans = KMeans(
            n_clusters=self.n_states, n_init=10, random_state=self.random_state
        )
        raw_labels = kmeans.fit_predict(kmeans_input)
        relabeled = relabel_by_mean_emission(
            raw_labels=raw_labels,
            emission_df=emission_df,
            int_low_cutoff=self.int_low_cutoff,
            int_high_cutoff=self.int_high_cutoff,
            window_specs=self.window_specs,
        )
        model = KMeansMethylationModel(
            kmeans=kmeans,
            scaler=scaler,
            imputer=imputer,
            pca=pca,
            feature_cols=feature_cols,
            n_states=self.n_states,
            cluster_space=self.cluster_space,
            n_pca=self.n_pca,
        )
        return model, pca_scores, relabeled

    def apply_kmeans_to_emissions(
        self,
        emission_df: pd.DataFrame,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a previously trained KMeansMethylationModel to a new emission_df.

        Parameters
        ----------
        emission_df
            Emission-feature table to score with the trained model.

        Returns
        -------
        tuple
            ``(pca_scores, raw_distances, raw_labels, relabeled_labels)`` for
            the supplied emission rows.
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        X_scaled = self.preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )

        if self.model.pca is not None:
            pca_scores = self.model.pca.transform(X_scaled)
            kmeans_input = pca_scores
        else:
            pca_scores = None
            kmeans_input = X_scaled

        raw_distances = self.model.kmeans.transform(kmeans_input)

        raw_labels = self.model.kmeans.predict(kmeans_input)
        relabeled = relabel_by_mean_emission(
            raw_labels=raw_labels,
            emission_df=emission_df,
            int_low_cutoff=self.int_low_cutoff,
            int_high_cutoff=self.int_high_cutoff,
            window_specs=self.window_specs,
        )
        return pca_scores, raw_distances, raw_labels, relabeled

    def _get_kmeans_metric_input(self, emission_df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        X_scaled = self.preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )
        metric_cluster_space = getattr(self.model, "cluster_space", self.cluster_space)
        if metric_cluster_space == "pca":
            if getattr(self.model, "pca", None) is None:
                raise ValueError(
                    "The trained KMeans model is configured for PCA clustering but "
                    "is missing a PCA model."
                )
            return self.model.pca.transform(X_scaled)
        return X_scaled

    def calculate_kmeans_cluster_metrics(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """
        Calculate clustering-quality metrics in the same feature space used for KMeans.

        Parameters
        ----------
        emission_df
            Emission-feature table aligned to ``labels``.
        labels
            Cluster labels to evaluate in the trained feature space.

        Returns
        -------
        dict of str to float or None
            Clustering metric values keyed by metric name. Metrics that cannot
            be computed are returned as ``None``.
        """
        metrics = {
            "silhouette_score": None,
            "davies_bouldin_score": None,
        }

        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        labels_array = np.asarray(labels)
        if labels_array.ndim == 0:
            labels_array = labels_array.reshape(1)
        if labels_array.size == 0:
            return metrics

        labels_numeric = self._normalize_plot_labels(
            labels=labels_array,
            expected_length=len(emission_df),
        )
        metric_input = self._get_kmeans_metric_input(emission_df)

        unique_labels = np.unique(labels_numeric)
        n_samples = metric_input.shape[0]
        n_clusters = len(unique_labels)
        if n_samples < 2 or n_clusters < 2 or n_clusters >= n_samples:
            return metrics

        try:
            metrics["silhouette_score"] = float(
                silhouette_score(metric_input, labels_numeric)
            )
        except ValueError:
            pass

        try:
            metrics["davies_bouldin_score"] = float(
                davies_bouldin_score(metric_input, labels_numeric)
            )
        except ValueError:
            pass

        return metrics

    def _fit_plot_pca(
        self,
        emission_df: pd.DataFrame,
        n_pca_plot: int,
    ) -> Tuple[PCA, np.ndarray, List[str], bool]:
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        if n_pca_plot not in (2, 3):
            raise ValueError("n_pca_plot must be either 2 or 3.")

        X_scaled = self.preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )
        feature_names = list(self.model.feature_cols)

        if self.model.pca is not None:
            pca = self.model.pca
            if pca.n_components_ < n_pca_plot:
                raise ValueError(
                    "The trained PCA model does not contain enough components for "
                    f"a {n_pca_plot}D PCA plot."
                )
            return pca, pca.transform(X_scaled), feature_names, True

        max_plot_components = min(X_scaled.shape[0], X_scaled.shape[1])
        if max_plot_components < n_pca_plot:
            raise ValueError(
                "Not enough samples/features are available to fit a temporary PCA "
                f"with {n_pca_plot} components."
            )

        pca = PCA(n_components=n_pca_plot, random_state=self.random_state)
        plot_scores = pca.fit_transform(X_scaled)
        return pca, plot_scores, feature_names, False

    def plot_embedding(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        meth_data: pd.DataFrame | None = None,
        *,
        method: str = "pca",
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
        n_components: int = 2,
        top_n_loadings: int = 5,
        hexbin: bool = False,
        hexbin_gridsize: int = 60,
        hexbin_bins: str | int | list[float] | np.ndarray | None = "log",
        hexbin_mincnt: int | None = 1,
        hexbin_alpha: float | None = None,
        hexbin_linewidths: float | None = None,
        interactive: bool = False,
        include_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        region_start: int | None = None,
        region_end: int | None = None,
        region_chrom: str | None = None,
        use_pca_features: bool = False,
        use_parallel: bool = True,
        show_plot: bool = True,
        state_colors: dict | None = None,
    ):
        """
        Plot PCA or UMAP embeddings for an emission table and state labels.

        Parameters
        ----------
        emission_df
            Emission-feature table to embed.
        labels
            Cluster or biological state labels aligned to ``emission_df``.
        meth_data
            Optional probe-level methylation table used for region-aware PCA
            highlighting.
        method
            Embedding method, either ``"pca"`` or ``"umap"``.
        sample_info
            Optional sample metadata used for plot titles.
        chrom
            Optional chromosome label used for plot titles and region-aware
            views.
        n_components
            Number of embedding dimensions to render.
        top_n_loadings
            Number of PCA loading features to show in tables and biplots.
        hexbin
            If ``True``, render 2-D PCA as hexbins instead of points.
        hexbin_gridsize
            Hexbin grid resolution for 2-D PCA hexbin plots.
        hexbin_bins
            Hexbin binning strategy for 2-D PCA hexbin plots, for example
            ``"log"``, an integer bin count, explicit bin edges, or ``None``.
        hexbin_mincnt
            Minimum points required to draw a hexbin in 2-D PCA hexbin plots.
            Use ``None`` to let matplotlib draw all bins.
        hexbin_alpha
            Optional global transparency multiplier for 2-D PCA hexbin plots.
        hexbin_linewidths
            Optional hexagon border width for 2-D PCA hexbin plots.
        interactive
            If ``True``, use interactive rendering when supported by the chosen
            method.
        include_metrics
            Include clustering-quality metrics where available.
        include_biplot
            Overlay top PCA loading vectors on 2-D PCA plots.
        label_title
            Legend or colorbar title.
        region_start, region_end, region_chrom
            Optional genomic interval used to highlight overlapping CpGs in PCA
            space.
        use_pca_features
            For UMAP, project the trained PCA features instead of scaled raw
            features.
        use_parallel
            Whether to allow UMAP's parallel execution mode.
        show_plot
            If ``True``, display the figure immediately.
        state_colors
            Optional biological-state color overrides.

        Returns
        -------
        object
            Matplotlib or Plotly figure object, depending on the selected
            rendering path.
        """
        method = str(method).lower()
        sample_name = None if sample_info is None else sample_info.sample_id

        if method == "pca":
            region_requested = any(
                value is not None for value in (region_start, region_end, region_chrom)
            )
            if region_requested:
                if meth_data is None:
                    raise ValueError(
                        "meth_data must be provided when requesting PCA region highlighting."
                    )
                if region_start is None or region_end is None:
                    raise ValueError(
                        "region_start and region_end must both be provided when "
                        "requesting PCA region highlighting."
                    )
                return self.plot_pca_clusters_with_region(
                    meth_data=meth_data,
                    emission_df=emission_df,
                    labels=labels,
                    region_start=region_start,
                    region_end=region_end,
                    region_chrom=region_chrom,
                    n_pca_plot=n_components,
                    top_n_loadings=top_n_loadings,
                    pca_hexbin=hexbin,
                    hexbin_gridsize=hexbin_gridsize,
                    hexbin_bins=hexbin_bins,
                    hexbin_mincnt=hexbin_mincnt,
                    hexbin_alpha=hexbin_alpha,
                    hexbin_linewidths=hexbin_linewidths,
                    interactive=interactive,
                    include_kmeans_metrics=include_metrics,
                    include_biplot=include_biplot,
                    label_title=label_title,
                    sample_name=sample_name,
                    chrom=chrom,
                    show_plot=show_plot,
                    state_colors=state_colors,
                )
            return self.plot_pca_clusters(
                emission_df=emission_df,
                labels=labels,
                n_pca_plot=n_components,
                top_n_loadings=top_n_loadings,
                pca_hexbin=hexbin,
                hexbin_gridsize=hexbin_gridsize,
                hexbin_bins=hexbin_bins,
                hexbin_mincnt=hexbin_mincnt,
                hexbin_alpha=hexbin_alpha,
                hexbin_linewidths=hexbin_linewidths,
                interactive=interactive,
                include_kmeans_metrics=include_metrics,
                include_biplot=include_biplot,
                label_title=label_title,
                sample_name=sample_name,
                chrom=chrom,
                show_plot=show_plot,
                state_colors=state_colors,
            )

        if method == "umap":
            return self.plot_umap_clusters(
                emission_df=emission_df,
                labels=labels,
                chrom=chrom,
                sample_name=sample_name,
                use_pca=use_pca_features,
                use_parallel=use_parallel,
                show_plot=show_plot,
            )

        raise ValueError(
            "method must be either 'pca' or 'umap'. "
            f"Received: {method!r}"
        )

    #TODO: remove this and make the plot embedding default to plotting training embedding
    def plot_training_embedding(
        self,
        *,
        method: str = "pca",
        n_components: int = 2,
        top_n_loadings: int = 5,
        hexbin: bool = False,
        hexbin_gridsize: int = 60,
        hexbin_bins: str | int | list[float] | np.ndarray | None = "log",
        hexbin_mincnt: int | None = 1,
        hexbin_alpha: float | None = None,
        hexbin_linewidths: float | None = None,
        interactive: bool = False,
        include_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        region_start: int | None = None,
        region_end: int | None = None,
        region_chrom: str | None = None,
        use_pca_features: bool = False,
        use_parallel: bool = True,
        show_plot: bool = True,
        state_colors: dict | None = None,
    ):
        """
        Plot embeddings for the cached training sample and labels.

        Parameters
        ----------
        method
            Embedding method, either ``"pca"`` or ``"umap"``.
        n_components
            Number of embedding dimensions to render.
        top_n_loadings
            Number of PCA loading features to show in tables and biplots.
        hexbin
            If ``True``, render 2-D PCA as hexbins instead of points.
        hexbin_gridsize
            Hexbin grid resolution for 2-D PCA hexbin plots.
        hexbin_bins
            Hexbin binning strategy for 2-D PCA hexbin plots, for example
            ``"log"``, an integer bin count, explicit bin edges, or ``None``.
        hexbin_mincnt
            Minimum points required to draw a hexbin in 2-D PCA hexbin plots.
            Use ``None`` to let matplotlib draw all bins.
        hexbin_alpha
            Optional global transparency multiplier for 2-D PCA hexbin plots.
        hexbin_linewidths
            Optional hexagon border width for 2-D PCA hexbin plots.
        interactive
            If ``True``, use interactive rendering when supported.
        include_metrics
            Include clustering-quality metrics where available.
        include_biplot
            Overlay top PCA loading vectors on 2-D PCA plots.
        label_title
            Legend or colorbar title.
        region_start, region_end, region_chrom
            Optional genomic interval used to highlight overlapping training
            CpGs in PCA space.
        use_pca_features
            For UMAP, project the trained PCA features instead of scaled raw
            features.
        use_parallel
            Whether to allow UMAP's parallel execution mode.
        show_plot
            If ``True``, display the figure immediately.
        state_colors
            Optional biological-state color overrides.

        Returns
        -------
        object
            Matplotlib or Plotly figure object.
        """
        required_attrs = ["train_emission_df", "train_labels", "train_sample_info"]
        if any(value is not None for value in (region_start, region_end, region_chrom)):
            required_attrs.append("train_meth")

        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise ValueError(
                "No saved training clustering artifacts found. "
                f"Missing attributes: {missing}. Train k-means first."
            )

        return self.plot_embedding(
            emission_df=self.train_emission_df,
            labels=self.train_labels,
            meth_data=getattr(self, "train_meth", None),
            method=method,
            sample_info=self.train_sample_info,
            chrom=self._format_train_chrom_label(),
            n_components=n_components,
            top_n_loadings=top_n_loadings,
            hexbin=hexbin,
            hexbin_gridsize=hexbin_gridsize,
            hexbin_bins=hexbin_bins,
            hexbin_mincnt=hexbin_mincnt,
            hexbin_alpha=hexbin_alpha,
            hexbin_linewidths=hexbin_linewidths,
            interactive=interactive,
            include_metrics=include_metrics,
            include_biplot=include_biplot,
            label_title=label_title,
            region_start=region_start,
            region_end=region_end,
            region_chrom=region_chrom,
            use_pca_features=use_pca_features,
            use_parallel=use_parallel,
            show_plot=show_plot,
            state_colors=state_colors,
        )

    def plot_umap_clusters(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        chrom: Optional[str] = None,
        sample_name: Optional[str] = None,
        use_pca: bool = False,
        use_parallel: bool = True,
        show_plot: bool = True,
    ):
        """
        Plot a 2-D UMAP embedding colored by state labels.

        Parameters
        ----------
        emission_df
            Emission-feature table to embed.
        labels
            State labels aligned to ``emission_df``.
        chrom
            Optional chromosome label used in the plot title.
        sample_name
            Optional sample identifier used in the plot title.
        use_pca
            If ``True``, run UMAP on the trained PCA features instead of scaled
            raw features.
        use_parallel
            If ``True``, allow UMAP to disable a fixed random seed for parallel
            execution.
        show_plot
            If ``True``, display the figure immediately.

        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot of the UMAP embedding.
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        random_state = None if use_parallel else self.random_state
        if use_parallel:
            print(
                "UMAP parallelisation cannot work with random seed, setting random_state to None for UMAP."
            )

        X_scaled = self.preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=self.model.feature_cols,
            fit=False,
        )

        if use_pca:
            if self.model.pca is None:
                raise ValueError(
                    "model.pca is None: PCA-backed UMAP requires fitting with n_pca > 0."
                )
            umap_input = self.model.pca.transform(X_scaled)
            title_suffix = "PCA features"
        else:
            umap_input = X_scaled
            title_suffix = "raw features"

        embedding = umap.UMAP(n_components=2, random_state=random_state).fit_transform(
            umap_input
        )

        cmap, norm, state_colors_rgba, _ = get_biological_state_colors()
        labels_numeric = MethylationStates.convert_to_numeric(labels)

        state_names = {state.value: state.name for state in MethylationStates}
        present_states = get_present_biological_states(labels_numeric)

        title_parts = []
        if sample_name is not None:
            title_parts.append(str(sample_name))
        if chrom is not None:
            title_parts.append(str(chrom))
        title_prefix = " ".join(title_parts) if title_parts else "Sample"

        fig = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels_numeric,
            cmap=cmap,
            norm=norm,
            s=8,
        )
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title(f"{title_prefix}: UMAP + KMeans States ({title_suffix})")
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=state_colors_rgba[state_value],
                markeredgecolor=state_colors_rgba[state_value],
                markersize=7,
                label=state_names.get(state_value, str(state_value)),
            )
            for state_value in present_states
        ]
        if legend_handles:
            plt.legend(handles=legend_handles, title="State", loc="best")
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig

    def plot_kmeans_clusters(
        self,
        meth_data: pd.DataFrame,
        labels: np.ndarray,
        chrom: Optional[str] = None,
        sample_name: Optional[str] = None,
        feature_cols_for_table: Optional[List[str]] = None,
        interactive: bool = False,
    ):
        """
        Plot genomic beta values colored by KMeans-derived state labels.

        Parameters
        ----------
        meth_data
            Probe-level methylation table containing genomic positions and beta
            values.
        labels
            Cluster or biological state labels aligned to ``meth_data``.
        chrom
            Optional chromosome label used in the plot title.
        sample_name
            Optional sample identifier used in the plot title.
        feature_cols_for_table
            Optional feature list to display alongside the static scatter plot.
        interactive
            If ``True``, render the Plotly version instead of the static
            matplotlib figure.

        Returns
        -------
        object
            Matplotlib or Plotly figure object.
        """
        # Convert Enum labels to integers if needed
        if isinstance(labels.flat[0], Enum):
            labels_numeric = np.array([lbl.value for lbl in labels])
        else:
            labels_numeric = labels

        n_states = self.n_states
        cmap, norm, state_colors_rgba, state_colors_hex = get_biological_state_colors()
        present_states = get_present_biological_states(labels_numeric)
        x_pos = meth_data["CpG_beg"].to_numpy()
        y_beta = meth_data["beta"].to_numpy()
        title_parts = []
        if sample_name is not None:
            title_parts.append(str(sample_name))
        if chrom is not None:
            title_parts.append(str(chrom))
        title_prefix = " ".join(title_parts) if title_parts else "Sample"
        if not interactive:

            if len(meth_data) != len(labels):
                raise ValueError("meth_data and labels must be the same length.")

            show_table = (
                feature_cols_for_table is not None and len(feature_cols_for_table) > 0
            )

            if show_table:
                fig = plt.figure(figsize=(18, 6))
                gs = GridSpec(nrows=1, ncols=2, width_ratios=[4, 1], figure=fig)
                ax_scatter = fig.add_subplot(gs[0])
                ax_table = fig.add_subplot(gs[1])
            else:
                fig = plt.figure(figsize=(12, 5))
                ax_scatter = fig.add_subplot(111)
                ax_table = None

            x_pos = meth_data["CpG_beg"].to_numpy()
            y_beta = meth_data["beta"].to_numpy()

            sc = ax_scatter.scatter(
                x_pos,
                y_beta,
                c=labels_numeric,
                cmap=cmap,
                norm=norm,
                s=10,
            )
            ax_scatter.set_xlabel("Genomic Position")
            ax_scatter.set_ylabel("Methylation (beta)")
            ax_scatter.set_title(f"{title_prefix}: Methylation Beta by KMeans Cluster")
            cbar = plt.colorbar(sc, ax=ax_scatter, ticks=present_states, label="State")
            cbar.set_ticklabels(
                [MethylationStates(state_value).name for state_value in present_states]
            )

            if show_table:
                ax_table.axis("off")
                rows = [
                    feature_cols_for_table[i : i + 2]
                    for i in range(0, len(feature_cols_for_table), 2)
                ]
                for r in rows:
                    if len(r) < 2:
                        r.append("")
                table = ax_table.table(
                    cellText=rows,
                    colLabels=["Feature", "Feature"],
                    loc="center",
                    cellLoc="left",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.3)
                ax_table.set_title("Features Used", pad=10)

            plt.tight_layout()
            plt.show()

        if interactive:
            # Always plot using readable biological state names so sparse
            # numeric labels like {0, 2, 3} map correctly to colors.
            labels_str = np.array(
                [MethylationStates(int(lbl)).name for lbl in labels_numeric]
            )

            # Ordered unique label names
            state_names = [
                MethylationStates(state_value).name for state_value in present_states
            ]

            color_map = {
                state_name: state_colors_hex[MethylationStates[state_name].value]
                for state_name in state_names
            }

            fig_interactive = px.scatter(
                x=x_pos,
                y=y_beta,
                color=labels_str,  # Enum names now :)
                color_discrete_map=color_map,
                category_orders={"color": state_names},
                labels={
                    "x": "Genomic Position",
                    "y": "Methylation (beta)",
                    "color": "State",
                },
                title=f"{title_prefix}: Methylation States (Interactive)",
            )
            fig_interactive.update_traces(marker=dict(size=4))
            fig_interactive.show(renderer="notebook")

    def plot_pca_clusters(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        n_pca_plot: int = 2,  # 2 or 3
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,  # True -> old hexbin behavior
        hexbin_gridsize: int = 60,
        hexbin_bins: str | int | list[float] | np.ndarray | None = "log",
        hexbin_mincnt: int | None = 1,
        hexbin_alpha: float | None = None,
        hexbin_linewidths: float | None = None,
        interactive: bool = False,  # 3D Plotly option
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        sample_name: str | None = None,
        chrom: str | None = None,
        show_plot: bool = True,
        state_colors: dict | None = None,
    ):
        """
        PCA embedding + loadings, using consistent colors per state.

        Parameters
        ----------
        emission_df
            Emission-feature table to embed with PCA.
        labels
            Cluster or biological state labels aligned to ``emission_df``.
        n_pca_plot
            Number of PCA dimensions to render, usually ``2`` or ``3``.
        top_n_loadings
            Number of loading features to show in the summary table.
        pca_hexbin
            If ``True``, render 2-D PCA with hexbins instead of points.
        hexbin_gridsize
            Hexbin grid resolution for 2-D PCA hexbin plots.
        hexbin_bins
            Hexbin binning strategy for 2-D PCA hexbin plots.
        hexbin_mincnt
            Minimum points required to draw a hexbin in 2-D PCA hexbin plots.
        hexbin_alpha
            Optional transparency multiplier for 2-D PCA hexbin plots.
        hexbin_linewidths
            Optional border width for 2-D PCA hexbin plots.
        interactive
            If ``True``, use interactive rendering when supported.
        include_kmeans_metrics
            Include clustering-quality metrics in the plotted annotation.
        include_biplot
            Overlay top PCA loading vectors on 2-D PCA plots.
        label_title
            Legend or colorbar title.
        sample_name
            Optional sample identifier used in the plot title.
        chrom
            Optional chromosome label used in the plot title.
        show_plot
            If ``True``, display the figure immediately.
        state_colors
            Optional biological-state color overrides.

        Returns
        -------
        object
            Matplotlib or Plotly figure object from the PCA plotting backend.

        Set ``include_kmeans_metrics=False`` to skip the expensive clustering
        quality metric calculation and annotation.

        In hexbin mode, ``hexbin_mincnt`` is evaluated separately for each
        state and hexagon. The default of one keeps sparse chromosome-level
        plots visible; use a larger value to show only dense bins.
        """
        return self._plot_pca_clusters_impl(
            emission_df=emission_df,
            labels=labels,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
            pca_hexbin=pca_hexbin,
            hexbin_gridsize=hexbin_gridsize,
            hexbin_bins=hexbin_bins,
            hexbin_mincnt=hexbin_mincnt,
            hexbin_alpha=hexbin_alpha,
            hexbin_linewidths=hexbin_linewidths,
            interactive=interactive,
            include_kmeans_metrics=include_kmeans_metrics,
            include_biplot=include_biplot,
            label_title=label_title,
            sample_name=sample_name,
            chrom=chrom,
            highlight_mask=None,
            show_plot=show_plot,
            state_colors=state_colors,
        )

    def _wrap_pca_loading_feature_name(
        self,
        feature_name: str,
        width: int = 24,
    ) -> str:
        if not feature_name:
            return ""
        return textwrap.fill(
            str(feature_name),
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )

    def _build_pca_loadings_table(
        self,
        pca: PCA,
        feature_names: List[str],
        n_pca_plot: int,
        top_n_loadings: int,
    ) -> Tuple[List[List[str]], List[str]]:
        n_pca = pca.n_components_
        signed_loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_pca)],
            index=feature_names,
        )
        abs_loadings = signed_loadings.abs()

        top_pc1 = abs_loadings["PC1"].sort_values(ascending=False).head(top_n_loadings)
        top_pc2 = abs_loadings["PC2"].sort_values(ascending=False).head(top_n_loadings)
        if n_pca_plot == 3 and "PC3" in abs_loadings.columns:
            top_pc3 = (
                abs_loadings["PC3"].sort_values(ascending=False).head(top_n_loadings)
            )
        else:
            top_pc3 = None

        table_data = []
        for i in range(top_n_loadings):
            row = [
                (
                    self._wrap_pca_loading_feature_name(top_pc1.index[i])
                    if i < len(top_pc1)
                    else ""
                ),
                (
                    f"{signed_loadings.loc[top_pc1.index[i], 'PC1']:.3f}"
                    if i < len(top_pc1)
                    else ""
                ),
                (
                    self._wrap_pca_loading_feature_name(top_pc2.index[i])
                    if i < len(top_pc2)
                    else ""
                ),
                (
                    f"{signed_loadings.loc[top_pc2.index[i], 'PC2']:.3f}"
                    if i < len(top_pc2)
                    else ""
                ),
            ]
            if n_pca_plot == 3 and top_pc3 is not None:
                row.extend(
                    [
                        (
                            self._wrap_pca_loading_feature_name(top_pc3.index[i])
                            if i < len(top_pc3)
                            else ""
                        ),
                        (
                            f"{signed_loadings.loc[top_pc3.index[i], 'PC3']:.3f}"
                            if i < len(top_pc3)
                            else ""
                        ),
                    ]
                )
            table_data.append(row)

        if n_pca_plot == 3 and top_pc3 is not None:
            col_labels = [
                "PC1 Feature",
                "Abs Loading",
                "PC2 Feature",
                "Abs Loading",
                "PC3 Feature",
                "Abs Loading",
            ]
        else:
            col_labels = ["PC1 Feature", "Abs Loading", "PC2 Feature", "Abs Loading"]

        return table_data, col_labels

    def _add_pca_highlight_overlay(
        self,
        ax,
        plot_scores: np.ndarray,
        highlight_mask: np.ndarray,
    ) -> Line2D | None:
        if not np.any(highlight_mask):
            return None

        ax.scatter(
            plot_scores[highlight_mask, 0],
            plot_scores[highlight_mask, 1],
            marker="^",
            c="red",
            edgecolors="black",
            linewidths=0.6,
            s=40,
            zorder=5,
        )
        return Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=8,
            label="Highlighted region",
        )

    def _format_kmeans_cluster_metrics(
        self,
        metrics: Dict[str, Optional[float]],
    ) -> str:
        def render_metric(metric_key: str) -> str:
            metric_value = metrics.get(metric_key)
            if metric_value is None or not np.isfinite(metric_value):
                return "n/a"
            return f"{metric_value:.3f}"

        return "\n".join(
            [
                f"Silhouette: {render_metric('silhouette_score')}",
                f"Davies-Bouldin: {render_metric('davies_bouldin_score')}",
            ]
        )

    def _add_pca_metrics_annotation(
        self,
        ax,
        metrics_text: Optional[str],
    ) -> None:
        if not metrics_text:
            return
        ax.text(
            0.98,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )

    def _add_pca_metrics_annotation_3d(
        self,
        ax,
        metrics_text: Optional[str],
    ) -> None:
        if not metrics_text:
            return
        ax.text2D(
            0.98,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )

    def _add_pca_metrics_annotation_plotly(
        self,
        fig,
        metrics_text: Optional[str],
    ) -> None:
        if not metrics_text:
            return
        fig.update_layout(
            annotations=[
                {
                    "x": 0.98,
                    "y": 0.02,
                    "xref": "paper",
                    "yref": "paper",
                    "text": metrics_text.replace("\n", "<br>"),
                    "showarrow": False,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "align": "right",
                    "bgcolor": "rgba(255, 255, 255, 0.85)",
                    "bordercolor": "black",
                    "borderwidth": 1,
                }
            ]
        )

    def _normalize_plot_labels(
        self,
        labels,
        expected_length: int,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1:
            labels_array = labels_array.reshape(-1)

        if len(labels_array) != expected_length:
            raise ValueError(
                "labels and emission_df must be the same length. "
                f"Received {len(labels_array)} labels for {expected_length} emission rows. "
                "If you are plotting HMM labels, make sure they come from the same "
                "meth_data/emissions_df pair rather than from a separately sampled "
                "training table."
            )

        if len(labels_array) == 0:
            return np.array([], dtype=int)

        first_value = labels_array[0]
        if isinstance(first_value, Enum):
            return np.array([label.value for label in labels_array], dtype=int)

        if isinstance(first_value, str):
            label_map = {state.name.lower(): state.value for state in MethylationStates}
            normalized = []
            for label in labels_array:
                label_key = str(label).strip().lower()
                if label_key not in label_map:
                    raise ValueError(
                        "String labels must match methylation state names: "
                        f"{sorted(label_map)}. Received: {label!r}"
                    )
                normalized.append(label_map[label_key])
            return np.array(normalized, dtype=int)

        return labels_array.astype(int)

    def _add_pca_biplot_overlay(
        self,
        ax,
        pca: PCA,
        plot_scores: np.ndarray,
        feature_names: List[str],
        top_n_loadings: int,
    ) -> None:
        if plot_scores.shape[1] < 2 or pca.components_.shape[0] < 2:
            raise ValueError("Biplot overlay requires at least two PCA components.")

        loadings = pd.DataFrame(
            pca.components_.T[:, :2],
            columns=["PC1", "PC2"],
            index=feature_names,
        )
        loading_magnitude = np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2)
        top_features = loading_magnitude.sort_values(ascending=False).head(
            top_n_loadings
        )
        if top_features.empty:
            return

        score_scale = np.max(np.abs(plot_scores[:, :2]), axis=0)
        score_scale = np.where(score_scale == 0, 1.0, score_scale)
        loading_scale = np.max(
            np.abs(loadings.loc[top_features.index, ["PC1", "PC2"]].to_numpy()),
            axis=0,
        )
        loading_scale = np.where(loading_scale == 0, 1.0, loading_scale)
        arrow_scale = 0.8 * np.min(score_scale / loading_scale)
        head_width = max(0.02 * np.max(score_scale), 1e-6)

        for feature_name in top_features.index:
            x_loading = float(loadings.loc[feature_name, "PC1"]) * arrow_scale
            y_loading = float(loadings.loc[feature_name, "PC2"]) * arrow_scale
            ax.arrow(
                0,
                0,
                x_loading,
                y_loading,
                color="black",
                width=0.0,
                head_width=head_width,
                length_includes_head=True,
                alpha=0.8,
                zorder=6,
            )
            ax.text(
                x_loading * 1.08,
                y_loading * 1.08,
                self._wrap_pca_loading_feature_name(feature_name, width=18),
                fontsize=8,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=7,
            )

    def _plot_pca_clusters_impl(
        self,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        n_pca_plot: int = 2,
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,
        hexbin_gridsize: int = 60,
        hexbin_bins: str | int | list[float] | np.ndarray | None = "log",
        hexbin_mincnt: int | None = 1,
        hexbin_alpha: float | None = None,
        hexbin_linewidths: float | None = None,
        interactive: bool = False,
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        sample_name: str | None = None,
        chrom: str | None = None,
        highlight_mask: Optional[np.ndarray] = None,
        show_plot: bool = True,
        state_colors: dict | None = None,
    ):
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        pca, plot_scores, feature_names, _ = self._fit_plot_pca(
            emission_df=emission_df,
            n_pca_plot=n_pca_plot,
        )
        cmap, norm, state_colors_rgba, state_colors_hex = get_biological_state_colors(
            state_colors=state_colors
        )

        labels_numeric = self._normalize_plot_labels(
            labels=labels,
            expected_length=plot_scores.shape[0],
        )
        present_states = get_present_biological_states(labels_numeric)
        if highlight_mask is not None:
            highlight_mask = np.asarray(highlight_mask, dtype=bool)
            if highlight_mask.ndim != 1 or len(highlight_mask) != plot_scores.shape[0]:
                raise ValueError(
                    "highlight_mask must be a 1D boolean array aligned to emission_df."
                )
            if n_pca_plot != 2 or interactive:
                raise ValueError(
                    "Region highlighting is currently supported only for 2-D "
                    "non-interactive PCA plots."
                )

        n_pca = pca.n_components_
        explained_variance = pca.explained_variance_ratio_
        pc_axis_labels = [
            f"PC{i+1} ({explained_variance[i] * 100:.1f}%)" for i in range(n_pca)
        ]
        metrics_text = None
        if include_kmeans_metrics:
            metrics_text = self._format_kmeans_cluster_metrics(
                self.calculate_kmeans_cluster_metrics(
                    emission_df=emission_df,
                    labels=labels,
                )
            )
        table_data, col_labels = self._build_pca_loadings_table(
            pca=pca,
            feature_names=feature_names,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
        )

        fig = plt.figure(figsize=(22 if n_pca_plot == 3 else 20, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.35])

        # Title
        title_parts = []
        if sample_name is not None:
            title_parts.append(str(sample_name))
        if chrom is not None:
            title_parts.append(str(chrom))
        title_prefix = " ".join(title_parts) if title_parts else "Sample"
        title_basis = "PCA"

        # --- PCA embedding ---
        if n_pca_plot == 2:
            ax0 = fig.add_subplot(gs[0])
            highlight_handle = None

            if pca_hexbin:
                cluster_cmaps = []
                for state_value in present_states:
                    base_color = state_colors_rgba[state_value]
                    cluster_cmaps.append(
                        mcolors.LinearSegmentedColormap.from_list(
                            f"cluster_{state_value}", [(1, 1, 1, 0.0), base_color]
                        )
                    )

                for cmap_idx, state_value in enumerate(present_states):
                    mask = labels_numeric == state_value
                    if not np.any(mask):
                        continue
                    hexbin_kwargs = {
                        "gridsize": hexbin_gridsize,
                        "cmap": cluster_cmaps[cmap_idx],
                    }
                    if hexbin_bins is not None:
                        hexbin_kwargs["bins"] = hexbin_bins
                    if hexbin_mincnt is not None:
                        hexbin_kwargs["mincnt"] = hexbin_mincnt
                    if hexbin_alpha is not None:
                        hexbin_kwargs["alpha"] = hexbin_alpha
                    if hexbin_linewidths is not None:
                        hexbin_kwargs["linewidths"] = hexbin_linewidths
                    ax0.hexbin(
                        plot_scores[mask, 0],
                        plot_scores[mask, 1],
                        **hexbin_kwargs,
                    )

                ax0.set_xlabel(pc_axis_labels[0])
                ax0.set_ylabel(pc_axis_labels[1])
                ax0.set_title(
                    f"{title_prefix}: PCA + {label_title}s (2-D Hexbin, {title_basis})"
                )
                self._add_pca_metrics_annotation(ax0, metrics_text)
                if highlight_mask is not None:
                    highlight_handle = self._add_pca_highlight_overlay(
                        ax=ax0,
                        plot_scores=plot_scores,
                        highlight_mask=highlight_mask,
                    )

                from matplotlib.patches import Patch

                handles = [
                    Patch(
                        color=state_colors_rgba[state_value],
                        label=MethylationStates(state_value).name,
                    )
                    for state_value in present_states
                ]
                if highlight_handle is not None:
                    handles.append(highlight_handle)
                ax0.legend(handles=handles, title=label_title, loc="best")

            else:
                sc0 = ax0.scatter(
                    plot_scores[:, 0],
                    plot_scores[:, 1],
                    c=labels_numeric,
                    cmap=cmap,
                    norm=norm,
                    s=8,
                )
                ax0.set_xlabel(pc_axis_labels[0])
                ax0.set_ylabel(pc_axis_labels[1])
                ax0.set_title(
                    f"{title_prefix}: PCA + {label_title}s (2-D, {title_basis})"
                )
                self._add_pca_metrics_annotation(ax0, metrics_text)
                cbar = plt.colorbar(
                    sc0,
                    ax=ax0,
                    ticks=present_states,
                    label=label_title,
                )
                cbar.set_ticklabels(
                    [
                        MethylationStates(state_value).name
                        for state_value in present_states
                    ]
                )
                if highlight_mask is not None:
                    highlight_handle = self._add_pca_highlight_overlay(
                        ax=ax0,
                        plot_scores=plot_scores,
                        highlight_mask=highlight_mask,
                    )
                    if highlight_handle is not None:
                        ax0.legend(
                            handles=[highlight_handle],
                            title="Overlay",
                            loc="best",
                        )

            if include_biplot:
                self._add_pca_biplot_overlay(
                    ax=ax0,
                    pca=pca,
                    plot_scores=plot_scores,
                    feature_names=feature_names,
                    top_n_loadings=top_n_loadings,
                )

        else:
            # 3-D PCA
            if include_biplot:
                raise ValueError(
                    "include_biplot is currently supported only for 2-D PCA plots."
                )
            if interactive and plot_scores.shape[1] >= 3:
                labels_str = np.array(
                    [MethylationStates(int(lbl)).name for lbl in labels_numeric]
                )
                fig_plotly = px.scatter_3d(
                    x=plot_scores[:, 0],
                    y=plot_scores[:, 1],
                    z=plot_scores[:, 2],
                    color=labels_str,
                    color_discrete_map={
                        MethylationStates(state_value).name: state_colors_hex[
                            state_value
                        ]
                        for state_value in present_states
                    },
                    category_orders={
                        "color": [
                            MethylationStates(state_value).name
                            for state_value in present_states
                        ]
                    },
                    labels={
                        "x": pc_axis_labels[0],
                        "y": pc_axis_labels[1],
                        "z": pc_axis_labels[2],
                        "color": label_title,
                    },
                    title=(
                        f"{title_prefix}: PCA + {label_title}s "
                        f"(3-D Interactive, {title_basis})"
                    ),
                )
                self._add_pca_metrics_annotation_plotly(fig_plotly, metrics_text)
                fig_plotly.update_traces(marker=dict(size=3))
                if show_plot:
                    fig_plotly.show(renderer="notebook")
                return fig_plotly
            else:
                ax0 = fig.add_subplot(gs[0], projection="3d")
                sc0 = ax0.scatter(
                    plot_scores[:, 0],
                    plot_scores[:, 1],
                    plot_scores[:, 2],
                    c=labels_numeric,
                    cmap=cmap,
                    norm=norm,
                    s=8,
                )
                ax0.set_xlabel(pc_axis_labels[0])
                ax0.set_ylabel(pc_axis_labels[1])
                ax0.set_zlabel(pc_axis_labels[2])
                ax0.set_title(
                    f"{title_prefix}: PCA + {label_title}s (3-D, {title_basis})"
                )
                self._add_pca_metrics_annotation_3d(ax0, metrics_text)
                cbar = fig.colorbar(sc0, ax=ax0, ticks=present_states, shrink=0.6)
                cbar.set_label(label_title)
                cbar.set_ticklabels(
                    [
                        MethylationStates(state_value).name
                        for state_value in present_states
                    ]
                )

        # --- Loadings table ---
        ax1 = fig.add_subplot(gs[1])
        ax1.axis("off")
        table = ax1.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)
        try:
            table.auto_set_column_width(col=list(range(len(col_labels))))
        except AttributeError:
            pass
        feature_col_indices = list(range(0, len(col_labels), 2))
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_ha("center")
                continue
            if col_idx in feature_col_indices:
                cell.get_text().set_ha("left")
            else:
                cell.get_text().set_ha("center")
        ax1.set_title(f"Top {top_n_loadings} Absolute Loadings", pad=10)

        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig

    def plot_pca_clusters_with_region(
        self,
        meth_data: pd.DataFrame,
        emission_df: pd.DataFrame,
        labels: np.ndarray,
        region_start: int,
        region_end: int,
        region_chrom: Optional[str] = None,
        n_pca_plot: int = 2,
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,
        hexbin_gridsize: int = 60,
        hexbin_bins: str | int | list[float] | np.ndarray | None = "log",
        hexbin_mincnt: int | None = 1,
        hexbin_alpha: float | None = None,
        hexbin_linewidths: float | None = None,
        interactive: bool = False,
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        label_title: str = "State",
        sample_name: str | None = None,
        chrom: str | None = None,
        show_plot: bool = True,
        state_colors: dict | None = None,
    ):
        """
        Plot PCA clusters while highlighting CpGs overlapping a genomic region.

        Parameters
        ----------
        meth_data
            Probe-level methylation table containing genomic coordinates.
        emission_df
            Emission-feature table aligned to ``meth_data``.
        labels
            Cluster or biological state labels aligned to both tables.
        region_start, region_end
            Inclusive genomic interval used for highlighting.
        region_chrom
            Chromosome of the highlighted interval. Required when ``meth_data``
            spans multiple chromosomes.
        n_pca_plot
            Number of PCA dimensions to render.
        top_n_loadings
            Number of PCA loading features to show in the side table.
        pca_hexbin
            If ``True``, render 2-D PCA as hexbins instead of points.
        hexbin_gridsize
            Hexbin grid resolution for 2-D PCA hexbin plots.
        hexbin_bins
            Hexbin binning strategy for 2-D PCA hexbin plots, for example
            ``"log"``, an integer bin count, explicit bin edges, or ``None``.
        hexbin_mincnt
            Minimum points required to draw a hexbin in 2-D PCA hexbin plots.
            Use ``None`` to let matplotlib draw all bins.
        hexbin_alpha
            Optional global transparency multiplier for 2-D PCA hexbin plots.
        hexbin_linewidths
            Optional hexagon border width for 2-D PCA hexbin plots.
        interactive
            If ``True``, use interactive rendering when supported.
        include_kmeans_metrics
            Include clustering-quality metrics in the plot.
        include_biplot
            Overlay top PCA loading vectors on 2-D PCA plots.
        label_title
            Legend or colorbar title.
        sample_name
            Optional sample identifier used in the plot title.
        chrom
            Optional chromosome label used in the plot title.
        show_plot
            If ``True``, display the figure immediately.
        state_colors
            Optional biological-state color overrides.

        Returns
        -------
        object
            Matplotlib or Plotly figure object from the PCA plotting backend.
        """
        required_cols = {"CpG_chrm", "CpG_beg", "CpG_end"}
        missing_cols = sorted(required_cols - set(meth_data.columns))
        if missing_cols:
            raise ValueError(
                "meth_data must contain genomic coordinate columns for region "
                f"highlighting. Missing: {missing_cols}"
            )

        if len(meth_data) != len(emission_df) or len(meth_data) != len(labels):
            raise ValueError(
                "meth_data, emission_df, and labels must be the same length."
            )

        resolved_region_start = int(region_start)
        resolved_region_end = int(region_end)
        if resolved_region_end < resolved_region_start:
            raise ValueError(
                "region_end must be greater than or equal to region_start."
            )

        meth_chroms = meth_data["CpG_chrm"].astype(str)
        if region_chrom is None:
            unique_chroms = meth_chroms.unique()
            if len(unique_chroms) != 1:
                raise ValueError(
                    "region_chrom is required when meth_data contains multiple "
                    "chromosomes."
                )
            resolved_region_chrom = str(unique_chroms[0])
        else:
            resolved_region_chrom = str(region_chrom)

        cpg_beg = pd.to_numeric(meth_data["CpG_beg"], errors="raise").to_numpy(
            dtype=np.int64
        )
        cpg_end = pd.to_numeric(meth_data["CpG_end"], errors="raise").to_numpy(
            dtype=np.int64
        )
        highlight_mask = (
            (meth_chroms.to_numpy() == resolved_region_chrom)
            & (cpg_beg <= resolved_region_end)
            & (cpg_end >= resolved_region_start)
        )

        if not np.any(highlight_mask):
            warnings.warn(
                "No CpGs overlapped the requested genomic region for PCA highlighting: "
                f"{resolved_region_chrom}:{resolved_region_start}-{resolved_region_end}.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self._plot_pca_clusters_impl(
            emission_df=emission_df,
            labels=labels,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
            pca_hexbin=pca_hexbin,
            hexbin_gridsize=hexbin_gridsize,
            hexbin_bins=hexbin_bins,
            hexbin_mincnt=hexbin_mincnt,
            hexbin_alpha=hexbin_alpha,
            hexbin_linewidths=hexbin_linewidths,
            interactive=interactive,
            include_kmeans_metrics=include_kmeans_metrics,
            include_biplot=include_biplot,
            label_title=label_title,
            sample_name=sample_name,
            chrom=chrom,
            highlight_mask=highlight_mask,
            show_plot=show_plot,
            state_colors=state_colors,
        )

    def _format_train_chrom_label(
        self,
    ) -> str | None:
        train_chroms = getattr(self, "train_chroms", None)
        if train_chroms is None:
            legacy_train_chrom = getattr(self, "train_chrom", None)
            train_chroms = (
                [legacy_train_chrom] if legacy_train_chrom is not None else None
            )

        if not train_chroms:
            return None

        chroms = [str(chrom) for chrom in train_chroms]
        if len(chroms) == 1:
            return chroms[0]
        if set(chroms) == set(CANONICAL_AUTOSOMES) and len(chroms) == len(
            CANONICAL_AUTOSOMES
        ):
            return "autosomes"
        if len(chroms) <= 3:
            return ",".join(chroms)
        return f"{len(chroms)} chromosomes"


    #TODO: plotting region-specific PCA clusters is currently only not working and needs to be debugged
    def plot_train_pca_clusters(
        self,
        n_pca_plot: int = 2,
        top_n_loadings: int = 5,
        pca_hexbin: bool = False,
        hexbin_gridsize: int = 60,
        hexbin_bins: str | int | list[float] | np.ndarray | None = "log",
        hexbin_mincnt: int | None = 1,
        hexbin_alpha: float | None = None,
        hexbin_linewidths: float | None = None,
        interactive: bool = False,
        include_kmeans_metrics: bool = True,
        include_biplot: bool = False,
        region_start: Optional[int] = None,
        region_end: Optional[int] = None,
        region_chrom: Optional[str] = None,
        show_plot: bool = True,
    ):
        """
        Convenience wrapper to plot the PCA embedding saved from k-means training.

        Parameters
        ----------
        n_pca_plot
            Number of PCA dimensions to render, usually ``2`` or ``3``.
        top_n_loadings
            Number of loading features to show in the summary table.
        pca_hexbin
            If ``True``, render 2-D PCA with hexbins instead of points.
        hexbin_gridsize
            Hexbin grid resolution for 2-D PCA hexbin plots.
        hexbin_bins
            Hexbin binning strategy for 2-D PCA hexbin plots.
        hexbin_mincnt
            Minimum points required to draw a hexbin in 2-D PCA hexbin plots.
        hexbin_alpha
            Optional transparency multiplier for 2-D PCA hexbin plots.
        hexbin_linewidths
            Optional border width for 2-D PCA hexbin plots.
        interactive
            If ``True``, use interactive rendering when supported.
        include_kmeans_metrics
            Include clustering-quality metrics in the plotted annotation.
        include_biplot
            Overlay top PCA loading vectors on 2-D PCA plots.
        region_start
            Optional genomic start coordinate for region highlighting.
        region_end
            Optional genomic end coordinate for region highlighting.
        region_chrom
            Optional chromosome for region highlighting.
        show_plot
            If ``True``, display the figure immediately.

        Returns
        -------
        object
            Matplotlib or Plotly figure object from the PCA plotting backend.
        """
        region_requested = any(
            value is not None for value in (region_start, region_end, region_chrom)
        )
        if region_requested and (region_start is None or region_end is None):
            raise ValueError(
                "region_start and region_end must both be provided when requesting "
                "region highlighting."
            )

        required_attrs = ["train_emission_df", "train_labels"]
        if region_requested:
            required_attrs.append("train_meth")

        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise ValueError(
                "No saved training clustering artifacts found. "
                f"Missing attributes: {missing}. Train k-means first."
            )

        resolved_sample_name = getattr(self, "train_sample", None)
        resolved_chrom_label = self._format_train_chrom_label()

        if region_start is not None and region_end is not None:
            return self.plot_pca_clusters_with_region(
                meth_data=self.train_meth,
                emission_df=self.train_emission_df,
                labels=self.train_labels,
                region_start=region_start,
                region_end=region_end,
                region_chrom=region_chrom,
                n_pca_plot=n_pca_plot,
                top_n_loadings=top_n_loadings,
                pca_hexbin=pca_hexbin,
                hexbin_gridsize=hexbin_gridsize,
                hexbin_bins=hexbin_bins,
                hexbin_mincnt=hexbin_mincnt,
                hexbin_alpha=hexbin_alpha,
                hexbin_linewidths=hexbin_linewidths,
                interactive=interactive,
                include_kmeans_metrics=include_kmeans_metrics,
                include_biplot=include_biplot,
                sample_name=resolved_sample_name,
                chrom=resolved_chrom_label,
                show_plot=show_plot,
            )

        return self.plot_pca_clusters(
            emission_df=self.train_emission_df,
            labels=self.train_labels,
            n_pca_plot=n_pca_plot,
            top_n_loadings=top_n_loadings,
            pca_hexbin=pca_hexbin,
            hexbin_gridsize=hexbin_gridsize,
            hexbin_bins=hexbin_bins,
            hexbin_mincnt=hexbin_mincnt,
            hexbin_alpha=hexbin_alpha,
            hexbin_linewidths=hexbin_linewidths,
            interactive=interactive,
            include_kmeans_metrics=include_kmeans_metrics,
            include_biplot=include_biplot,
            sample_name=resolved_sample_name,
            chrom=resolved_chrom_label,
            show_plot=show_plot,
        )

    def _subset_emission_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        windows_to_use: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if windows_to_use is None:
            return X, feature_names

        keep_cols = ["beta"]
        for label in windows_to_use:
            keep_cols.extend(
                [
                    f"{label}_avg_meth",
                    f"{label}_std",
                    f"{label}_high_pct",
                    f"{label}_int_pct",
                    f"{label}_low_pct",
                    f"{label}_n_cpg",
                ]
            )

        col_indices = [feature_names.index(c) for c in keep_cols]
        return X[:, col_indices], keep_cols

    def _prepare_filtered_sample_for_clustering(
        self,
        meth_data: pd.DataFrame,
        windows_to_use: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(meth_data) == 0:
            raise ValueError("No CpGs remaining after filtering.")

        positions = meth_data["CpG_beg"].to_numpy(dtype=np.int64)
        betas = meth_data["beta"].to_numpy(dtype=np.float64)

        order = np.argsort(positions, kind="mergesort")
        positions = positions[order]
        betas = betas[order]
        meth_data = meth_data.iloc[order].reset_index(drop=True)

        X, feature_names = self.build_emission_matrix(
            positions=positions,
            betas=betas,
            window_specs=self.window_specs,
            int_low_cutoff=self.int_low_cutoff,
            int_high_cutoff=self.int_high_cutoff,
            high_cutoff=self.high_cutoff,
        )
        X, feature_names = self._subset_emission_features(
            X=X,
            feature_names=feature_names,
            windows_to_use=windows_to_use,
        )
        emission_df = pd.DataFrame(X, columns=feature_names)
        return meth_data, emission_df

    def prepare_sample_for_clustering(
        self,
        sample_info: SampleInfo,
        chrom: Optional[str] = None,
        windows_to_use: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare probe-level and emission-feature tables for clustering.

        Parameters
        ----------
        sample_info
            Prepared sample whose methylation rows should be summarized.
        chrom
            Optional chromosome restriction for per-chromosome preparation.
        windows_to_use
            Optional subset of configured window labels to retain in the
            emission matrix.

        Returns
        -------
        tuple
            ``(meth_data, emission_df)`` aligned for downstream clustering or
            plotting.
        """

        meth_data = sample_info.meth_data.copy()

        if chrom is not None:
            meth_data = meth_data[meth_data["CpG_chrm"] == chrom]
            return self._prepare_filtered_sample_for_clustering(
                meth_data=meth_data,
                windows_to_use=windows_to_use,
            )

        if len(meth_data) == 0:
            raise ValueError("No CpGs remaining after filtering.")

        chrom_series = meth_data["CpG_chrm"].astype(str)
        chrom_order = chrom_series.drop_duplicates().tolist()
        if len(chrom_order) <= 1:
            return self._prepare_filtered_sample_for_clustering(
                meth_data=meth_data,
                windows_to_use=windows_to_use,
            )

        meth_frames = []
        emission_frames = []

        for chrom_name in chrom_order:
            chrom_meth = meth_data[chrom_series == chrom_name].copy()
            chrom_meth, chrom_emission_df = (
                self._prepare_filtered_sample_for_clustering(
                    meth_data=chrom_meth,
                    windows_to_use=windows_to_use,
                )
            )
            meth_frames.append(chrom_meth)
            emission_frames.append(chrom_emission_df)

        combined_meth = pd.concat(meth_frames, ignore_index=True)
        combined_emission_df = pd.concat(emission_frames, ignore_index=True)
        return combined_meth, combined_emission_df

    def _resolve_train_chroms(
        self,
        sample_info: SampleInfo,
        train_chroms: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        available_chroms = set(sample_info.meth_data["CpG_chrm"].astype(str).unique())

        if train_chroms is None:
            resolved = [
                chrom for chrom in CANONICAL_AUTOSOMES if chrom in available_chroms
            ]
            missing = []
        else:
            resolved = []
            missing = []
            for chrom in train_chroms:
                chrom = str(chrom)
                if chrom in available_chroms:
                    resolved.append(chrom)
                else:
                    missing.append(chrom)

        if not resolved:
            requested = (
                list(train_chroms)
                if train_chroms is not None
                else list(CANONICAL_AUTOSOMES)
            )
            raise ValueError(
                "No eligible training chromosomes remained after filtering. "
                f"Requested: {requested}"
            )

        return resolved, missing

    def _prepare_training_data_for_kmeans(
        self,
        sample_info: SampleInfo,
        train_chroms: Optional[List[str]] = None,
        windows_to_use: Optional[List[str]] = None,
        max_cpg_per_chrom: Optional[int] = 50_000,
        sampling_random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if max_cpg_per_chrom is not None and max_cpg_per_chrom <= 0:
            raise ValueError("max_cpg_per_chrom must be > 0 or None.")

        resolved_chroms, missing_chroms = self._resolve_train_chroms(
            sample_info=sample_info,
            train_chroms=train_chroms,
        )

        seed = (
            self.random_state
            if sampling_random_state is None
            else sampling_random_state
        )
        rng = np.random.default_rng(seed)

        meth_frames = []
        emission_frames = []
        summary_rows = []

        for chrom in resolved_chroms:
            meth_data, emission_df = self.prepare_sample_for_clustering(
                sample_info=sample_info,
                chrom=chrom,
                windows_to_use=windows_to_use,
            )

            n_total = len(meth_data)
            if max_cpg_per_chrom is not None and n_total > max_cpg_per_chrom:
                sampled_idx = np.sort(
                    rng.choice(n_total, size=max_cpg_per_chrom, replace=False)
                )
                was_sampled = True
            else:
                sampled_idx = np.arange(n_total)
                was_sampled = False

            meth_frames.append(meth_data.iloc[sampled_idx].reset_index(drop=True))
            emission_frames.append(emission_df.iloc[sampled_idx].reset_index(drop=True))

            summary_rows.append(
                {
                    "chrom": chrom,
                    "available": True,
                    "selected_for_training": True,
                    "n_cpg_total": int(n_total),
                    "n_cpg_sampled": int(len(sampled_idx)),
                    "sampled": was_sampled,
                    "sampling_fraction": float(len(sampled_idx) / n_total),
                }
            )

        for chrom in missing_chroms:
            summary_rows.append(
                {
                    "chrom": chrom,
                    "available": False,
                    "selected_for_training": False,
                    "n_cpg_total": 0,
                    "n_cpg_sampled": 0,
                    "sampled": False,
                    "sampling_fraction": 0.0,
                }
            )

        train_meth_data = pd.concat(meth_frames, ignore_index=True)
        train_emission_df = pd.concat(emission_frames, ignore_index=True)
        training_summary_df = pd.DataFrame(summary_rows)

        return train_meth_data, train_emission_df, training_summary_df

    def train_kmeans_for_sample(
        self,
        sample_info: SampleInfo,
        train_chroms: Optional[List[str]] = None,
        windows_to_use: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        max_cpg_per_chrom: Optional[int] = 50_000,
        sampling_random_state: Optional[int] = None,
    ):
        """
        Train a KMeans model on one prepared sample.

        Parameters
        ----------
        sample_info
            Prepared sample used to build the training emission matrix.
        train_chroms
            Optional chromosome list to use for training. When omitted, uses
            the canonical autosomes present in the sample.
        windows_to_use
            Optional subset of configured window labels to retain in the
            emission matrix.
        feature_cols
            Optional explicit feature-column list passed to clustering.
        max_cpg_per_chrom
            Optional maximum CpGs to sample per chromosome before fitting.
        sampling_random_state
            Optional random seed controlling per-chromosome subsampling.

        Returns
        -------
        tuple
            ``(model, meth_data, emission_df, pca_scores, labels)`` for the
            fitted training sample.
        """
        self.train_sample = sample_info.sample_id
        self.train_sample_info = sample_info
        meth_data, emission_df, training_summary_df = (
            self._prepare_training_data_for_kmeans(
                sample_info=sample_info,
                train_chroms=train_chroms,
                windows_to_use=windows_to_use,
                max_cpg_per_chrom=max_cpg_per_chrom,
                sampling_random_state=sampling_random_state,
            )
        )
        self.train_chroms, _ = self._resolve_train_chroms(
            sample_info=sample_info,
            train_chroms=train_chroms,
        )
        self.max_cpg_per_chrom = max_cpg_per_chrom
        self.train_meth = meth_data
        self.train_emission_df = emission_df
        self.training_summary_df = training_summary_df

        model, pca_scores, labels = self.fit_kmeans_on_emissions(
            emission_df=emission_df,
            feature_cols=feature_cols,
        )
        self.model = model
        self.train_pca_scores = pca_scores
        self.train_labels = labels

        return model, meth_data, emission_df, pca_scores, labels

    def apply_kmeans_to_sample(
        self,
        sample_info: SampleInfo,
        chrom: Optional[str] = None,
        windows_to_use: Optional[List[str]] = None,
        sample_meth_data: Optional[pd.DataFrame] = None,
    ):
        """
        Apply an already-trained KMeans model to a prepared sample.

        Parameters
        ----------
        sample_info
            Prepared sample to score with the trained model.
        chrom
            Optional chromosome restriction for per-chromosome scoring.
        windows_to_use
            Optional subset of configured window labels to retain in the
            emission matrix.
        sample_meth_data
            Reserved compatibility argument for prefiltered methylation rows.

        Returns
        -------
        tuple
            ``(meth_data, emission_df, pca_scores, raw_distances, raw_labels,
            relabeled_labels)`` for the supplied sample.
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")

        meth_data, emission_df = self.prepare_sample_for_clustering(
            sample_info=sample_info,
            chrom=chrom,
            windows_to_use=windows_to_use,
        )

        missing = [c for c in self.model.feature_cols if c not in emission_df.columns]
        if missing:
            raise ValueError(
                f"emission_df for sample {sample_info.sample_id} is missing features: {missing}"
            )

        pca_scores, raw_distances, raw_labels, relabeled_labels = (
            self.apply_kmeans_to_emissions(emission_df)
        )
        return (
            meth_data,
            emission_df,
            pca_scores,
            raw_distances,
            raw_labels,
            relabeled_labels,
        )

    def get_pca_loadings(
        self,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of PCA loadings for the features used in the model.
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        pca = self.model.pca
        if pca is None:
            raise ValueError("model.pca is None: PCA loadings are not available.")

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=self.model.feature_cols,
        )
        return loadings
