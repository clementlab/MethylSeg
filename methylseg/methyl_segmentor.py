"""HMM-backed segmentation from per-CpG state labels to genomic regions."""

from typing import Dict, List, Optional, Tuple


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .utils import (
    plot_interactive_beta_scatter,
    relabel_by_mean_emission,
    resolve_overlay_plot_args,
    resolve_region_overlay_df,
)
from .helper_classes import (
    HMMObservationMode,
    MethylStateAssignmentMethod,
    MethylationStates,
    SampleInfo,
)
from .methylseg_hmm import MethylSegHMM
from .methyl_state_analyzer import MethylStateAnalyzer


class MethylSegmentor:
    """
    Class to handle segmentation of methylation data using HMMs.
    Recommend CTHMM for sparse data with variable probe spacing, and the
    sticky categorical smoother for dense discrete state-label smoothing.
    """

    def __init__(
        self,
        analyzer: MethylStateAnalyzer,
        hmm_model: MethylSegHMM,
        state_assignment_method: MethylStateAssignmentMethod = MethylStateAssignmentMethod.DEFINITION,
        hmm_observation_mode: HMMObservationMode = HMMObservationMode.DISCRETE_STATES,
        out_dir=".",
        random_state: int = 42,
    ):
        self.analyzer = analyzer
        self.hmm_model = hmm_model
        self.state_assignment_method = MethylStateAssignmentMethod(
            state_assignment_method
        )
        self.hmm_observation_mode = HMMObservationMode(hmm_observation_mode)
        self.out_dir = out_dir
        self.random_state = random_state
        self.segment_results = {}
        self.default_sample_info: SampleInfo | None = None

    def _encode_states_for_hmm(self, states: np.ndarray) -> np.ndarray:
        numeric_states = MethylationStates.convert_to_numeric(states)
        unique_states = np.sort(np.unique(numeric_states))
        supported_observations = getattr(
            self.hmm_model,
            "n_emissions",
            getattr(self.hmm_model, "n_states", None),
        )
        if supported_observations is not None and len(unique_states) > int(
            supported_observations
        ):
            raise ValueError(
                "Observed state labels contain more distinct categories than the "
                "configured HMM can represent."
            )

        state_to_obs = {
            int(state_value): obs_idx
            for obs_idx, state_value in enumerate(unique_states.tolist())
        }
        return np.array(
            [state_to_obs[int(state_value)] for state_value in numeric_states],
            dtype=int,
        )

    def _get_state_cutoffs(self) -> Optional[Dict[str, object]]:
        return getattr(self.analyzer, "state_cutoffs", None)

    def _prepare_emissions(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        meth_data, emissions_df = (
            self.analyzer.assigner.prepare_sample_for_clustering(
                sample_info=sample_info,
                chrom=chrom,
            )
        )
        self.meth_data = meth_data.copy()
        self.emissions_df = emissions_df.copy()
        return self.meth_data, self.emissions_df

    def _derive_sequence_lengths(
        self,
        meth_data: pd.DataFrame,
        chrom: str | None = None,
    ) -> Optional[List[int]]:
        if chrom is not None or len(meth_data) == 0:
            return None

        chrom_values = meth_data["CpG_chrm"].astype(str).to_numpy()
        lengths = []
        current_chrom = chrom_values[0]
        current_length = 1
        for chrom_name in chrom_values[1:]:
            if chrom_name == current_chrom:
                current_length += 1
            else:
                lengths.append(current_length)
                current_chrom = chrom_name
                current_length = 1
        lengths.append(current_length)
        return lengths if len(lengths) > 1 else None

    def _set_hmm_sequence_lengths(
        self,
        lengths: Optional[List[int]],
    ) -> None:
        if not hasattr(self.hmm_model, "lengths"):
            return
        self.hmm_model.lengths = (
            None if lengths is None else [int(length) for length in lengths]
        )

    def _prepare_gaussian_feature_matrix(
        self,
        emission_df: pd.DataFrame,
    ) -> np.ndarray:
        feature_cols = self.analyzer.assigner.resolve_feature_cols(emission_df)
        x_scaled, _imputer, _scaler = (
            self.analyzer.assigner.preprocess_emission_features(
                emission_df=emission_df,
                feature_cols=feature_cols,
                fit=True,
            )
        )
        return x_scaled

    def _get_gaussian_init_labels(
        self,
        emission_df: pd.DataFrame,
        X_scaled: np.ndarray,
    ) -> np.ndarray:
        assigner = self.analyzer.assigner
        if hasattr(assigner, "model") and getattr(assigner, "model", None) is not None:
            try:
                _, _, raw_labels, _ = assigner.apply_kmeans_to_emissions(emission_df)
                return np.asarray(raw_labels, dtype=int)
            except Exception:
                pass

        temp_kmeans = KMeans(
            n_clusters=self.hmm_model.n_states,
            n_init=10,
            random_state=self.random_state,
        )
        return temp_kmeans.fit_predict(X_scaled)

    def _prepare_pca_feature_matrix(
        self,
        emission_df: pd.DataFrame,
    ) -> np.ndarray:
        assigner = self.analyzer.assigner
        feature_cols = assigner.resolve_feature_cols(emission_df)

        if hasattr(assigner, "model") and getattr(assigner, "model", None) is not None:
            model = assigner.model
            X_scaled = assigner.preprocess_emission_features(
                emission_df=emission_df,
                feature_cols=model.feature_cols,
                fit=False,
            )
            if model.pca is not None:
                return model.pca.transform(X_scaled)

        X_scaled = self._prepare_gaussian_feature_matrix(emission_df)
        if assigner.n_pca is None or assigner.n_pca <= 0:
            raise ValueError(
                "PCA-emission observation mode requires n_pca to be a positive integer."
            )

        n_components = min(assigner.n_pca, X_scaled.shape[0], X_scaled.shape[1])
        if n_components <= 0:
            raise ValueError(
                "Cannot fit PCA-emission features because the emission matrix is empty."
            )

        return PCA(
            n_components=n_components,
            random_state=self.random_state,
        ).fit_transform(X_scaled)

    def _get_pca_init_labels(
        self,
        emission_df: pd.DataFrame,
        X_pca: np.ndarray,
    ) -> np.ndarray:
        assigner = self.analyzer.assigner
        if (
            hasattr(assigner, "model")
            and getattr(assigner, "model", None) is not None
            and getattr(assigner.model, "pca", None) is not None
        ):
            try:
                _, _, raw_labels, _ = assigner.apply_kmeans_to_emissions(emission_df)
                return np.asarray(raw_labels, dtype=int)
            except Exception:
                pass

        temp_kmeans = KMeans(
            n_clusters=self.hmm_model.n_states,
            n_init=10,
            random_state=self.random_state,
        )
        return temp_kmeans.fit_predict(X_pca)

    def _segment_sample_discrete_states(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.assign_states(sample_info, chrom)
        states = self._encode_states_for_hmm(
            self.meth_data["state"].to_numpy(dtype=int)
        )
        sequence_lengths = self._derive_sequence_lengths(self.meth_data, chrom=chrom)
        self._set_hmm_sequence_lengths(sequence_lengths)

        self.hmm_model.create_model()
        self.hmm_model.fit(states, sample_info, chrom)
        hidden_states = self.hmm_model.predict(states)
        readable_states = relabel_by_mean_emission(
            hidden_states,
            self.emissions_df,
            self._get_state_cutoffs(),
            self.analyzer.assigner.int_low_cutoff,
            self.analyzer.assigner.int_high_cutoff,
            self.analyzer.assigner.window_specs,
        )
        return hidden_states, readable_states

    def _segment_sample_gaussian_emissions(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._prepare_emissions(sample_info=sample_info, chrom=chrom)
        X_scaled = self._prepare_gaussian_feature_matrix(self.emissions_df)
        km_labels = self._get_gaussian_init_labels(self.emissions_df, X_scaled)

        init_readable_states = relabel_by_mean_emission(
            km_labels,
            self.emissions_df,
            self._get_state_cutoffs(),
            self.analyzer.assigner.int_low_cutoff,
            self.analyzer.assigner.int_high_cutoff,
            self.analyzer.assigner.window_specs,
        )
        self.meth_data["state"] = MethylationStates.convert_to_numeric(
            init_readable_states
        )
        self.meth_data["state_readable"] = init_readable_states

        sequence_lengths = self._derive_sequence_lengths(self.meth_data, chrom=chrom)

        if not hasattr(self.hmm_model, "initialize_from_kmeans"):
            raise ValueError(
                "Gaussian-emission observation mode requires an HMM model that "
                "supports KMeans-based initialization."
            )

        self.hmm_model.create_model()
        self.hmm_model.initialize_from_kmeans(
            X_scaled=X_scaled,
            km_labels=km_labels,
            lengths=sequence_lengths,
        )
        self.hmm_model.fit(X_scaled, sample_info, chrom)
        hidden_states = self.hmm_model.predict(X_scaled)
        readable_states = relabel_by_mean_emission(
            hidden_states,
            self.emissions_df,
            self._get_state_cutoffs(),
            self.analyzer.assigner.int_low_cutoff,
            self.analyzer.assigner.int_high_cutoff,
            self.analyzer.assigner.window_specs,
        )
        return hidden_states, readable_states

    def assign_states(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> np.ndarray:
        meth_data, emissions_df = self._prepare_emissions(
            sample_info=sample_info, chrom=chrom
        )
        if (
            self.state_assignment_method.value
            == MethylStateAssignmentMethod.DEFINITION.value
        ):
            states = self.analyzer.define_states_by_rules(
                sample_info=sample_info,
                chrom=chrom,
                sample_emissions=emissions_df,
            )
        elif (
            self.state_assignment_method.value
            == MethylStateAssignmentMethod.KMEANS.value
        ):
            _, _, _, states = self.analyzer.assigner.apply_kmeans_to_emissions(
                emissions_df
            )
        elif (
            self.state_assignment_method.value == MethylStateAssignmentMethod.AUTO.value
        ):
            raise NotImplementedError(
                "AUTO state assignment method not implemented yet."
            )
        else:
            raise ValueError(
                f"Unknown state assignment method: {self.state_assignment_method}"
            )
        meth_data = meth_data.copy()
        meth_data["state"] = MethylationStates.convert_to_numeric(states)
        meth_data["state_readable"] = states

        self.meth_data = meth_data
        self.emissions_df = emissions_df

        return meth_data, emissions_df

    def _segment_sample_pca_emissions(
        self,
        sample_info: SampleInfo,
        chrom: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._prepare_emissions(sample_info=sample_info, chrom=chrom)
        X_pca = self._prepare_pca_feature_matrix(self.emissions_df)
        km_labels = self._get_pca_init_labels(self.emissions_df, X_pca)

        init_readable_states = relabel_by_mean_emission(
            km_labels,
            self.emissions_df,
            self._get_state_cutoffs(),
            self.analyzer.assigner.int_low_cutoff,
            self.analyzer.assigner.int_high_cutoff,
            self.analyzer.assigner.window_specs,
        )
        self.meth_data["state"] = MethylationStates.convert_to_numeric(
            init_readable_states
        )
        self.meth_data["state_readable"] = init_readable_states

        sequence_lengths = self._derive_sequence_lengths(self.meth_data, chrom=chrom)

        if not hasattr(self.hmm_model, "initialize_from_kmeans"):
            raise ValueError(
                "PCA-emission observation mode requires an HMM model that "
                "supports KMeans-based initialization."
            )

        self.hmm_model.create_model()
        self.hmm_model.initialize_from_kmeans(
            X_scaled=X_pca,
            km_labels=km_labels,
            lengths=sequence_lengths,
        )
        self.hmm_model.fit(X_pca, sample_info, chrom)
        hidden_states = self.hmm_model.predict(X_pca)
        readable_states = relabel_by_mean_emission(
            hidden_states,
            self.emissions_df,
            self._get_state_cutoffs(),
            self.analyzer.assigner.int_low_cutoff,
            self.analyzer.assigner.int_high_cutoff,
            self.analyzer.assigner.window_specs,
        )
        return hidden_states, readable_states

    def segment_sample(
        self,
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
        force_resegment: bool = False,
    ) -> Tuple[pd.DataFrame, object]:
        """
        Segment a sample and refresh probe-level results plus raw regions.

        Returns the segmented probe-level methylation table and fitted HMM
        object. Raw contiguous regions are stored on ``self.regions_df``.
        """
        if sample_info is None:
            sample_info = self.default_sample_info
        if sample_info is None:
            raise ValueError(
                "No sample_info provided and no default_sample_info configured."
            )
        chrom_segmented_on_sample = (
            sample_info.sample_id in self.segment_results
            and chrom in self.segment_results[sample_info.sample_id]
        )
        if not chrom_segmented_on_sample or force_resegment:
            if (
                self.hmm_observation_mode.value
                == HMMObservationMode.DISCRETE_STATES.value
            ):
                hidden_states, readable_states = self._segment_sample_discrete_states(
                    sample_info=sample_info,
                    chrom=chrom,
                )
            elif (
                self.hmm_observation_mode.value
                == HMMObservationMode.GAUSSIAN_EMISSIONS.value
            ):
                hidden_states, readable_states = (
                    self._segment_sample_gaussian_emissions(
                        sample_info=sample_info,
                        chrom=chrom,
                    )
                )
            elif (
                self.hmm_observation_mode.value
                == HMMObservationMode.PCA_EMISSIONS.value
            ):
                hidden_states, readable_states = self._segment_sample_pca_emissions(
                    sample_info=sample_info,
                    chrom=chrom,
                )
            else:
                raise ValueError(
                    f"Unknown HMM observation mode: {self.hmm_observation_mode}"
                )

            self.segment_results.setdefault(sample_info.sample_id, {})
            self.segment_results[sample_info.sample_id][chrom] = {
                "meth_data": self.meth_data.copy(),
                "emissions_df": self.emissions_df.copy(),
                "hmm_state": hidden_states,
                "hmm_state_readable": readable_states,
            }
        else:
            self.meth_data = self.segment_results[sample_info.sample_id][chrom][
                "meth_data"
            ].copy()
            self.emissions_df = self.segment_results[sample_info.sample_id][chrom][
                "emissions_df"
            ].copy()
            hidden_states = self.segment_results[sample_info.sample_id][chrom][
                "hmm_state"
            ]
            readable_states = self.segment_results[sample_info.sample_id][chrom][
                "hmm_state_readable"
            ]

        cache_entry = self.segment_results[sample_info.sample_id][chrom]

        # Attach HMM states and refresh the raw contiguous regions.
        self.meth_data["hmm_state"] = hidden_states
        self.meth_data["hmm_state_readable"] = readable_states
        self.regions_df = self.create_regions(
            state_col="hmm_state_readable",
            region_min_probes=1,
        )
        cache_entry["meth_data"] = self.meth_data.copy()
        cache_entry["regions_df"] = self.regions_df.copy()

        # print(f"State relabeling completed in {time.time() - start:.2f} seconds.")

        return self.meth_data, self.hmm_model.hmm_model

    def create_regions(self, state_col="hmm_state_readable", region_min_probes=1):
        """
        Create regions (start, end) for contiguous segments of the same state.

        Parameters
        ----------
        meth_data : DataFrame
            Must contain 'CpG_chrm', 'CpG_beg', 'CpG_end', and state_col.
        state_col : str
            Column name for the state labels.
        region_min_probes : int
            Minimum number of probes required to form a region.
        Returns
        -------
        regions_df : DataFrame
            Columns: 'CpG_chrm', 'start', 'end', state_col
        """

        for col in ["CpG_chrm", "CpG_beg", "CpG_end", state_col]:
            if col not in self.meth_data.columns:
                raise ValueError(f"Column {col} not found in meth_data.")

        regions = []
        current_chrom = None
        current_state = None
        region_start = None
        region_end = None
        current_meth_sum = 0
        current_probe_count = 0

        for idx, row in self.meth_data.iterrows():
            chrom = row["CpG_chrm"]
            state = row[state_col]
            beg = row["CpG_beg"]
            end = row["CpG_end"]
            beta_val = float(row["beta"])

            if (chrom != current_chrom) or (state != current_state):
                # Save previous region
                if (
                    current_chrom is not None
                    and current_probe_count >= region_min_probes
                ):
                    regions.append(
                        {
                            "CpG_chrm": current_chrom,
                            "start": region_start,
                            "end": region_end,
                            "avg_beta": current_meth_sum / current_probe_count,
                            "probe_count": current_probe_count,
                            "state": current_state,
                        }
                    )
                # Start new region
                current_chrom = chrom
                current_state = state
                region_start = beg
                region_end = end
                current_meth_sum = beta_val
                current_probe_count = 1
            else:
                # Extend current region
                region_end = end
                current_meth_sum += beta_val
                current_probe_count += 1
        # Save last region
        if current_chrom is not None and current_probe_count >= region_min_probes:
            regions.append(
                {
                    "CpG_chrm": current_chrom,
                    "start": region_start,
                    "end": region_end,
                    "avg_beta": current_meth_sum / current_probe_count,
                    "probe_count": current_probe_count,
                    "state": current_state,
                }
            )
        regions_df = pd.DataFrame(
            regions,
            columns=["CpG_chrm", "start", "end", "avg_beta", "probe_count", "state"],
        )
        self.regions_df = regions_df
        return regions_df

    def regions_to_bed(self, bed_path: str, separate_beds_by_state: bool = False):
        """
        Save regions DataFrame to BED file.

        Parameters
        ----------
        regions_df : DataFrame
            Must contain 'CpG_chrm', 'start', 'end', and 'state'.
        bed_path : str
            Output path for the BED file.
        """
        if not bed_path.lower().endswith(".bed"):
            bed_path += ".bed"
        regions_df = self.regions_df.copy()
        regions_df["start"] = regions_df["start"].astype(int)
        regions_df["end"] = regions_df["end"].astype(int)
        if not separate_beds_by_state:
            bed_df = regions_df[["CpG_chrm", "start", "end", "state"]].copy()
            bed_df.to_csv(bed_path, sep="\t", header=False, index=False)
        else:
            for state in MethylationStates:
                state_df = regions_df[regions_df["state"] == state]
                bed_df = state_df[["CpG_chrm", "start", "end", "state"]].copy()
                state_bed_path = bed_path.replace(".bed", f"_{state.name}.bed")
                bed_df.to_csv(state_bed_path, sep="\t", header=False, index=False)

    def plot_interactive_beta_by_label(
        self,
        sample_info: SampleInfo | None = None,
        sample_info_removed: pd.DataFrame | None = None,
        label_type: str = "hmm",
        chrom: str | None = None,
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,
        color_pmd_only: bool = False,
        color_regions_df: pd.DataFrame | None = None,
    ):
        overlay_regions_df, overlay_style = resolve_overlay_plot_args(
            color_pmd_only=color_pmd_only,
            color_regions_df=color_regions_df,
        )
        return self.plot_labels(
            sample_info=sample_info,
            chrom=chrom,
            sample_info_removed=sample_info_removed,
            overlay_regions_df=overlay_regions_df,
            overlay_style=overlay_style,
            x_col=x_col,
            y_col=y_col,
            label_title=label_title,
            show_plot=show_plot,
            max_points=max_points,
        )

    def plot_labels(
        self,
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
        sample_info_removed: pd.DataFrame | None = None,
        overlay_regions_df: pd.DataFrame | None = None,
        overlay_style: str = "state",
        region_start: int | None = None,
        region_end: int | None = None,
        region_chrom: str | None = None,
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,
    ):
        """
        Plot genomic-position vs beta for HMM labels.
        """
        meth_data, _ = self.segment_sample(sample_info=sample_info, chrom=chrom)
        resolved_sample_info = (
            self.default_sample_info if sample_info is None else sample_info
        )
        overlay_regions_df, resolved_overlay_style = resolve_region_overlay_df(
            overlay_regions_df=overlay_regions_df,
            region_start=region_start,
            region_end=region_end,
            region_chrom=region_chrom,
        )
        return plot_interactive_beta_scatter(
            df_plot=meth_data.copy(),
            sample_info=resolved_sample_info,
            sample_info_removed=sample_info_removed,
            chrom=chrom,
            out_dir=self.out_dir,
            label_col="hmm_state_readable",
            x_col=x_col,
            y_col=y_col,
            label_title=label_title if label_title is not None else "HMM state",
            show_plot=show_plot,
            max_points=max_points,
            overlay_regions_df=overlay_regions_df,
            overlay_style=(
                resolved_overlay_style
                if overlay_regions_df is not None
                else overlay_style
            ),
            region_start=region_start,
            region_end=region_end,
            region_chrom=region_chrom,
        )
