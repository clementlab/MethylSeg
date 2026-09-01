"""Utilities for analyzing, comparing, and tuning methylation state assignments."""

from enum import Enum

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
import seaborn as sns

from .methyl_state_assigner import MethylStateAssigner
from .methylseg_config import MethylSegConfig
from .helper_classes import (
    MethylationStates,
    SampleInfo,
)
from .utils import (
    get_biological_state_colors,
    get_regional_window_labels,
    normalize_state_label,
    plot_interactive_beta_scatter,
    relabel_by_mean_emission,
    resolve_overlay_plot_args,
    resolve_region_overlay_df,
)


class MethylStateAnalyzer:
    """Inspect training outputs and optimize rule-based state cutoffs."""

    def __init__(self, assigner: MethylStateAssigner, out_dir="."):
        self.assigner = assigner
        self.out_dir = out_dir
        self.train_joint = None
        self.window_specs = assigner.window_specs

    def _populate_kmeans_state_display(self, train_joint: pd.DataFrame) -> pd.DataFrame:
        train_joint = train_joint.copy()
        if "kmeans_label" not in train_joint.columns:
            train_joint["kmeans_label"] = self.assigner.train_labels
        if hasattr(self.assigner, "model"):
            _, _, raw_labels, _ = self.assigner.apply_kmeans_to_emissions(
                self.assigner.train_emission_df.copy()
            )
            train_joint["kmeans_state_display"] = relabel_by_mean_emission(
                raw_labels=raw_labels,
                emission_df=self.assigner.train_emission_df.copy(),
                state_cutoffs=getattr(self, "state_cutoffs", None),
                int_low_cutoff=self.assigner.int_low_cutoff,
                int_high_cutoff=self.assigner.int_high_cutoff,
                window_specs=self.assigner.window_specs,
            )
        else:
            train_joint["kmeans_state_display"] = train_joint["kmeans_label"]
        train_joint["kmeans_state_display"] = train_joint["kmeans_state_display"].apply(
            normalize_state_label
        )
        return train_joint

    def _build_train_joint(self):
        if self.train_joint is not None:
            if (
                "kmeans_label" not in self.train_joint.columns
                or "kmeans_state_display" not in self.train_joint.columns
            ):
                self.train_joint = self._populate_kmeans_state_display(self.train_joint)
            return
        if not hasattr(self.assigner, "model"):
            raise ValueError("No trained model found. Please train a model first.")
        train_joint = pd.concat(
            [self.assigner.train_meth.copy(), self.assigner.train_emission_df.copy()],
            axis=1,
        )
        train_joint = train_joint.loc[:, ~train_joint.columns.duplicated()]
        self.train_joint = self._populate_kmeans_state_display(train_joint)

    def plot_feature_distributions_by_kmeans_state(
        self,
        show_plots=True,
        state_colors: dict | None = None,
    ):
        """
        Plot per-feature histograms stratified by learned KMeans state.

        Parameters
        ----------
        show_plots
            If ``True``, display each figure immediately. Otherwise, save plots
            to ``out_dir`` when configured.
        """
        self._build_train_joint()
        train_loadings = self.assigner.get_pca_loadings()
        ranked_features = list(
            train_loadings["PC2"].abs().sort_values(ascending=False).index
        )
        if "beta" in train_loadings.index:
            ranked_features = ["beta"] + [
                feature for feature in ranked_features if feature != "beta"
            ]
        _, _, _, state_colors_hex = get_biological_state_colors(
            state_colors=state_colors
        )
        ordered_states = [state.name for state in MethylationStates]
        for emission in ranked_features:
            fig, ax = plt.subplots()
            plotted = False
            for state_name in ordered_states:
                df = self.train_joint.loc[
                    self.train_joint["kmeans_state_display"].eq(state_name)
                ]
                if df.empty:
                    continue
                df[emission].hist(
                    bins=50,
                    alpha=0.5,
                    label=state_name,
                    color=state_colors_hex[MethylationStates[state_name].value],
                    ax=ax,
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel(emission)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {emission} by KMeans State")
            ax.legend()
            fig.tight_layout()
            if show_plots:
                plt.show()
            elif self.out_dir is not None:
                fig.savefig(f"{self.out_dir}/feature_distribution_{emission}.png")
            plt.close(fig)

    def define_states_by_rules_param(
        self,
        meth_emissions: pd.DataFrame,
        beta_low_max: float,
        beta_high_min: float,
        pmd_cutoffs: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        Apply rule-based state labels with tunable per-window PMD cutoffs.

        Parameters
        ----------
        meth_emissions
            Emission table containing ``beta`` and the per-window summary columns
            used by the PMD rules.
        beta_low_max
            Upper beta threshold for the low-methylation regime.
        beta_high_min
            Lower beta threshold for the high-methylation regime.
        pmd_cutoffs
            Mapping of window label to rule cutoffs with ``int_min``, ``std_max``,
            ``high_max``, and ``low_max`` entries.

        Returns
        -------
        numpy.ndarray
            Array of ``MethylationStates`` values for each input row.
        """
        beta = meth_emissions["beta"].values
        n = len(meth_emissions)
        labels = np.full(len(meth_emissions), None, dtype=object)

        if not hasattr(self, "assigner"):
            raise AttributeError(
                "MethylStateAnalyzer must have an `assigner` with `window_specs`."
            )

        window_labels = [label for _, label in self.assigner.window_specs]
        if not window_labels:
            raise ValueError("No window_specs found on assigner.")
        regional_window_labels = get_regional_window_labels(self.window_specs)

        pmd_window_masks = []

        for label in regional_window_labels:
            if label not in pmd_cutoffs:
                raise KeyError(
                    f"No PMD cutoffs provided for window '{label}'. "
                    f"Expected a key in pmd_cutoffs for each label in window_specs."
                )

            cfg = pmd_cutoffs[label]
            try:
                int_min = cfg["int_min"]
                std_max = cfg["std_max"]
                high_max = cfg["high_max"]
                low_max = cfg.get("low_max", high_max)
            except KeyError as e:
                raise KeyError(
                    f"PMD cutoff for window '{label}' must contain "
                    f"keys 'int_min', 'std_max', 'high_max'. Missing: {e}"
                )

            int_col = f"{label}_int_pct"
            std_col = f"{label}_std"
            high_col = f"{label}_high_pct"
            low_col = f"{label}_low_pct"

            if int_col not in meth_emissions.columns:
                raise KeyError(f"Column '{int_col}' not found in meth_emissions.")
            if std_col not in meth_emissions.columns:
                raise KeyError(f"Column '{std_col}' not found in meth_emissions.")
            if high_col not in meth_emissions.columns:
                raise KeyError(f"Column '{high_col}' not found in meth_emissions.")
            if low_col not in meth_emissions.columns:
                raise KeyError(f"Column '{low_col}' not found in meth_emissions.")

            int_vals = meth_emissions[int_col].values
            std_vals = meth_emissions[std_col].values
            high_vals = meth_emissions[high_col].values
            low_vals = meth_emissions[low_col].values

            pmd_window_masks.append(
                (int_vals >= int_min)
                & (std_vals <= std_max)
                & (high_vals <= high_max)
                & (low_vals <= low_max)
            )

        regional_any = np.logical_or.reduce(pmd_window_masks)
        pmd_mask = regional_any & (beta >= beta_low_max) & (beta <= beta_high_min)

        low_mask = (beta <= beta_low_max) & ~pmd_mask
        high_mask = (beta >= beta_high_min) & ~pmd_mask
        interm_mask = ~(pmd_mask | low_mask | high_mask)

        labels[low_mask] = MethylationStates.LOW
        labels[pmd_mask] = MethylationStates.PMD
        labels[interm_mask] = MethylationStates.INTERMEDIATE
        labels[high_mask] = MethylationStates.HIGH

        return labels

    def evaluate_rules_against_kmeans(
        self,
        meth_emissions: pd.DataFrame,
        kmeans_labels: np.ndarray = None,
        **rule_params,
    ):
        """
        Apply rule-based states and compare to KMeans labels.
        Returns (metrics_dict, rule_labels_array).

        rule_params must contain:
          - beta_low_max
          - beta_high_min
          - pmd_cutoffs (dict[label -> {'int_min','std_max','high_max','low_max'}])
        """
        y_true = np.asarray(kmeans_labels)
        y_pred = self.define_states_by_rules_param(meth_emissions, **rule_params)

        if isinstance(y_true.flat[0], Enum):
            y_true = np.array([lbl.value for lbl in y_true])
        if isinstance(y_pred.flat[0], Enum):
            y_pred_numeric = np.array([lbl.value for lbl in y_pred])
        else:
            y_pred_numeric = y_pred

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred_numeric),
            "F1_macro": f1_score(
                y_true, y_pred_numeric, average="macro", zero_division=0
            ),
            "F1_weighted": f1_score(
                y_true, y_pred_numeric, average="weighted", zero_division=0
            ),
            "Precision_macro": precision_score(
                y_true, y_pred_numeric, average="macro", zero_division=0
            ),
            "Recall_macro": recall_score(
                y_true, y_pred_numeric, average="macro", zero_division=0
            ),
            "ARI": adjusted_rand_score(y_true, y_pred_numeric),
            "NMI": normalized_mutual_info_score(y_true, y_pred_numeric),
        }

        return metrics, y_pred

    # TODO: improve the rules definition logic to improve accuracy when based against k-means, e.g. by adding interaction terms or more complex combinations of stats across windows.
    def optimize_rule_params_random(
        self,
        n_iter: int = 500,
        score_key: str = "F1_macro",
        random_state: int = 42,
        param_distributions: dict | None = None,
    ):
        """
        Tune rule-based cutoffs with random search against KMeans labels.

        Parameters
        ----------
        n_iter
            Number of random parameter draws to evaluate.
        score_key
            Metric name from :meth:`evaluate_rules_against_kmeans` used to pick
            the best rule set.
        random_state
            Seed for reproducible sampling.
        param_distributions
            Optional nested dictionary describing the search ranges. When omitted,
            defaults are created for each window label in ``assigner.window_specs``.
        """
        self._build_train_joint()
        rng = np.random.default_rng(random_state)

        # Build default distributions if none provided
        if param_distributions is None:
            window_labels = [label for _, label in self.assigner.window_specs]
            pmd_dist = {
                label: {
                    "int_min": (0.40, 0.90),
                    "std_max": (0.10, 0.40),
                    "high_max": (0.05, 0.40),
                    "low_max": (0.05, 0.40),
                }
                for label in window_labels
            }
            param_distributions = {
                "beta_low_max": (0.05, 0.35),
                "beta_high_min": (0.60, 0.95),
                "pmd": pmd_dist,
            }

        # Helper: sample a valid param set
        def sample_params():
            while True:
                beta_low_max = float(rng.uniform(*param_distributions["beta_low_max"]))
                beta_high_min = float(
                    rng.uniform(*param_distributions["beta_high_min"])
                )
                if beta_low_max >= beta_high_min:
                    continue

                pmd_cutoffs = {}
                for label, ranges in param_distributions["pmd"].items():
                    pmd_cutoffs[label] = {
                        "int_min": float(rng.uniform(*ranges["int_min"])),
                        "std_max": float(rng.uniform(*ranges["std_max"])),
                        "high_max": float(rng.uniform(*ranges["high_max"])),
                        "low_max": float(rng.uniform(*ranges["low_max"])),
                    }

                return {
                    "beta_low_max": beta_low_max,
                    "beta_high_min": beta_high_min,
                    "pmd_cutoffs": pmd_cutoffs,
                }

        def flatten_rule_params(params: dict) -> dict:
            out = {
                "beta_low_max": params["beta_low_max"],
                "beta_high_min": params["beta_high_min"],
            }
            for label, cfg in params["pmd_cutoffs"].items():
                out[f"{label}_int_min"] = cfg["int_min"]
                out[f"{label}_std_max"] = cfg["std_max"]
                out[f"{label}_high_max"] = cfg["high_max"]
                out[f"{label}_low_max"] = cfg["low_max"]
            return out

        best_score = -np.inf
        best_params = None
        best_metrics = None
        best_labels = None
        history = []

        for i in tqdm(range(n_iter), desc="Random search"):
            rule_params = sample_params()

            metrics, labels = self.evaluate_rules_against_kmeans(
                self.assigner.train_emission_df,
                self.train_joint["kmeans_label"],
                **rule_params,
            )
            score = metrics[score_key]

            record = {"iter": i, "score": score}
            record.update(flatten_rule_params(rule_params))
            history.append(record)

            if score > best_score:
                best_score = score
                best_params = rule_params
                best_metrics = metrics
                best_labels = labels

        history_df = pd.DataFrame(history)
        self.cutoffs_set_manually = False
        self.state_cutoffs = best_params

        states_by_rules = self.define_states_by_rules(
            sample_info=self.assigner.train_sample_info,
            sample_emissions=self.assigner.train_emission_df,
        )
        self.train_joint["rule_based_label"] = states_by_rules

        best_configs = pd.DataFrame([flatten_rule_params(best_params)])
        best_results = pd.DataFrame(best_metrics, index=[0])
        optimization_summary = pd.concat([best_configs, best_results], axis=1)
        if self.out_dir is not None:
            optimization_summary.to_csv(
                f"{self.out_dir}/rule_based_optimization_summary.csv", index=False
            )
        return best_params, best_metrics, best_labels, history_df

    def define_states_by_rules(
        self, sample_info: SampleInfo, chrom=None, sample_emissions: pd.DataFrame = None
    ) -> np.ndarray:
        """
        Apply the current rule-based cutoff set to a sample or emission table.

        Parameters
        ----------
        sample_info
            Prepared methylation sample used when ``sample_emissions`` is not
            supplied.
        chrom
            Optional chromosome restriction passed through to emission
            preparation.
        sample_emissions
            Precomputed emission table to label directly.

        Returns
        -------
        numpy.ndarray
            Rule-based ``MethylationStates`` assignments for each emission row.

        Raises
        ------
        ValueError
            If rule cutoffs have not been defined.
        """
        if sample_emissions is not None:
            meth_emissions = sample_emissions
        else:
            test_meth, test_emissions = (
                self.assigner.prepare_sample_for_clustering(sample_info, chrom)
            )
            meth_emissions = test_emissions

        if not hasattr(self, "state_cutoffs"):
            raise ValueError(
                "State cutoffs not defined. Please run optimization or set cutoffs manually."
            )

        states_by_rules = self.define_states_by_rules_param(
            meth_emissions, **self.state_cutoffs
        )
        return states_by_rules

    def __set_from_config(self, state_cfg: dict | None):
        if state_cfg is not None:
            cutoffs = state_cfg
            self.set_state_cutoffs(
                beta_low_max=cutoffs.get("beta_low_max"),
                beta_high_min=cutoffs.get("beta_high_min"),
                pmd_cutoffs=cutoffs.get("pmd_cutoffs"),
            )

    def set_state_cutoffs_from_yaml(self, yaml_file: str):
        """
        Load rule cutoffs from a YAML file written by ``MethylSegConfig``.
        """
        config = MethylSegConfig.from_yaml(yaml_file)
        self.__set_from_config(config.get_state_cutoffs())
        self.cutoffs_set_manually = bool(
            config.config.get("state_cutoffs_set_manually", False)
        )

    def set_state_cutoffs(
        self,
        beta_low_max: float | None = None,
        beta_high_min: float | None = None,
        pmd_cutoffs: dict | None = None,
    ):
        """
        Simple manual state cutoff setter with clean pythonic defaults.

        - If user does not provide pmd_cutoffs, defaults are used:
            int_min = 0.56
            std_max = 0.264
            high_max = 0.246
            low_max = 0.246
        - Defaults apply to every window given in assigner.window_specs
        """
        int_min_default = 0.56
        std_max_default = 0.264
        high_max_default = 0.246
        low_max_default = 0.246
        beta_high_max_default = 0.694
        beta_low_max_default = 0.290
        if beta_low_max is None:
            beta_low_max = beta_low_max_default
        if beta_high_min is None:
            beta_high_min = beta_high_max_default

        final_pmd_cutoffs = {}

        for _, label in self.assigner.window_specs:
            cfg = (pmd_cutoffs or {}).get(label, {})
            final_pmd_cutoffs[label] = {
                "int_min": cfg.get("int_min", int_min_default),
                "std_max": cfg.get("std_max", std_max_default),
                "high_max": cfg.get("high_max", high_max_default),
                "low_max": cfg.get("low_max", low_max_default),
            }

        # Save all cutoffs
        self.state_cutoffs = {
            "beta_low_max": float(beta_low_max),
            "beta_high_min": float(beta_high_min),
            "pmd_cutoffs": final_pmd_cutoffs,
        }

        self.cutoffs_set_manually = True

    def pretty_print_rules(self):
        """
        Print the active rule-based state definitions in a compact form.
        """
        if not hasattr(self, "state_cutoffs"):
            raise ValueError(
                "State cutoffs not defined. Please run optimization or set cutoffs manually."
            )

        c = self.state_cutoffs
        beta_low_max = c["beta_low_max"]
        beta_high_min = c["beta_high_min"]
        pmd_cutoffs = c["pmd_cutoffs"]
        regional_window_labels = get_regional_window_labels(self.window_specs)

        print("PMD:")
        print(f"{beta_low_max:.3f} <= beta <= {beta_high_min:.3f}")

        regional_parts = []
        for label in regional_window_labels:
            cfg = pmd_cutoffs[label]
            low_max = cfg.get("low_max", cfg["high_max"])
            regional_parts.append(
                "("
                f"{label}_int_pct >= {cfg['int_min']:.3f} AND "
                f"{label}_std <= {cfg['std_max']:.3f} AND "
                f"{label}_high_pct <= {cfg['high_max']:.3f} AND "
                f"{label}_low_pct <= {low_max:.3f}"
                ")"
            )
        print("AND " + " OR ".join(regional_parts) + "\n")

        print("Low methylation:")
        print(f"beta <= {beta_low_max:.3f} AND NOT PMD\n")

        print("Intermediate methylation:")
        print(f"{beta_low_max:.3f} < beta < {beta_high_min:.3f} AND NOT PMD\n")

        print("High methylation:")
        print(f"beta >= {beta_high_min:.3f} AND NOT PMD\n")

    def evaluate_clustering_concordance(
        self,
        use_train_data: bool = True,
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
    ):
        """
        Evaluate rule-based labels against KMeans cluster labels as ground truth.
        """
        if not hasattr(self, "state_cutoffs"):
            raise ValueError(
                "State cutoffs not defined. Please run optimization or set cutoffs manually."
            )
        if use_train_data:
            self._build_train_joint()
            y_true = self.train_joint["kmeans_label"]
            y_pred = self.train_joint["rule_based_label"]
        else:
            y_true = self.assigner.apply_kmeans_to_sample(
                sample_info=sample_info, chrom=chrom
            )[5]
            y_pred = self.define_states_by_rules(sample_info=sample_info, chrom=chrom)

        # --- Ensure numpy arrays ---
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # --- Convert Enum → integer for sklearn ---
        if isinstance(y_true[0], Enum):
            y_true = np.array([lbl.value for lbl in y_true])
        if isinstance(y_pred[0], Enum):
            y_pred_numeric = np.array([lbl.value for lbl in y_pred])
        else:
            y_pred_numeric = y_pred

        # ⬇️ Detailed per-class performance
        print("\nClassification Report:")
        target_names = [s.name for s in MethylationStates]
        print(
            classification_report(
                y_true, y_pred_numeric, target_names=target_names, zero_division=0
            )
        )

        # ⬇️ Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred_numeric)
        cm_df = pd.DataFrame(
            cm,
            index=[label.name for label in MethylationStates],
            columns=[label.name for label in MethylationStates],
        )

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix — KMeans (True) vs Rule-Based (Pred)")
        plt.ylabel("True (KMeans)")
        plt.xlabel("Predicted (Rule-Based)")
        plt.tight_layout()
        plt.show()

        return cm_df

    def plot_interactive_beta_by_label(
        self,
        label_type: str = "kmeans",
        use_train_data: bool = True,
        chrom: str | None = None,
        sample_info: SampleInfo | None = None,
        sample_info_removed: pd.DataFrame | None = None,
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,
        color_pmd_only: bool = False,
        color_regions_df: pd.DataFrame | None = None,
    ):
        """
        Backward-compatible wrapper for interactive analyzer-owned label plots.

        Parameters
        ----------
        label_type
            Label source to visualize, typically ``"kmeans"`` or
            ``"rule_based"``.
        use_train_data
            If ``True``, plot cached training labels instead of a new sample.
        chrom
            Optional chromosome restriction when plotting a new sample.
        sample_info
            Sample to analyze when ``use_train_data`` is ``False``.
        sample_info_removed
            Optional table of filtered CpGs to overlay as removed points.
        x_col, y_col
            Probe-level columns used for the scatter plot axes.
        label_title
            Legend title override.
        show_plot
            If ``True``, display the plot immediately.
        max_points
            Maximum number of probe rows to render in the interactive scatter.
        color_pmd_only, color_regions_df
            Legacy overlay controls translated into ``overlay_regions_df``.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive scatter plot returned by ``plot_labels``.
        """
        overlay_regions_df, overlay_style = resolve_overlay_plot_args(
            color_pmd_only=color_pmd_only,
            color_regions_df=color_regions_df,
        )
        return self.plot_labels(
            sample_info=None if use_train_data else sample_info,
            chrom=chrom,
            sample_info_removed=sample_info_removed,
            label_source=label_type,
            overlay_regions_df=overlay_regions_df,
            x_col=x_col,
            y_col=y_col,
            label_title=label_title,
            show_plot=show_plot,
            max_points=max_points,
            overlay_style=overlay_style,
        )

    def plot_labels(
        self,
        sample_info: SampleInfo | None = None,
        chrom: str | None = None,
        sample_info_removed: pd.DataFrame | None = None,
        label_source: str = "kmeans",
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
        state_colors: dict | None = None,
    ):
        """
        Plot genomic-position vs beta for analyzer-owned labels.

        Region args only zoom the x-axis viewport; they do not create a
        highlight overlay unless one is passed explicitly.
        """
        label_source = str(label_source).lower()
        if label_source not in {"kmeans", "rule_based"}:
            raise ValueError(
                "label_source must be either 'kmeans' or 'rule_based' for "
                f"{self.__class__.__name__}. Received: {label_source!r}"
            )

        if sample_info is None:
            self._build_train_joint()
            df_plot = self.train_joint.copy()
            resolved_sample_info = getattr(self.assigner, "train_sample_info", None)
            if (
                label_source == "rule_based"
                and "rule_based_label" not in df_plot.columns
            ):
                if resolved_sample_info is None:
                    raise ValueError(
                        "No train_sample_info is available to compute rule-based labels."
                    )
                df_plot["rule_based_label"] = self.define_states_by_rules(
                    sample_info=resolved_sample_info,
                    sample_emissions=self.assigner.train_emission_df,
                )
        else:
            meth_data, emission_df, _, _, _, labels = (
                self.assigner.apply_kmeans_to_sample(
                    sample_info=sample_info, chrom=chrom
                )
            )

            df_plot = pd.concat([meth_data, emission_df], axis=1)
            df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
            df_plot["kmeans_label"] = labels

            if label_source == "rule_based":
                df_plot["rule_based_label"] = self.define_states_by_rules(
                    sample_info=sample_info,
                    chrom=chrom,
                    sample_emissions=emission_df,
                )
            resolved_sample_info = sample_info

        overlay_regions_df, resolved_overlay_style = resolve_region_overlay_df(
            overlay_regions_df=overlay_regions_df,
        )
        label_col = f"{label_source}_label"
        return plot_interactive_beta_scatter(
            df_plot=df_plot,
            sample_info=resolved_sample_info,
            sample_info_removed=sample_info_removed,
            chrom=chrom,
            out_dir=self.out_dir,
            label_col=label_col,
            x_col=x_col,
            y_col=y_col,
            label_title=label_title,
            show_plot=show_plot,
            max_points=max_points,
            overlay_regions_df=overlay_regions_df,
            state_colors=state_colors,
            overlay_style=(
                resolved_overlay_style
                if overlay_regions_df is not None
                else overlay_style
            ),
            region_start=region_start,
            region_end=region_end,
            region_chrom=region_chrom,
        )
