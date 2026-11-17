"""Utilities for analyzing, comparing, and tuning methylation state assignments."""

from enum import Enum
from io import BytesIO
from pathlib import Path

from graphviz import Source
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
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
    plot_state_labels,
    relabel_by_mean_emission,
    resolve_overlay_plot_args,
)


class MethylStateAnalyzer:
    """Analyze learned state separation and define rule-based CpG states.

    Uses KMeans assignments as a reference for inspecting state separation and
    evaluating rule concordance. It also defines, tunes, and applies
    interpretable cutoff rules that characterize each CpG as a biological
    methylation state.
    """

    def __init__(self, assigner: MethylStateAssigner, out_dir="."):
        """
        Initialize rule-based state analysis around a fitted state assigner.

        Parameters
        ----------
        assigner
            State assigner that provides training emissions, KMeans labels, and
            feature configuration.
        out_dir
            Directory for analysis artifacts and figures.
        """
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

        Parameters
        ----------
        meth_emissions
            Emission table to label with the supplied rule parameters.
        kmeans_labels
            Reference KMeans labels aligned to ``meth_emissions``.
        rule_params
            Keyword arguments forwarded to ``define_states_by_rules_param``.
            Expected keys are ``beta_low_max``, ``beta_high_min``, and
            ``pmd_cutoffs``.

        Returns
        -------
        tuple
            ``(metrics_dict, rule_labels_array)`` comparing the supplied
            KMeans labels against the derived rule-based labels.
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

    def fit_complex_definition(
        self,
        max_depth: int | None = 4,
        min_samples_leaf: float | int | None = None,
        random_state: int | None = 42,
        min_leaf_fraction: float | None = None,
        max_prune_loss: float | None = None,
    ) -> dict:
        """Fit a compact raw-feature threshold surrogate for KMeans states."""
        if not hasattr(self.assigner, "model"):
            raise ValueError("Complex definitions require a fitted KMeans model.")
        model = self.assigner.model
        if max_depth is not None and (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or max_depth < 1
        ):
            raise ValueError("complex_definition_max_depth must be a positive integer.")
        if min_samples_leaf is None:
            min_samples_leaf = 1 if max_depth is None else 0.01
        if isinstance(min_samples_leaf, bool) or not isinstance(min_samples_leaf, (int, float)):
            raise ValueError("complex_definition_min_samples_leaf must be an int or float.")
        if isinstance(min_samples_leaf, int) and min_samples_leaf < 1:
            raise ValueError("Integer complex_definition_min_samples_leaf must be at least 1.")
        if isinstance(min_samples_leaf, float) and not 0 < min_samples_leaf <= 0.5:
            raise ValueError("Float complex_definition_min_samples_leaf must be in (0, 0.5].")
        if (min_leaf_fraction is None) != (max_prune_loss is None):
            raise ValueError(
                "complex_definition_min_leaf_fraction and "
                "complex_definition_max_prune_loss must be supplied together."
            )
        if min_leaf_fraction is not None:
            if not isinstance(min_leaf_fraction, (int, float)) or isinstance(min_leaf_fraction, bool) or not 0 < min_leaf_fraction <= 0.5:
                raise ValueError("complex_definition_min_leaf_fraction must be in (0, 0.5].")
            if not isinstance(max_prune_loss, (int, float)) or isinstance(max_prune_loss, bool) or not 0 <= max_prune_loss <= 1:
                raise ValueError("complex_definition_max_prune_loss must be in [0, 1].")

        feature_names = list(model.feature_cols)
        emissions = self.assigner.train_emission_df
        missing = [name for name in feature_names if name not in emissions.columns]
        if not feature_names or missing:
            raise ValueError(
                f"Training emissions are missing complex-rule features: {missing}"
            )
        values = emissions[feature_names].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Complex definitions require finite raw emission features.")
        _, _, _, relabeled = self.assigner.apply_kmeans_to_emissions(emissions)
        target = np.asarray(
            [value.value if isinstance(value, Enum) else int(value) for value in relabeled],
            dtype=int,
        )
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ).fit(values, target)
        same_state_pruned = self._prune_same_state_subtrees(tree)
        small_leaf_pruned, prune_loss = self._prune_small_leaf_subtrees(
            tree,
            values,
            target,
            min_leaf_fraction=min_leaf_fraction,
            max_prune_loss=max_prune_loss,
        )
        predicted = tree.predict(values).astype(int)
        labels = [state.value for state in MethylationStates]
        report = classification_report(
            target, predicted, labels=labels, output_dict=True, zero_division=0
        )
        self.complex_definition_tree = tree
        self.complex_definition = {
            "kind": "feature_threshold_tree",
            "feature_names": feature_names,
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_leaf": min_samples_leaf,
            "pruning": {
                "same_state_pruned_nodes": same_state_pruned,
                "min_leaf_fraction": min_leaf_fraction,
                "max_prune_loss": max_prune_loss,
                "small_leaf_pruned_nodes": small_leaf_pruned,
                "training_loss_increase": prune_loss,
            },
            "metrics": {
                "biological_state_concordance": float(accuracy_score(target, predicted)),
                "per_state": {
                    MethylationStates(label).name: {
                        "support": int(report[str(label)]["support"]),
                        "precision": float(report[str(label)]["precision"]),
                        "recall": float(report[str(label)]["recall"]),
                        "f1": float(report[str(label)]["f1-score"]),
                    }
                    for label in labels if str(label) in report
                },
            },
        }
        self.complex_definition["leaves"] = self.get_complex_definition_table().to_dict("records")
        return self.complex_definition

    @staticmethod
    def _collapse_tree_node(tree, node_id: int) -> None:
        """Convert an internal sklearn tree node into a leaf in place."""
        tree.children_left[node_id] = _tree.TREE_LEAF
        tree.children_right[node_id] = _tree.TREE_LEAF
        tree.feature[node_id] = _tree.TREE_UNDEFINED
        tree.threshold[node_id] = _tree.TREE_UNDEFINED

    def _prune_same_state_subtrees(self, classifier: DecisionTreeClassifier) -> int:
        """Losslessly collapse any subtree whose descendant leaves share a state."""
        tree = classifier.tree_
        classes = classifier.classes_
        pruned = 0

        def visit(node_id: int) -> int:
            nonlocal pruned
            left, right = tree.children_left[node_id], tree.children_right[node_id]
            if left == _tree.TREE_LEAF:
                return int(classes[int(np.argmax(tree.value[node_id][0]))])
            left_state, right_state = visit(left), visit(right)
            if left_state == right_state:
                self._collapse_tree_node(tree, node_id)
                pruned += 1
                return left_state
            return int(classes[int(np.argmax(tree.value[node_id][0]))])

        visit(0)
        return pruned

    def _prune_small_leaf_subtrees(
        self,
        classifier: DecisionTreeClassifier,
        values: pd.DataFrame,
        target: np.ndarray,
        *,
        min_leaf_fraction: float | None,
        max_prune_loss: float | None,
    ) -> tuple[int, float]:
        """Optionally collapse low-support children within a bounded loss budget."""
        if min_leaf_fraction is None:
            return 0, 0.0
        tree = classifier.tree_
        baseline_loss = 1.0 - accuracy_score(target, classifier.predict(values))
        total_rows = len(target)
        pruned = 0

        def visit(node_id: int):
            nonlocal pruned
            left, right = tree.children_left[node_id], tree.children_right[node_id]
            if left == _tree.TREE_LEAF:
                return
            visit(left)
            visit(right)
            left, right = tree.children_left[node_id], tree.children_right[node_id]
            if left == _tree.TREE_LEAF or right == _tree.TREE_LEAF:
                if min(tree.n_node_samples[left], tree.n_node_samples[right]) / total_rows >= min_leaf_fraction:
                    return
                saved = (
                    tree.children_left[node_id], tree.children_right[node_id],
                    tree.feature[node_id], tree.threshold[node_id],
                )
                self._collapse_tree_node(tree, node_id)
                added_loss = (1.0 - accuracy_score(target, classifier.predict(values))) - baseline_loss
                if added_loss <= max_prune_loss + 1e-12:
                    pruned += 1
                else:
                    (
                        tree.children_left[node_id], tree.children_right[node_id],
                        tree.feature[node_id], tree.threshold[node_id],
                    ) = saved

        visit(0)
        final_loss = 1.0 - accuracy_score(target, classifier.predict(values))
        return pruned, float(final_loss - baseline_loss)

    def _complex_feature_values(self, emissions: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "complex_definition_tree"):
            raise ValueError("Complex definition not fitted or loaded.")
        features = self.complex_definition["feature_names"]
        missing = [name for name in features if name not in emissions.columns]
        if missing:
            raise ValueError(f"Complex definition requires emission features: {missing}")
        values = emissions[features].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Complex definition input features must be finite.")
        return values

    def get_complex_definition_table(self) -> pd.DataFrame:
        """Return raw-feature paths, state labels, support, and purity by leaf."""
        if not hasattr(self, "complex_definition_tree"):
            raise ValueError("Complex definition not fitted or loaded.")
        classifier = self.complex_definition_tree
        tree = classifier.tree_
        classes = classifier.classes_
        features = self.complex_definition["feature_names"]
        rows = []

        def visit(node_id: int, conditions: list[str]):
            left, right = tree.children_left[node_id], tree.children_right[node_id]
            if left == right:
                counts = tree.value[node_id][0]
                state = int(classes[int(np.argmax(counts))])
                rows.append({
                    "leaf_id": int(node_id),
                    "state": MethylationStates(state).name,
                    "support": int(tree.n_node_samples[node_id]),
                    "purity": float(np.max(counts) / np.sum(counts)),
                    "conditions": " AND ".join(conditions) if conditions else "ALL CpGs",
                    "state_counts": {
                        MethylationStates(int(label)).name: int(count)
                        for label, count in zip(classes, counts) if count
                    },
                })
                return
            feature, threshold = features[tree.feature[node_id]], tree.threshold[node_id]
            visit(left, conditions + [f"{feature} <= {threshold:.6g}"])
            visit(right, conditions + [f"{feature} > {threshold:.6g}"])

        visit(0, [])
        return pd.DataFrame(rows)

    def define_states_by_complex_rules(
        self,
        sample_info: SampleInfo | None = None,
        chrom=None,
        sample_emissions: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply a fitted complex definition and return biological states."""
        if not hasattr(self, "complex_definition"):
            raise ValueError("Complex definition not fitted. Run fit_complex_definition first.")
        if sample_emissions is None:
            if sample_info is None:
                raise ValueError("sample_info is required when sample_emissions is not supplied.")
            _, sample_emissions = self.assigner.prepare_sample_for_clustering(
                sample_info, chrom
            )
        raw = self.complex_definition_tree.predict(
            self._complex_feature_values(sample_emissions)
        ).astype(int)
        return np.asarray(
            [MethylationStates(label) for label in raw],
            dtype=object,
        )

    def pretty_print_complex_definition(self, include_leaf_stats: bool = False) -> None:
        """Print compact, factored direct-feature rules for each biological state.

        Parameters
        ----------
        include_leaf_stats
            If ``True``, annotate each terminal rule with its training support
            and purity. Detailed leaf data is always available from
            :meth:`get_complex_definition_table`.
        """
        table = self.get_complex_definition_table()
        print("Complex definition (direct feature thresholds):")
        for state in [member.name for member in MethylationStates]:
            state_leaves = table.loc[table["state"].eq(state)]
            if state_leaves.empty:
                continue
            # Build a prefix tree from this state's decision paths.  Rendering
            # the trie factors conditions shared by all (or a subset of) leaves
            # so users see each shared threshold once instead of in every DNF
            # clause.
            root = {"children": {}, "leaf": None}
            for row in state_leaves.itertuples(index=False):
                conditions = [] if row.conditions == "ALL CpGs" else row.conditions.split(" AND ")
                node = root
                for condition in conditions:
                    node = node["children"].setdefault(
                        condition,
                        {"children": {}, "leaf": None},
                    )
                node["leaf"] = row

            def render_branch(condition, node) -> str:
                expression = condition
                if node["children"]:
                    expression += f" AND ({render_choices(node)})"
                elif include_leaf_stats:
                    leaf = node["leaf"]
                    expression += f" [support={leaf.support}, purity={leaf.purity:.3f}]"
                return expression

            def render_choices(node) -> str:
                children = list(node["children"].items())
                if not children:
                    leaf = node["leaf"]
                    expression = "ALL CpGs"
                    if include_leaf_stats:
                        expression += f" [support={leaf.support}, purity={leaf.purity:.3f}]"
                    return expression
                if len(children) == 1:
                    condition, child = children[0]
                    return render_branch(condition, child)
                return " OR ".join(
                    f"({render_branch(condition, child)})"
                    for condition, child in children
                )

            print(f"{state}: {render_choices(root)}")

    def plot_complex_definition_tree(
        self,
        save_path: str | Path | None = None,
        figsize: tuple[float, float] = (24, 12),
        dpi: int = 150,
        show: bool = True,
    ):
        """Render the direct feature-threshold complex definition as a tree.

        Parameters
        ----------
        save_path
            Optional PNG/PDF/SVG destination. Parent directories are created.
        figsize, dpi
            Matplotlib figure dimensions and raster resolution.
        show
            If ``True``, display the figure before returning it.

        Returns
        -------
        tuple
            The ``(figure, axes)`` containing the rendered decision tree.
        """
        if not hasattr(self, "complex_definition_tree"):
            raise ValueError("Complex definition not fitted or loaded.")
        if len(figsize) != 2 or any(value <= 0 for value in figsize):
            raise ValueError("figsize must contain two positive values.")
        if dpi <= 0:
            raise ValueError("dpi must be positive.")

        classifier = self.complex_definition_tree
        class_names = [MethylationStates(int(label)).name for label in classifier.classes_]
        return self._plot_complex_definition_tree_graphviz(
            classifier=classifier,
            class_names=class_names,
            save_path=save_path,
            figsize=figsize,
            dpi=dpi,
            show=show,
        )

    def _plot_complex_definition_tree_graphviz(
        self,
        *,
        classifier: DecisionTreeClassifier,
        class_names: list[str],
        save_path: str | Path | None,
        figsize: tuple[float, float],
        dpi: int,
        show: bool,
    ):
        """Render the fitted direct-rule tree in a compact Graphviz style."""
        tree = classifier.tree_
        feature_names = self.complex_definition["feature_names"]
        state_colors = {
            "LOW": "#f4a261",
            "PMD": "#69db5a",
            "INTERMEDIATE": "#74c0fc",
            "HIGH": "#da4ee8",
        }
        dot_rows = [
            "digraph complex_definition {",
            'graph [ranksep=0.6, nodesep=0.25, pad=0.2, bgcolor="white", label="MethylSeg complex definition: direct feature thresholds", labelloc=t, fontsize=20];',
            'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#555555"];',
            'edge [fontname="Helvetica", fontsize=10, color="#555555"];',
        ]

        def add_node(node_id: int) -> None:
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            if left == _tree.TREE_LEAF:
                values = tree.value[node_id][0]
                state_index = int(np.argmax(values))
                state = class_names[state_index]
                purity = float(values[state_index] / values.sum()) if values.sum() else 0.0
                label = (
                    f"class = {state}\\n"
                    f"support = {int(tree.n_node_samples[node_id]):,}\\n"
                    f"purity = {purity:.3f}"
                )
                dot_rows.append(
                    f'n{node_id} [label="{label}", fillcolor="{state_colors.get(state, "#eeeeee")}"];'
                )
                return
            feature = feature_names[tree.feature[node_id]]
            dot_rows.append(
                f'n{node_id} [label="{feature} <= {tree.threshold[node_id]:.6g}", fillcolor="#ffffff"];'
            )
            add_node(left)
            add_node(right)
            dot_rows.append(f'n{node_id} -> n{left} [label="True"];')
            dot_rows.append(f'n{node_id} -> n{right} [label="False"];')

        add_node(0)
        dot_rows.append("}")
        image = plt.imread(
            BytesIO(Source("\n".join(dot_rows) + "\n", format="png").pipe(format="png")),
            format="png",
        )
        figure, axes = plt.subplots(figsize=figsize, dpi=dpi)
        axes.imshow(image)
        axes.axis("off")
        axes.set_title("MethylSeg complex definition: direct feature thresholds")
        figure.tight_layout()
        if save_path is not None:
            destination = Path(save_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return figure, axes

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

        Parameters
        ----------
        yaml_file
            Path to a serialized methylseg YAML config file.

        Returns
        -------
        None
            Updates ``state_cutoffs`` and the manual-cutoff flag from the YAML
            contents.
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

        Parameters
        ----------
        beta_low_max
            Upper beta threshold for the low-methylation state. Uses the
            package default when omitted.
        beta_high_min
            Lower beta threshold for the high-methylation state. Uses the
            package default when omitted.
        pmd_cutoffs
            Optional per-window PMD cutoff mapping. Missing values fall back to
            the package defaults for every configured window.

        Returns
        -------
        None
            Stores the resolved cutoff dictionary on the analyzer.
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

        Parameters
        ----------
        use_train_data
            If ``True``, compare cached training labels. Otherwise, generate
            labels for ``sample_info``.
        sample_info
            Sample to evaluate when ``use_train_data`` is ``False``.
        chrom
            Optional chromosome restriction for non-training evaluation.

        Returns
        -------
        pandas.DataFrame
            Confusion matrix comparing KMeans labels to rule-based labels.
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
        x_col: str = "CpG_beg",
        y_col: str = "beta",
        label_title: str | None = None,
        show_plot: bool = True,
        max_points: int = 120_000,
        state_colors: dict | None = None,
    ):
        """
        Plot KMeans or rule-based labels to analyze state relationships.

        This shared analysis helper renders learned KMeans clusters alongside
        rule-derived clusters so their separation, agreement, and rule-based
        CpG characterization can be inspected with the same plot controls.

        Parameters
        ----------
        sample_info
            Sample to analyze. When omitted, uses the cached training data.
        chrom
            Optional chromosome restriction for sample-level plotting.
        sample_info_removed
            Optional table of CpGs removed during preprocessing to show as a
            background layer.
        label_source
            Label family to plot, either ``"kmeans"`` or ``"rule_based"``.
        overlay_regions_df
            Optional region table used to recolor points by overlapping
            intervals.
        overlay_style
            Overlay mode, either ``"state"`` or ``"highlight"``.
        region_start
            Optional genomic start coordinate for x-axis zooming.
        region_end
            Optional genomic end coordinate for x-axis zooming.
        x_col
            Probe-level column used for the x-axis.
        y_col
            Probe-level column used for the y-axis.
        label_title
            Optional legend title override.
        show_plot
            If ``True``, display the Plotly figure immediately.
        max_points
            Maximum number of plotted points before downsampling.
        state_colors
            Optional biological-state color overrides.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive beta scatter plot for the requested labels.

        Region args only zoom the x-axis viewport; they do not create a
        highlight overlay unless one is passed explicitly.
        """
        label_source = str(label_source).lower()
        if label_source not in {"kmeans", "rule_based"}:
            raise ValueError(
                "label_source must be either 'kmeans' or 'rule_based' for "
                f"{self.__class__.__name__}. Received: {label_source!r}"
            )

        if label_source == "kmeans":
            df_plot, resolved_sample_info = (
                self.assigner._prepare_kmeans_label_plot_data(
                    sample_info=sample_info,
                    chrom=chrom,
                )
            )
        elif sample_info is None:
            self._build_train_joint()
            df_plot = self.train_joint.copy()
            resolved_sample_info = getattr(self.assigner, "train_sample_info", None)
            if "rule_based_label" not in df_plot.columns:
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

            df_plot["rule_based_label"] = self.define_states_by_rules(
                sample_info=sample_info,
                chrom=chrom,
                sample_emissions=emission_df,
            )
            resolved_sample_info = sample_info

        label_col = f"{label_source}_label"
        return plot_state_labels(
            df_plot=df_plot,
            sample_info=resolved_sample_info,
            sample_info_removed=sample_info_removed,
            chrom=chrom,
            out_dir=self.out_dir,
            label_col=label_col,
            overlay_regions_df=overlay_regions_df,
            overlay_style=overlay_style,
            region_start=region_start,
            region_end=region_end,
            x_col=x_col,
            y_col=y_col,
            label_title=label_title,
            show_plot=show_plot,
            max_points=max_points,
            state_colors=state_colors,
        )
