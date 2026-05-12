import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from matplotlib import pyplot as plt

def _install_umap_stub() -> None:
    umap = types.ModuleType("umap")

    class _UMAP:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, X):
            return X

    umap.UMAP = _UMAP
    sys.modules.setdefault("umap", umap)


def _install_cthmm_stub() -> None:
    cthmm = types.ModuleType("cthmm")

    class _MultinomialCTHMM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict(self, observations, *args, **kwargs):
            return np.zeros(len(observations), dtype=int)

    cthmm.MultinomialCTHMM = _MultinomialCTHMM
    sys.modules.setdefault("cthmm", cthmm)


try:
    import umap  # noqa: F401
except ModuleNotFoundError:
    _install_umap_stub()

try:
    import cthmm  # noqa: F401
except ModuleNotFoundError:
    _install_cthmm_stub()

from analysis.shared_utils.methyl_seg.methyl_seg import (
    GaussianMethylSegHMM,
    HMMObservationMode,
    KMeansMethylationModel,
    MethylSegPathway,
    MethylSegmenter,
    MethylStateAnalyzer,
    MethylStateAssigner,
    MethylStateAssignmentMethod,
    MethylationStates,
    MultinomialSegHMM,
    SampleInfo,
    StickyCategoricalMethylSegHMM,
)


class StickyCategoricalMethylSegHMMTests(unittest.TestCase):
    def test_make_sticky_transmat_uses_stay_prob(self):
        hmm_model = StickyCategoricalMethylSegHMM(n_states=4, stay_prob=0.8)

        transmat = hmm_model.make_sticky_transmat()

        self.assertEqual(transmat.shape, (4, 4))
        np.testing.assert_allclose(np.diag(transmat), np.full(4, 0.8))
        np.testing.assert_allclose(
            transmat[np.triu_indices(4, k=1)],
            np.full(6, (1.0 - 0.8) / 3.0),
        )
        np.testing.assert_allclose(transmat.sum(axis=1), np.ones(4))

    def test_fixed_smoother_builds_expected_model(self):
        hmm_model = StickyCategoricalMethylSegHMM(
            n_states=4,
            stay_prob=0.9,
            emission_mismatch_prob=0.08,
            transition_prior_strength=7.5,
            fit_transitions=False,
        )

        hmm_model.create_model()

        self.assertEqual(hmm_model.hmm_model.__class__.__name__, "CategoricalHMM")
        np.testing.assert_allclose(hmm_model.hmm_model.transmat_, hmm_model.prior_trans)
        np.testing.assert_allclose(
            np.diag(hmm_model.hmm_model.emissionprob_),
            np.full(4, 0.92),
        )
        np.testing.assert_allclose(
            hmm_model.hmm_model.transmat_prior,
            1.0 + 7.5 * hmm_model.prior_trans,
        )
        np.testing.assert_allclose(
            hmm_model.hmm_model.emissionprob_.sum(axis=1),
            np.ones(4),
        )

    def test_fixed_smoother_fit_is_noop(self):
        hmm_model = StickyCategoricalMethylSegHMM(
            n_states=4,
            stay_prob=0.9,
            fit_transitions=False,
        )
        hmm_model.create_model()
        before = hmm_model.hmm_model.transmat_.copy()

        hmm_model.fit(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int))

        np.testing.assert_allclose(hmm_model.hmm_model.transmat_, before)

    def test_fit_transitions_updates_transition_matrix(self):
        hmm_model = StickyCategoricalMethylSegHMM(
            n_states=4,
            stay_prob=0.8,
            fit_transitions=True,
            n_iter=5,
        )
        hmm_model.create_model()
        before = hmm_model.hmm_model.transmat_.copy()

        hmm_model.fit(
            np.array(
                [0] * 12 + [1] * 8 + [0] * 10 + [2] * 6 + [3] * 9,
                dtype=int,
            )
        )

        self.assertEqual(hmm_model.hmm_model.transmat_.shape, (4, 4))
        np.testing.assert_allclose(
            hmm_model.hmm_model.transmat_.sum(axis=1),
            np.ones(4),
            atol=1e-8,
        )
        self.assertFalse(np.allclose(hmm_model.hmm_model.transmat_, before))

    def test_invalid_stay_prob_raises(self):
        for stay_prob in (0, 1, -0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "stay_prob"):
                StickyCategoricalMethylSegHMM(n_states=4, stay_prob=stay_prob)

    def test_invalid_emission_mismatch_prob_raises(self):
        for emission_mismatch_prob in (-0.1, 1.0, 1.2):
            with self.assertRaisesRegex(ValueError, "emission_mismatch_prob"):
                StickyCategoricalMethylSegHMM(
                    n_states=4,
                    emission_mismatch_prob=emission_mismatch_prob,
                )

    def test_invalid_transition_prior_strength_raises(self):
        for transition_prior_strength in (-0.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "transition_prior_strength"):
                StickyCategoricalMethylSegHMM(
                    n_states=4,
                    transition_prior_strength=transition_prior_strength,
                )


class HmmlearnDiscreteLengthHandlingTests(unittest.TestCase):
    def test_sticky_wrapper_passes_lengths_to_hmmlearn_fit_and_predict(self):
        hmm_model = StickyCategoricalMethylSegHMM(
            n_states=4,
            fit_transitions=True,
        )
        hmm_model.hmm_model = mock.Mock()
        hmm_model.lengths = [2, 2]
        emissions = np.array([0, 0, 1, 1], dtype=int)

        hmm_model.fit(emissions)
        hmm_model.predict(emissions)

        np.testing.assert_array_equal(
            hmm_model.hmm_model.fit.call_args.args[0],
            emissions.reshape(-1, 1),
        )
        self.assertEqual(hmm_model.hmm_model.fit.call_args.kwargs["lengths"], [2, 2])
        np.testing.assert_array_equal(
            hmm_model.hmm_model.predict.call_args.args[0],
            emissions.reshape(-1, 1),
        )
        self.assertEqual(
            hmm_model.hmm_model.predict.call_args.kwargs["lengths"],
            [2, 2],
        )

    def test_multinomial_wrapper_passes_lengths_to_hmmlearn_fit_and_predict(self):
        hmm_model = MultinomialSegHMM(n_states=4)
        hmm_model.hmm_model = mock.Mock()
        hmm_model.lengths = [2, 2]
        emissions = np.array([0, 0, 1, 1], dtype=int)
        formatted_emissions = np.eye(4, dtype=int)[emissions]

        hmm_model.fit(emissions)
        hmm_model.predict(emissions)

        np.testing.assert_array_equal(
            hmm_model.hmm_model.fit.call_args.args[0],
            formatted_emissions,
        )
        self.assertEqual(hmm_model.hmm_model.fit.call_args.kwargs["lengths"], [2, 2])
        np.testing.assert_array_equal(
            hmm_model.hmm_model.predict.call_args.args[0],
            formatted_emissions,
        )
        self.assertEqual(
            hmm_model.hmm_model.predict.call_args.kwargs["lengths"],
            [2, 2],
        )


class StickyPathwayIntegrationTests(unittest.TestCase):
    def make_sample_info(self) -> SampleInfo:
        meth_df = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30, 40],
                "CpG_end": [11, 21, 31, 41],
                "beta": [0.1, 0.2, 0.8, 0.9],
            }
        )
        return SampleInfo(sample_id="train_sample", meth_data=meth_df)

    def test_pathway_initializes_default_sticky_model(self):
        pathway = MethylSegPathway(
            train_sample_info=self.make_sample_info(),
            hmm_type="sticky",
            hmm_params={},
            out_dir=".",
        )

        self.assertIsInstance(pathway.hmm_model, StickyCategoricalMethylSegHMM)
        self.assertAlmostEqual(pathway.hmm_model.stay_prob, 0.995)
        self.assertAlmostEqual(pathway.hmm_model.emission_mismatch_prob, 0.01)
        self.assertAlmostEqual(pathway.hmm_model.transition_prior_strength, 50.0)
        self.assertFalse(pathway.hmm_model.fit_transitions)
        self.assertEqual(pathway.assigner.cluster_space, "pca")
        self.assertEqual(pathway.assigner.n_pca, 5)

    def test_pathway_exposes_cluster_config_at_top_level(self):
        pathway = MethylSegPathway(
            train_sample_info=self.make_sample_info(),
            hmm_type="sticky",
            hmm_params={},
            out_dir=".",
            cluster_space="raw",
            n_pca=None,
        )

        self.assertEqual(pathway.cluster_space, "raw")
        self.assertIsNone(pathway.n_pca)
        self.assertEqual(pathway.assigner.cluster_space, "raw")
        self.assertIsNone(pathway.assigner.n_pca)

    def test_pathway_initializes_gaussian_emission_mode(self):
        pathway = MethylSegPathway(
            train_sample_info=self.make_sample_info(),
            hmm_type="gaussian",
            hmm_observation_mode=HMMObservationMode.GAUSSIAN_EMISSIONS,
            hmm_params={},
            out_dir=".",
        )

        self.assertIsInstance(pathway.hmm_model, GaussianMethylSegHMM)
        self.assertEqual(
            pathway.hmm_observation_mode,
            HMMObservationMode.GAUSSIAN_EMISSIONS,
        )
        self.assertEqual(
            pathway.segmentor.hmm_observation_mode,
            HMMObservationMode.GAUSSIAN_EMISSIONS,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None,
        "pyarrow required for feather-backed roundtrip",
    )
    def test_yaml_roundtrip_preserves_sticky_hmm_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pathway = MethylSegPathway(
                train_sample_info=self.make_sample_info(),
                hmm_type="sticky",
                hmm_params={
                    "stay_prob": 0.97,
                    "emission_mismatch_prob": 0.03,
                    "transition_prior_strength": 12.0,
                    "fit_transitions": True,
                },
                out_dir=str(tmpdir_path),
            )

            yaml_path = tmpdir_path / "sticky_config.yaml"
            pathway.to_yaml(str(yaml_path), include_learned=False)

            restored = MethylSegPathway.from_yaml(str(yaml_path), load_learned=False)

            self.assertEqual(restored.hmm_type, "sticky")
            self.assertEqual(restored.hmm_params["stay_prob"], 0.97)
            self.assertEqual(restored.hmm_params["emission_mismatch_prob"], 0.03)
            self.assertEqual(restored.hmm_params["transition_prior_strength"], 12.0)
            self.assertTrue(restored.hmm_params["fit_transitions"])

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None,
        "pyarrow required for feather-backed roundtrip",
    )
    def test_yaml_roundtrip_preserves_gaussian_observation_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pathway = MethylSegPathway(
                train_sample_info=self.make_sample_info(),
                hmm_type="gaussian",
                hmm_observation_mode=HMMObservationMode.GAUSSIAN_EMISSIONS,
                state_assignment_method=MethylStateAssignmentMethod.KMEANS,
                hmm_params={"n_iter": 12, "tol": 1e-2},
                out_dir=str(tmpdir_path),
            )

            yaml_path = tmpdir_path / "gaussian_config.yaml"
            pathway.to_yaml(str(yaml_path), include_learned=False)

            restored = MethylSegPathway.from_yaml(str(yaml_path), load_learned=False)

            self.assertEqual(restored.hmm_type, "gaussian")
            self.assertEqual(
                restored.hmm_observation_mode,
                HMMObservationMode.GAUSSIAN_EMISSIONS,
            )
            self.assertEqual(
                restored.segmentor.hmm_observation_mode,
                HMMObservationMode.GAUSSIAN_EMISSIONS,
            )
            self.assertEqual(
                restored.state_assignment_method,
                MethylStateAssignmentMethod.KMEANS,
            )
            self.assertEqual(restored.hmm_params["n_iter"], 12)
            self.assertEqual(restored.hmm_params["tol"], 1e-2)


class KMeansAssignmentTests(unittest.TestCase):
    def make_sample_info(self) -> SampleInfo:
        positions = list(range(10, 170, 10))
        betas = [
            0.02,
            0.04,
            0.08,
            0.18,
            0.28,
            0.38,
            0.48,
            0.58,
            0.68,
            0.78,
            0.86,
            0.90,
            0.94,
            0.72,
            0.52,
            0.22,
        ]
        meth_df = pd.DataFrame(
            {
                "CpG_chrm": ["chr1"] * len(positions),
                "CpG_beg": positions,
                "CpG_end": [pos + 1 for pos in positions],
                "beta": betas,
            }
        )
        return SampleInfo(sample_id="kmeans_sample", meth_data=meth_df)

    def test_assign_states_kmeans_reuses_precomputed_emissions(self):
        sample_info = self.make_sample_info()

        with tempfile.TemporaryDirectory() as tmpdir:
            pathway = MethylSegPathway(
                train_sample_info=sample_info,
                hmm_type="sticky",
                out_dir=str(tmpdir),
                state_assignment_method=MethylStateAssignmentMethod.KMEANS,
            )
            pathway.fit_pathway()

            _, _emission_matrix, emission_df = pathway.assigner.prepare_sample_for_clustering(
                sample_info,
                chrom="chr1",
            )
            _, _, _, expected_labels = pathway.assigner.apply_kmeans_to_emissions(
                emission_df
            )

            with mock.patch.object(
                pathway.assigner,
                "apply_kmeans_to_sample",
                side_effect=AssertionError("assign_states should reuse precomputed emissions"),
            ):
                actual_meth_data, actual_emissions = pathway.segmentor.assign_states(
                    sample_info,
                    chrom="chr1",
                )

            pd.testing.assert_frame_equal(
                actual_emissions.reset_index(drop=True),
                emission_df.reset_index(drop=True),
            )
            np.testing.assert_array_equal(
                actual_meth_data["state"].to_numpy(dtype=int),
                MethylationStates.convert_to_numeric(expected_labels),
            )

    def test_train_and_apply_kmeans_raw_cluster_space_returns_none_pca_scores(self):
        sample_info = self.make_sample_info()
        assigner = MethylStateAssigner(
            window_specs=[(20, "20bp"), (40, "40bp")],
            random_state=17,
            cluster_space="raw",
            n_pca=None,
        )

        model, _meth_data, emission_df, pca_scores, labels = assigner.train_kmeans_for_sample(
            sample_info=sample_info,
            train_chroms=["chr1"],
            max_cpg_per_chrom=None,
        )

        self.assertEqual(model.cluster_space, "raw")
        self.assertIsNone(model.n_pca)
        self.assertIsNone(model.pca)
        self.assertIsNone(pca_scores)
        self.assertEqual(len(labels), len(emission_df))

        apply_scores, _, _, apply_labels = assigner.apply_kmeans_to_emissions(
            emission_df
        )

        self.assertIsNone(apply_scores)
        self.assertEqual(len(apply_labels), len(emission_df))

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None,
        "pyarrow required for feather-backed roundtrip",
    )
    def test_yaml_roundtrip_preserves_cluster_space_and_n_pca(self):
        sample_info = self.make_sample_info()

        test_cases = (
            ("raw", None, False),
            ("pca", 2, True),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for cluster_space, n_pca, expect_pca in test_cases:
                case_dir = tmpdir_path / cluster_space
                case_dir.mkdir()
                pathway = MethylSegPathway(
                    train_sample_info=sample_info,
                    hmm_type="sticky",
                    out_dir=str(case_dir),
                    state_assignment_method=MethylStateAssignmentMethod.KMEANS,
                    cluster_space=cluster_space,
                    n_pca=n_pca,
                )
                pathway.fit_pathway()

                yaml_path = case_dir / "config.yaml"
                pathway.to_yaml(str(yaml_path), include_learned=True)

                restored = MethylSegPathway.from_yaml(str(yaml_path), load_learned=True)

                self.assertEqual(restored.assigner.cluster_space, cluster_space)
                self.assertEqual(restored.assigner.model.cluster_space, cluster_space)
                self.assertEqual(restored.assigner.n_pca, n_pca)
                self.assertEqual(restored.assigner.model.n_pca, n_pca)
                self.assertEqual(restored.cluster_space, cluster_space)
                self.assertEqual(restored.n_pca, n_pca)
                self.assertEqual(restored.assigner.model.pca is not None, expect_pca)


class RelabelingRuleTests(unittest.TestCase):
    def make_assigner(self) -> MethylStateAssigner:
        return MethylStateAssigner(
            window_specs=[
                (25_000, "25kb"),
                (500_000, "500kb"),
                (1_000_000, "1Mb"),
            ],
            random_state=17,
        )

    def make_analyzer(self) -> MethylStateAnalyzer:
        return MethylStateAnalyzer(assigner=self.make_assigner(), out_dir=".")

    def make_cluster_rows(
        self,
        beta: float,
        label_stats: dict[str, dict[str, float]],
        n_rows: int = 2,
    ) -> list[dict[str, float]]:
        rows = []
        for _ in range(n_rows):
            row = {"beta": beta}
            for label, stats in label_stats.items():
                row[f"{label}_int_pct"] = stats["int"]
                row[f"{label}_std"] = stats["std"]
                row[f"{label}_high_pct"] = stats["high"]
                row[f"{label}_low_pct"] = stats["low"]
                row[f"{label}_n_cpg"] = stats["n_cpg"]
            rows.append(row)
        return rows

    def test_relabel_by_mean_emission_uses_regional_windows_and_low_pct_for_pmr(self):
        assigner = self.make_assigner()
        rows = []
        raw_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        rows.extend(
            self.make_cluster_rows(
                0.05,
                {
                    "25kb": {"int": 0.05, "std": 0.05, "high": 0.0, "low": 0.95, "n_cpg": 20},
                    "500kb": {"int": 0.10, "std": 0.08, "high": 0.0, "low": 0.90, "n_cpg": 200},
                    "1Mb": {"int": 0.12, "std": 0.09, "high": 0.0, "low": 0.88, "n_cpg": 250},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.45,
                {
                    "25kb": {"int": 0.20, "std": 0.30, "high": 0.20, "low": 0.20, "n_cpg": 40},
                    "500kb": {"int": 0.82, "std": 0.11, "high": 0.05, "low": 0.12, "n_cpg": 220},
                    "1Mb": {"int": 0.82, "std": 0.11, "high": 0.05, "low": 0.12, "n_cpg": 250},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.50,
                {
                    "25kb": {"int": 0.95, "std": 0.05, "high": 0.02, "low": 0.02, "n_cpg": 50},
                    "500kb": {"int": 0.82, "std": 0.11, "high": 0.05, "low": 0.42, "n_cpg": 220},
                    "1Mb": {"int": 0.82, "std": 0.11, "high": 0.05, "low": 0.42, "n_cpg": 250},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.92,
                {
                    "25kb": {"int": 0.05, "std": 0.06, "high": 0.95, "low": 0.0, "n_cpg": 25},
                    "500kb": {"int": 0.08, "std": 0.09, "high": 0.92, "low": 0.0, "n_cpg": 210},
                    "1Mb": {"int": 0.10, "std": 0.10, "high": 0.90, "low": 0.0, "n_cpg": 260},
                },
            )
        )

        emission_df = pd.DataFrame(rows)

        relabeled = assigner.relabel_by_mean_emission(raw_labels, emission_df)

        expected = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.LOW,
                MethylationStates.INTERMEDIATE,
                MethylationStates.INTERMEDIATE,
                MethylationStates.PMR,
                MethylationStates.PMR,
                MethylationStates.HIGH,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        np.testing.assert_array_equal(relabeled, expected)

    def test_relabel_by_mean_emission_single_cluster_can_return_low(self):
        assigner = self.make_assigner()
        raw_labels = np.array([0, 0])
        emission_df = pd.DataFrame(
            self.make_cluster_rows(
                0.08,
                {
                    "25kb": {
                        "int": 0.05,
                        "std": 0.05,
                        "high": 0.02,
                        "low": 0.93,
                        "n_cpg": 20,
                    },
                    "500kb": {
                        "int": 0.08,
                        "std": 0.08,
                        "high": 0.04,
                        "low": 0.88,
                        "n_cpg": 200,
                    },
                    "1Mb": {
                        "int": 0.10,
                        "std": 0.09,
                        "high": 0.04,
                        "low": 0.86,
                        "n_cpg": 240,
                    },
                },
            )
        )

        relabeled = assigner.relabel_by_mean_emission(raw_labels, emission_df)

        np.testing.assert_array_equal(
            relabeled,
            np.array([MethylationStates.LOW, MethylationStates.LOW], dtype=object),
        )

    def test_relabel_by_mean_emission_single_cluster_prefers_pmr_when_intermediate_dominates_and_low_exceeds_high(self):
        assigner = self.make_assigner()
        raw_labels = np.array([0, 0])
        emission_df = pd.DataFrame(
            self.make_cluster_rows(
                0.48,
                {
                    "25kb": {
                        "int": 0.55,
                        "std": 0.15,
                        "high": 0.15,
                        "low": 0.25,
                        "n_cpg": 45,
                    },
                    "500kb": {
                        "int": 0.82,
                        "std": 0.10,
                        "high": 0.06,
                        "low": 0.12,
                        "n_cpg": 220,
                    },
                    "1Mb": {
                        "int": 0.84,
                        "std": 0.10,
                        "high": 0.05,
                        "low": 0.11,
                        "n_cpg": 250,
                    },
                },
            )
        )

        relabeled = assigner.relabel_by_mean_emission(raw_labels, emission_df)

        np.testing.assert_array_equal(
            relabeled,
            np.array([MethylationStates.PMR, MethylationStates.PMR], dtype=object),
        )

    def test_relabel_by_mean_emission_single_cluster_prefers_intermediate_when_high_exceeds_low(self):
        assigner = self.make_assigner()
        raw_labels = np.array([0, 0])
        emission_df = pd.DataFrame(
            self.make_cluster_rows(
                0.52,
                {
                    "25kb": {
                        "int": 0.55,
                        "std": 0.14,
                        "high": 0.26,
                        "low": 0.12,
                        "n_cpg": 45,
                    },
                    "500kb": {
                        "int": 0.84,
                        "std": 0.10,
                        "high": 0.09,
                        "low": 0.03,
                        "n_cpg": 220,
                    },
                    "1Mb": {
                        "int": 0.86,
                        "std": 0.10,
                        "high": 0.08,
                        "low": 0.02,
                        "n_cpg": 250,
                    },
                },
            )
        )

        relabeled = assigner.relabel_by_mean_emission(raw_labels, emission_df)

        np.testing.assert_array_equal(
            relabeled,
            np.array(
                [MethylationStates.INTERMEDIATE, MethylationStates.INTERMEDIATE],
                dtype=object,
            ),
        )

    def test_relabel_by_mean_emission_single_cluster_can_return_high(self):
        assigner = self.make_assigner()
        raw_labels = np.array([0, 0])
        emission_df = pd.DataFrame(
            self.make_cluster_rows(
                0.88,
                {
                    "25kb": {
                        "int": 0.06,
                        "std": 0.05,
                        "high": 0.92,
                        "low": 0.02,
                        "n_cpg": 18,
                    },
                    "500kb": {
                        "int": 0.08,
                        "std": 0.08,
                        "high": 0.90,
                        "low": 0.02,
                        "n_cpg": 210,
                    },
                    "1Mb": {
                        "int": 0.10,
                        "std": 0.09,
                        "high": 0.88,
                        "low": 0.02,
                        "n_cpg": 250,
                    },
                },
            )
        )

        relabeled = assigner.relabel_by_mean_emission(raw_labels, emission_df)

        np.testing.assert_array_equal(
            relabeled,
            np.array([MethylationStates.HIGH, MethylationStates.HIGH], dtype=object),
        )

    def test_relabel_by_mean_emission_with_cutoffs_requires_same_regional_window(self):
        assigner = self.make_assigner()
        rows = []
        raw_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        rows.extend(
            self.make_cluster_rows(
                0.05,
                {
                    "25kb": {"int": 0.05, "std": 0.05, "high": 0.0, "low": 0.95, "n_cpg": 20},
                    "500kb": {"int": 0.10, "std": 0.08, "high": 0.0, "low": 0.90, "n_cpg": 200},
                    "1Mb": {"int": 0.12, "std": 0.09, "high": 0.0, "low": 0.88, "n_cpg": 250},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.47,
                {
                    "25kb": {"int": 0.30, "std": 0.20, "high": 0.15, "low": 0.15, "n_cpg": 35},
                    "500kb": {"int": 0.85, "std": 0.25, "high": 0.20, "low": 0.10, "n_cpg": 210},
                    "1Mb": {"int": 0.40, "std": 0.10, "high": 0.05, "low": 0.10, "n_cpg": 240},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.53,
                {
                    "25kb": {"int": 0.45, "std": 0.22, "high": 0.12, "low": 0.18, "n_cpg": 45},
                    "500kb": {"int": 0.82, "std": 0.12, "high": 0.05, "low": 0.10, "n_cpg": 225},
                    "1Mb": {"int": 0.84, "std": 0.10, "high": 0.05, "low": 0.12, "n_cpg": 255},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.92,
                {
                    "25kb": {"int": 0.05, "std": 0.06, "high": 0.95, "low": 0.0, "n_cpg": 25},
                    "500kb": {"int": 0.08, "std": 0.09, "high": 0.92, "low": 0.0, "n_cpg": 210},
                    "1Mb": {"int": 0.10, "std": 0.10, "high": 0.90, "low": 0.0, "n_cpg": 260},
                },
            )
        )

        emission_df = pd.DataFrame(rows)
        state_cutoffs = {
            "beta_low_max": 0.20,
            "beta_high_min": 0.70,
            "pmr_cutoffs": {
                "500kb": {
                    "int_min": 0.80,
                    "std_max": 0.15,
                    "high_max": 0.10,
                    "low_max": 0.20,
                },
                "1Mb": {
                    "int_min": 0.80,
                    "std_max": 0.15,
                    "high_max": 0.10,
                    "low_max": 0.20,
                },
            },
        }

        relabeled = assigner.relabel_by_mean_emission(
            raw_labels,
            emission_df,
            state_cutoffs=state_cutoffs,
        )

        expected = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.LOW,
                MethylationStates.INTERMEDIATE,
                MethylationStates.INTERMEDIATE,
                MethylationStates.PMR,
                MethylationStates.PMR,
                MethylationStates.HIGH,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        np.testing.assert_array_equal(relabeled, expected)

    def test_relabel_by_mean_emission_four_state_assignment_is_deterministic_on_ties(self):
        assigner = self.make_assigner()
        rows = []
        raw_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        rows.extend(
            self.make_cluster_rows(
                0.10,
                {
                    "25kb": {"int": 0.08, "std": 0.05, "high": 0.02, "low": 0.90, "n_cpg": 20},
                    "500kb": {"int": 0.10, "std": 0.08, "high": 0.02, "low": 0.88, "n_cpg": 200},
                    "1Mb": {"int": 0.12, "std": 0.09, "high": 0.02, "low": 0.86, "n_cpg": 250},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.50,
                {
                    "25kb": {"int": 0.50, "std": 0.14, "high": 0.18, "low": 0.24, "n_cpg": 40},
                    "500kb": {"int": 0.80, "std": 0.10, "high": 0.05, "low": 0.12, "n_cpg": 220},
                    "1Mb": {"int": 0.82, "std": 0.10, "high": 0.05, "low": 0.12, "n_cpg": 250},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.50,
                {
                    "25kb": {"int": 0.50, "std": 0.14, "high": 0.18, "low": 0.24, "n_cpg": 40},
                    "500kb": {"int": 0.80, "std": 0.10, "high": 0.05, "low": 0.12, "n_cpg": 220},
                    "1Mb": {"int": 0.82, "std": 0.10, "high": 0.05, "low": 0.12, "n_cpg": 250},
                },
            )
        )
        rows.extend(
            self.make_cluster_rows(
                0.90,
                {
                    "25kb": {"int": 0.08, "std": 0.05, "high": 0.90, "low": 0.02, "n_cpg": 20},
                    "500kb": {"int": 0.10, "std": 0.08, "high": 0.88, "low": 0.02, "n_cpg": 200},
                    "1Mb": {"int": 0.12, "std": 0.09, "high": 0.86, "low": 0.02, "n_cpg": 250},
                },
            )
        )

        emission_df = pd.DataFrame(rows)

        relabeled = assigner.relabel_by_mean_emission(raw_labels, emission_df)

        expected = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.LOW,
                MethylationStates.PMR,
                MethylationStates.PMR,
                MethylationStates.INTERMEDIATE,
                MethylationStates.INTERMEDIATE,
                MethylationStates.HIGH,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        np.testing.assert_array_equal(relabeled, expected)

    def test_define_states_by_rules_param_requires_same_regional_window_and_low_max(self):
        analyzer = self.make_analyzer()
        meth_emissions = pd.DataFrame(
            [
                {
                    "beta": 0.50,
                    "25kb_int_pct": 0.30,
                    "25kb_std": 0.22,
                    "25kb_high_pct": 0.12,
                    "25kb_low_pct": 0.18,
                    "500kb_int_pct": 0.85,
                    "500kb_std": 0.25,
                    "500kb_high_pct": 0.20,
                    "500kb_low_pct": 0.10,
                    "1Mb_int_pct": 0.40,
                    "1Mb_std": 0.10,
                    "1Mb_high_pct": 0.05,
                    "1Mb_low_pct": 0.10,
                },
                {
                    "beta": 0.52,
                    "25kb_int_pct": 0.45,
                    "25kb_std": 0.22,
                    "25kb_high_pct": 0.12,
                    "25kb_low_pct": 0.18,
                    "500kb_int_pct": 0.82,
                    "500kb_std": 0.12,
                    "500kb_high_pct": 0.05,
                    "500kb_low_pct": 0.10,
                    "1Mb_int_pct": 0.84,
                    "1Mb_std": 0.10,
                    "1Mb_high_pct": 0.05,
                    "1Mb_low_pct": 0.12,
                },
                {
                    "beta": 0.54,
                    "25kb_int_pct": 0.45,
                    "25kb_std": 0.22,
                    "25kb_high_pct": 0.12,
                    "25kb_low_pct": 0.18,
                    "500kb_int_pct": 0.82,
                    "500kb_std": 0.12,
                    "500kb_high_pct": 0.05,
                    "500kb_low_pct": 0.35,
                    "1Mb_int_pct": 0.84,
                    "1Mb_std": 0.10,
                    "1Mb_high_pct": 0.05,
                    "1Mb_low_pct": 0.35,
                },
            ]
        )

        labels = analyzer.define_states_by_rules_param(
            meth_emissions,
            beta_low_max=0.20,
            beta_high_min=0.70,
            pmr_cutoffs={
                "500kb": {
                    "int_min": 0.80,
                    "std_max": 0.15,
                    "high_max": 0.10,
                    "low_max": 0.20,
                },
                "1Mb": {
                    "int_min": 0.80,
                    "std_max": 0.15,
                    "high_max": 0.10,
                    "low_max": 0.20,
                },
            },
        )

        expected = np.array(
            [
                MethylationStates.INTERMEDIATE,
                MethylationStates.PMR,
                MethylationStates.INTERMEDIATE,
            ],
            dtype=object,
        )
        np.testing.assert_array_equal(labels, expected)

    def test_set_state_cutoffs_adds_low_max_when_missing(self):
        analyzer = self.make_analyzer()

        analyzer.set_state_cutoffs(
            pmr_cutoffs={
                "500kb": {
                    "int_min": 0.60,
                    "std_max": 0.20,
                    "high_max": 0.15,
                }
            }
        )

        self.assertAlmostEqual(
            analyzer.state_cutoffs["pmr_cutoffs"]["500kb"]["low_max"],
            0.246,
        )
        self.assertAlmostEqual(
            analyzer.state_cutoffs["pmr_cutoffs"]["1Mb"]["low_max"],
            0.246,
        )


class EmissionFeatureTests(unittest.TestCase):
    def make_sample_info(self) -> SampleInfo:
        meth_df = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30, 40],
                "CpG_end": [11, 21, 31, 41],
                "beta": [0.1, 0.2, 0.8, 0.9],
            }
        )
        return SampleInfo(sample_id="emission_sample", meth_data=meth_df)

    def make_assigner(self) -> MethylStateAssigner:
        return MethylStateAssigner(
            window_specs=[(20, "20bp"), (40, "40bp")],
            random_state=17,
        )

    def test_prepare_sample_for_clustering_includes_per_window_low_pct_and_n_cpg(self):
        sample_info = self.make_sample_info()
        assigner = self.make_assigner()

        _, emission_matrix, emission_df = assigner.prepare_sample_for_clustering(
            sample_info,
            chrom="chr1",
        )

        self.assertEqual(emission_matrix.shape, (4, 13))
        self.assertIn("20bp_low_pct", emission_df.columns)
        self.assertIn("40bp_low_pct", emission_df.columns)
        self.assertIn("20bp_n_cpg", emission_df.columns)
        self.assertIn("40bp_n_cpg", emission_df.columns)
        np.testing.assert_allclose(
            emission_df["20bp_low_pct"].to_numpy(),
            np.array([0.5, 1.0 / 3.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            emission_df["40bp_low_pct"].to_numpy(),
            np.array([1.0 / 3.0, 0.25, 0.25, 0.0]),
        )
        np.testing.assert_allclose(
            emission_df["20bp_n_cpg"].to_numpy(),
            np.array([2.0, 3.0, 3.0, 2.0]),
        )
        np.testing.assert_allclose(
            emission_df["40bp_n_cpg"].to_numpy(),
            np.array([3.0, 4.0, 4.0, 3.0]),
        )

    def test_create_emission_df_windows_to_use_retains_n_cpg(self):
        sample_info = self.make_sample_info()
        assigner = self.make_assigner()

        summary_stats = assigner.generate_multi_window_summary_centered(
            sample_info.meth_data,
            chrom="chr1",
        )
        emission_df = assigner.create_emission_df(
            summary_stats,
            windows_to_use=["40bp"],
        )

        self.assertListEqual(
            list(emission_df.columns),
            [
                "beta",
                "40bp_avg_meth",
                "40bp_std",
                "40bp_high_pct",
                "40bp_int_pct",
                "40bp_low_pct",
                "40bp_n_cpg",
            ],
        )
        np.testing.assert_allclose(
            emission_df["40bp_low_pct"].to_numpy(),
            np.array([1.0 / 3.0, 0.25, 0.25, 0.0]),
        )
        np.testing.assert_allclose(
            emission_df["40bp_n_cpg"].to_numpy(),
            np.array([3.0, 4.0, 4.0, 3.0]),
        )

    def test_prepare_sample_for_clustering_windows_to_use_preserves_base_schema(self):
        sample_info = self.make_sample_info()
        assigner = self.make_assigner()

        _, emission_matrix, emission_df = assigner.prepare_sample_for_clustering(
            sample_info,
            chrom="chr1",
            windows_to_use=["40bp"],
        )

        self.assertEqual(emission_matrix.shape, (4, 7))
        self.assertListEqual(
            list(emission_df.columns),
            [
                "beta",
                "40bp_avg_meth",
                "40bp_std",
                "40bp_high_pct",
                "40bp_int_pct",
                "40bp_low_pct",
                "40bp_n_cpg",
            ],
        )


class _FakeFigure:
    def __init__(self):
        self.layout_updates = []
        self.write_html_path = None
        self.for_each_trace_called = False
        self.show_called = False

    def update_traces(self, *args, **kwargs):
        return self

    def update_layout(self, *args, **kwargs):
        self.layout_updates.append(kwargs)
        return self

    def for_each_trace(self, callback):
        self.for_each_trace_called = True
        return self

    def show(self, *args, **kwargs):
        self.show_called = True
        return self

    def write_html(self, path):
        self.write_html_path = path
        return None


class _FakeColorbar:
    def __init__(self):
        self.ticklabels = None
        self.label = None

    def set_ticklabels(self, labels):
        self.ticklabels = list(labels)

    def set_label(self, label):
        self.label = label


class PlottingBehaviorTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def make_umap_assigner(
        self,
        scaler: mock.Mock,
        imputer: mock.Mock,
        pca: mock.Mock | None = None,
        random_state: int = 17,
        cluster_space: str = "pca",
        n_pca: int | None = 2,
    ) -> MethylStateAssigner:
        assigner = MethylStateAssigner(
            random_state=random_state,
            cluster_space=cluster_space,
            n_pca=n_pca,
        )
        assigner.model = KMeansMethylationModel(
            kmeans=mock.Mock(),
            scaler=scaler,
            imputer=imputer,
            pca=pca,
            feature_cols=["feat_a", "feat_b"],
            n_states=assigner.n_states,
            cluster_space=cluster_space,
            n_pca=n_pca,
        )
        return assigner

    def make_umap_emissions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feat_a": [0.1, np.nan, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )

    def make_analyzer(self, out_dir: str) -> MethylStateAnalyzer:
        assigner = SimpleNamespace(
            model=object(),
            n_states=4,
            window_specs=[(500_000, "500kb")],
            train_meth=pd.DataFrame(
                {
                    "CpG_chrm": ["chr1", "chr1", "chr1"],
                    "CpG_beg": [10, 20, 30],
                    "CpG_end": [11, 21, 31],
                    "beta": [0.2, 0.45, 0.85],
                }
            ),
            train_emission_df=pd.DataFrame(
                {
                    "beta": [0.2, 0.45, 0.85],
                    "500kb_int_pct": [0.1, 0.8, 0.2],
                }
            ),
            train_labels=np.array(
                [
                    MethylationStates.LOW,
                    MethylationStates.PMR,
                    MethylationStates.HIGH,
                ],
                dtype=object,
            ),
        )
        return MethylStateAnalyzer(assigner=assigner, out_dir=out_dir)

    def make_segmenter(self, out_dir: str) -> MethylSegmenter:
        analyzer = mock.Mock()
        analyzer.assigner = SimpleNamespace(n_states=4)
        return MethylSegmenter(
            analyzer=analyzer,
            hmm_model=mock.Mock(),
            out_dir=out_dir,
        )

    def make_segmenter_meth_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
                "beta": [0.2, 0.45, 0.85],
                "hmm_state_readable": [
                    MethylationStates.LOW,
                    MethylationStates.PMR,
                    MethylationStates.HIGH,
                ],
                "state_readable": [
                    MethylationStates.LOW,
                    MethylationStates.PMR,
                    MethylationStates.HIGH,
                ],
            }
        )

    def test_plot_umap_clusters_uses_scaled_raw_features_by_default(self):
        emission_df = self.make_umap_emissions()
        scaled_features = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        imputed_features = np.array(
            [
                [0.1, 1.0],
                [0.2, 2.0],
                [0.3, 3.0],
            ]
        )
        imputer = mock.Mock()
        imputer.transform.return_value = imputed_features
        scaler = mock.Mock()
        scaler.transform.return_value = scaled_features
        assigner = self.make_umap_assigner(scaler=scaler, imputer=imputer)
        labels = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.PMR,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )

        umap_runner = mock.Mock()
        umap_runner.fit_transform.return_value = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.umap.UMAP",
            return_value=umap_runner,
        ) as umap_ctor, mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.plt.show"
        ):
            assigner.plot_umap_clusters(
                emission_df=emission_df,
                labels=labels,
                sample_name="sample_a",
                chrom="chr1",
            )

        expected_input = emission_df[["feat_a", "feat_b"]].to_numpy()
        np.testing.assert_allclose(
            imputer.transform.call_args.args[0],
            expected_input,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            scaler.transform.call_args.args[0], imputed_features
        )
        np.testing.assert_allclose(
            umap_runner.fit_transform.call_args.args[0], scaled_features
        )
        umap_ctor.assert_called_once_with(n_components=2, random_state=None)

    def test_plot_umap_clusters_can_use_pca_features(self):
        emission_df = self.make_umap_emissions()
        scaled_features = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        pca_scores = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )
        imputed_features = np.array(
            [
                [0.1, 1.0],
                [0.2, 2.0],
                [0.3, 3.0],
            ]
        )
        imputer = mock.Mock()
        imputer.transform.return_value = imputed_features
        scaler = mock.Mock()
        scaler.transform.return_value = scaled_features
        pca = mock.Mock()
        pca.transform.return_value = pca_scores
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=imputer,
            pca=pca,
        )

        umap_runner = mock.Mock()
        umap_runner.fit_transform.return_value = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.umap.UMAP",
            return_value=umap_runner,
        ), mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_umap_clusters(
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                use_pca=True,
            )

        np.testing.assert_allclose(pca.transform.call_args.args[0], scaled_features)
        np.testing.assert_allclose(
            umap_runner.fit_transform.call_args.args[0], pca_scores
        )

    def test_plot_umap_clusters_requires_trained_model(self):
        assigner = MethylStateAssigner()

        with self.assertRaisesRegex(ValueError, "No trained model found"):
            assigner.plot_umap_clusters(
                emission_df=self.make_umap_emissions(),
                labels=np.array([0, 1, 2]),
            )

    def test_plot_umap_clusters_requires_pca_model_when_requested(self):
        imputer = mock.Mock()
        imputer.transform.return_value = np.array([[0.1, 1.0]])
        scaler = mock.Mock()
        scaler.transform.return_value = np.array([[0.0, 0.1]])
        assigner = self.make_umap_assigner(scaler=scaler, imputer=imputer, pca=None)

        with self.assertRaisesRegex(ValueError, "n_pca > 0"):
            assigner.plot_umap_clusters(
                emission_df=pd.DataFrame({"feat_a": [0.1], "feat_b": [1.0]}),
                labels=np.array([0]),
                use_pca=True,
            )

    def test_calculate_kmeans_cluster_metrics_uses_scaled_features_for_raw_models(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3, 0.4],
                "feat_b": [1.0, 2.0, 3.0, 4.0],
            }
        )
        labels = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.PMR,
                MethylationStates.INTERMEDIATE,
                MethylationStates.PMR,
            ],
            dtype=object,
        )
        scaled_features = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
                [1.5, 1.6],
            ]
        )
        scaler = mock.Mock()
        scaler.transform.return_value = scaled_features
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.silhouette_score",
            return_value=0.412,
        ) as silhouette_mock, mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.davies_bouldin_score",
            return_value=0.873,
        ) as db_mock:
            metrics = assigner.calculate_kmeans_cluster_metrics(
                emission_df=emission_df,
                labels=labels,
            )

        np.testing.assert_allclose(silhouette_mock.call_args.args[0], scaled_features)
        np.testing.assert_array_equal(
            silhouette_mock.call_args.args[1],
            np.array([0, 1, 2, 1]),
        )
        np.testing.assert_allclose(db_mock.call_args.args[0], scaled_features)
        np.testing.assert_array_equal(
            db_mock.call_args.args[1],
            np.array([0, 1, 2, 1]),
        )
        self.assertEqual(metrics["silhouette_score"], 0.412)
        self.assertEqual(metrics["davies_bouldin_score"], 0.873)

    def test_calculate_kmeans_cluster_metrics_uses_pca_scores_for_pca_models(self):
        emission_df = pd.DataFrame(
            [
                {"feat_a": 0.1, "feat_b": 1.0},
                {"feat_a": 0.2, "feat_b": 2.0},
                {"feat_a": 0.3, "feat_b": 3.0},
                {"feat_a": 0.4, "feat_b": 4.0},
            ]
        )
        scaled_features = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
                [1.5, 1.6],
            ]
        )
        trained_scores = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
            ]
        )
        scaler = mock.Mock()
        scaler.transform.return_value = scaled_features
        pca = mock.Mock()
        pca.transform.return_value = trained_scores
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=pca,
            cluster_space="pca",
            n_pca=2,
        )

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.silhouette_score",
            return_value=0.512,
        ) as silhouette_mock, mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.davies_bouldin_score",
            return_value=0.743,
        ) as db_mock:
            metrics = assigner.calculate_kmeans_cluster_metrics(
                emission_df=emission_df,
                labels=np.array([0, 1, 2, 1]),
            )

        np.testing.assert_allclose(pca.transform.call_args.args[0], scaled_features)
        np.testing.assert_allclose(silhouette_mock.call_args.args[0], trained_scores)
        np.testing.assert_array_equal(
            silhouette_mock.call_args.args[1],
            np.array([0, 1, 2, 1]),
        )
        np.testing.assert_allclose(db_mock.call_args.args[0], trained_scores)
        np.testing.assert_array_equal(db_mock.call_args.args[1], np.array([0, 1, 2, 1]))
        self.assertEqual(metrics["silhouette_score"], 0.512)
        self.assertEqual(metrics["davies_bouldin_score"], 0.743)

    def test_plot_pca_clusters_falls_back_to_temporary_pca_when_model_has_no_pca(self):
        emission_df = self.make_umap_emissions()
        imputed_features = np.array(
            [
                [0.1, 1.0],
                [0.2, 2.0],
                [0.3, 3.0],
            ]
        )
        scaled_features = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        imputer = mock.Mock()
        imputer.transform.return_value = imputed_features
        scaler = mock.Mock()
        scaler.transform.return_value = scaled_features
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=imputer,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )

        with mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                n_pca_plot=2,
            )

    def test_plot_pca_clusters_reuses_trained_pca_when_available(self):
        emission_df = self.make_umap_emissions()
        imputed_features = np.array(
            [
                [0.1, 1.0],
                [0.2, 2.0],
                [0.3, 3.0],
            ]
        )
        scaled_features = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        trained_scores = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )
        imputer = mock.Mock()
        imputer.transform.return_value = imputed_features
        scaler = mock.Mock()
        scaler.transform.return_value = scaled_features
        pca = mock.Mock()
        pca.n_components_ = 2
        pca.explained_variance_ratio_ = np.array([0.7, 0.3])
        pca.components_ = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        )
        pca.transform.return_value = trained_scores
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=imputer,
            pca=pca,
            cluster_space="pca",
            n_pca=2,
        )

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.PCA",
            side_effect=AssertionError("temporary PCA should not be created"),
        ), mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                n_pca_plot=2,
            )

        np.testing.assert_allclose(pca.transform.call_args.args[0], scaled_features)

    def test_plot_train_pca_clusters_works_without_saved_train_pca_scores(self):
        emission_df = self.make_umap_emissions()
        imputed_features = np.array(
            [
                [0.1, 1.0],
                [0.2, 2.0],
                [0.3, 3.0],
            ]
        )
        scaled_features = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        imputer = mock.Mock()
        imputer.transform.return_value = imputed_features
        scaler = mock.Mock()
        scaler.transform.return_value = scaled_features
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=imputer,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )
        assigner.train_emission_df = emission_df
        assigner.train_labels = np.array([0, 1, 2])
        assigner.train_sample = "sample_a"
        assigner.train_chroms = ["chr1"]

        with mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_train_pca_clusters(n_pca_plot=2)

    def test_plot_pca_clusters_wraps_long_feature_names_in_loadings_table(self):
        long_feature_a = "region_window_super_long_feature_name_for_pc1_loading_display"
        long_feature_b = "another_extremely_verbose_feature_name_for_pc2_loading_display"
        emission_df = pd.DataFrame(
            {
                long_feature_a: [0.1, 0.2, 0.3],
                long_feature_b: [1.0, 2.0, 3.0],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = emission_df.to_numpy()
        pca = mock.Mock()
        pca.n_components_ = 2
        pca.explained_variance_ratio_ = np.array([0.7, 0.3])
        pca.components_ = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ]
        )
        pca.transform.return_value = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=pca,
            cluster_space="pca",
            n_pca=2,
        )
        assigner.model.feature_cols = [long_feature_a, long_feature_b]

        with mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                n_pca_plot=2,
            )

        fig = plt.gcf()
        table_ax = next(ax for ax in fig.axes if len(ax.tables) > 0)
        table = next(iter(table_ax.tables))
        self.assertIn("\n", table[(1, 0)].get_text().get_text())

    def test_plot_pca_clusters_adds_metrics_textbox_to_main_axis(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )

        with mock.patch.object(
            assigner,
            "calculate_kmeans_cluster_metrics",
            return_value={
                "silhouette_score": 0.412,
                "davies_bouldin_score": 0.873,
            },
        ), mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                n_pca_plot=2,
            )

        fig = plt.gcf()
        main_ax = next(ax for ax in fig.axes if ax.get_xlabel().startswith("PC1"))
        textbox_texts = [text.get_text() for text in main_ax.texts]
        self.assertTrue(
            any(
                "Silhouette: 0.412" in text and "Davies-Bouldin: 0.873" in text
                for text in textbox_texts
            )
        )

    def test_plot_pca_clusters_can_skip_kmeans_metrics(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )

        with mock.patch.object(
            assigner,
            "calculate_kmeans_cluster_metrics",
            side_effect=AssertionError("cluster metrics should be skipped"),
        ), mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                n_pca_plot=2,
                include_kmeans_metrics=False,
            )

        fig = plt.gcf()
        main_ax = next(ax for ax in fig.axes if ax.get_xlabel().startswith("PC1"))
        textbox_texts = [text.get_text() for text in main_ax.texts]
        self.assertFalse(any("Silhouette:" in text for text in textbox_texts))

    def test_plot_pca_clusters_shows_na_metrics_when_cluster_metrics_are_undefined(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )

        with mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 0, 0]),
                n_pca_plot=2,
            )

        fig = plt.gcf()
        main_ax = next(ax for ax in fig.axes if ax.get_xlabel().startswith("PC1"))
        textbox_texts = [text.get_text() for text in main_ax.texts]
        self.assertTrue(
            any(
                "Silhouette: n/a" in text and "Davies-Bouldin: n/a" in text
                for text in textbox_texts
            )
        )

    def test_plot_pca_clusters_with_region_highlights_expected_rows(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        meth_data = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
            }
        )
        trained_scores = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )
        scaler = mock.Mock()
        scaler.transform.return_value = emission_df.to_numpy()
        pca = mock.Mock()
        pca.n_components_ = 2
        pca.explained_variance_ratio_ = np.array([0.7, 0.3])
        pca.components_ = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        )
        pca.transform.return_value = trained_scores
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=pca,
            cluster_space="pca",
            n_pca=2,
        )

        with mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters_with_region(
                meth_data=meth_data,
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                region_start=15,
                region_end=30,
                n_pca_plot=2,
            )

        fig = plt.gcf()
        main_ax = next(ax for ax in fig.axes if "PCA + KMeans States" in ax.get_title())
        highlight_offsets = np.asarray(main_ax.collections[-1].get_offsets())
        np.testing.assert_allclose(highlight_offsets, trained_scores[[1, 2], :2])
        legend = main_ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertIn(
            "Highlighted region",
            [text.get_text() for text in legend.get_texts()],
        )

    def test_plot_pca_clusters_with_region_requires_region_chrom_for_multi_chrom(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        meth_data = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr2", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = emission_df.to_numpy()
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )

        with self.assertRaisesRegex(ValueError, "region_chrom is required"):
            assigner.plot_pca_clusters_with_region(
                meth_data=meth_data,
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                region_start=10,
                region_end=20,
            )

    def test_plot_pca_clusters_with_region_warns_when_interval_has_no_matches(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        meth_data = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = emission_df.to_numpy()
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )

        with self.assertWarnsRegex(RuntimeWarning, "No CpGs overlapped"), mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.plt.show"
        ) as show_mock:
            assigner.plot_pca_clusters_with_region(
                meth_data=meth_data,
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                region_start=100,
                region_end=120,
            )

        show_mock.assert_called_once()

    def test_plot_train_pca_clusters_with_region_uses_saved_train_meth(self):
        emission_df = self.make_umap_emissions()
        scaler = mock.Mock()
        scaler.transform.return_value = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )
        assigner.train_meth = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
            }
        )
        assigner.train_emission_df = emission_df
        assigner.train_labels = np.array([0, 1, 2])
        assigner.train_sample = "sample_a"
        assigner.train_chroms = ["chr1"]

        with mock.patch.object(
            assigner,
            "plot_pca_clusters_with_region",
            return_value=None,
        ) as wrapper_mock:
            assigner.plot_train_pca_clusters(
                n_pca_plot=2,
                region_start=15,
                region_end=25,
                include_kmeans_metrics=False,
            )

        self.assertIs(wrapper_mock.call_args.kwargs["meth_data"], assigner.train_meth)
        self.assertEqual(wrapper_mock.call_args.kwargs["region_start"], 15)
        self.assertEqual(wrapper_mock.call_args.kwargs["region_end"], 25)
        self.assertFalse(wrapper_mock.call_args.kwargs["include_kmeans_metrics"])

    def test_plot_train_pca_clusters_forwards_include_kmeans_metrics(self):
        emission_df = self.make_umap_emissions()
        scaler = mock.Mock()
        scaler.transform.return_value = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )
        assigner.train_emission_df = emission_df
        assigner.train_labels = np.array([0, 1, 2])

        with mock.patch.object(
            assigner,
            "plot_pca_clusters",
            return_value=None,
        ) as wrapper_mock:
            assigner.plot_train_pca_clusters(
                n_pca_plot=2,
                include_kmeans_metrics=False,
            )

        self.assertFalse(wrapper_mock.call_args.kwargs["include_kmeans_metrics"])

    def test_plot_train_pca_clusters_requires_region_bounds_for_region_highlighting(self):
        scaler = mock.Mock()
        scaler.transform.return_value = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )
        assigner.train_meth = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
            }
        )
        assigner.train_emission_df = self.make_umap_emissions()
        assigner.train_labels = np.array([0, 1, 2])

        with self.assertRaisesRegex(ValueError, "region_start and region_end must both"):
            assigner.plot_train_pca_clusters(region_chrom="chr1")

    def test_plot_pca_clusters_with_region_rejects_unsupported_modes(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        meth_data = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
            }
        )

        interactive_scaler = mock.Mock()
        interactive_scaler.transform.return_value = emission_df.to_numpy()
        interactive_assigner = self.make_umap_assigner(
            scaler=interactive_scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )
        with self.assertRaisesRegex(ValueError, "2-D non-interactive PCA plots"):
            interactive_assigner.plot_pca_clusters_with_region(
                meth_data=meth_data,
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                region_start=10,
                region_end=20,
                interactive=True,
            )

        pca_scaler = mock.Mock()
        pca_scaler.transform.return_value = emission_df.to_numpy()
        pca = mock.Mock()
        pca.n_components_ = 3
        pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        pca.components_ = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.5, 0.5],
            ]
        )
        pca.transform.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.4, 0.5],
                [0.5, 0.6, 0.7],
            ]
        )
        pca_assigner = self.make_umap_assigner(
            scaler=pca_scaler,
            imputer=None,
            pca=pca,
            cluster_space="pca",
            n_pca=3,
        )
        with self.assertRaisesRegex(ValueError, "2-D non-interactive PCA plots"):
            pca_assigner.plot_pca_clusters_with_region(
                meth_data=meth_data,
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                region_start=10,
                region_end=20,
                n_pca_plot=3,
            )

    def test_plot_pca_clusters_uses_present_biological_states_for_sparse_labels(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = emission_df[["feat_a", "feat_b"]].to_numpy()
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )
        fake_colorbar = _FakeColorbar()

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.plt.colorbar",
            return_value=fake_colorbar,
        ) as colorbar_mock, mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.plt.show"
        ):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 2, 3]),
                n_pca_plot=2,
            )

        self.assertListEqual(list(colorbar_mock.call_args.kwargs["ticks"]), [0, 2, 3])
        self.assertEqual(
            fake_colorbar.ticklabels,
            ["LOW", "INTERMEDIATE", "HIGH"],
        )

    def test_plot_pca_clusters_interactive_3d_adds_metrics_annotation(self):
        emission_df = pd.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        scaler = mock.Mock()
        scaler.transform.return_value = np.array(
            [
                [0.0, 0.1],
                [0.5, 0.6],
                [1.0, 1.1],
            ]
        )
        pca = mock.Mock()
        pca.n_components_ = 3
        pca.explained_variance_ratio_ = np.array([0.6, 0.3, 0.1])
        pca.components_ = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.5, 0.5],
            ]
        )
        pca.transform.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.4, 0.5],
                [0.5, 0.6, 0.7],
            ]
        )
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=pca,
            cluster_space="pca",
            n_pca=3,
        )
        fake_fig = _FakeFigure()

        with mock.patch.object(
            assigner,
            "calculate_kmeans_cluster_metrics",
            return_value={
                "silhouette_score": 0.412,
                "davies_bouldin_score": 0.873,
            },
        ), mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.px.scatter_3d",
            return_value=fake_fig,
        ), mock.patch("analysis.shared_utils.methyl_seg.methyl_seg.plt.show"):
            assigner.plot_pca_clusters(
                emission_df=emission_df,
                labels=np.array([0, 1, 2]),
                n_pca_plot=3,
                interactive=True,
            )

        annotation = next(
            update["annotations"][0]
            for update in fake_fig.layout_updates
            if "annotations" in update
        )
        self.assertEqual(annotation["x"], 0.98)
        self.assertEqual(annotation["y"], 0.02)
        self.assertEqual(annotation["xref"], "paper")
        self.assertEqual(annotation["yref"], "paper")
        self.assertIn("Silhouette: 0.412", annotation["text"])
        self.assertIn("Davies-Bouldin: 0.873", annotation["text"])
        self.assertIn("<br>", annotation["text"])

    def test_plot_kmeans_clusters_interactive_uses_state_names_for_sparse_labels(self):
        scaler = mock.Mock()
        scaler.transform.return_value = np.eye(3)
        assigner = self.make_umap_assigner(
            scaler=scaler,
            imputer=None,
            pca=None,
            cluster_space="raw",
            n_pca=None,
        )
        fake_fig = _FakeFigure()
        meth_data = pd.DataFrame(
            {
                "CpG_beg": [10, 20, 30],
                "beta": [0.1, 0.5, 0.9],
            }
        )

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.px.scatter",
            return_value=fake_fig,
        ) as scatter_mock:
            assigner.plot_kmeans_clusters(
                meth_data=meth_data,
                labels=np.array([0, 2, 3]),
                interactive=True,
            )

        kwargs = scatter_mock.call_args.kwargs
        self.assertEqual(set(kwargs["color"]), {"LOW", "INTERMEDIATE", "HIGH"})
        self.assertEqual(
            set(kwargs["color_discrete_map"].keys()),
            {"LOW", "INTERMEDIATE", "HIGH"},
        )
        self.assertEqual(
            kwargs["category_orders"]["color"],
            ["LOW", "INTERMEDIATE", "HIGH"],
        )

    def test_analyzer_plot_interactive_beta_by_label_supports_pmr_only_colors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = self.make_analyzer(out_dir=tmpdir)
            fake_fig = _FakeFigure()

            with mock.patch(
                "analysis.shared_utils.methyl_seg.methyl_seg.px.scatter",
                return_value=fake_fig,
            ) as scatter_mock:
                analyzer.plot_interactive_beta_by_label(
                    show_plot=False,
                    color_pmr_only=True,
                )

            kwargs = scatter_mock.call_args.kwargs
            df_plot = kwargs["data_frame"]
            self.assertEqual(kwargs["color"], "kmeans_label_pmr_status")
            self.assertEqual(
                kwargs["color_discrete_map"],
                {"PMR": "#d62728", "non-PMR": "#1f77b4"},
            )
            self.assertEqual(
                kwargs["category_orders"]["kmeans_label_pmr_status"],
                ["PMR", "non-PMR"],
            )
            self.assertEqual(
                set(df_plot["kmeans_label_pmr_status"]), {"PMR", "non-PMR"}
            )
            self.assertEqual(fake_fig.layout_updates[-1]["legend_title_text"], "PMR status")
            self.assertFalse(fake_fig.for_each_trace_called)
            self.assertTrue(
                fake_fig.write_html_path.endswith(
                    "interactive_beta_by_kmeans_label_pmr_only.html"
                )
            )

    def test_analyzer_plot_interactive_beta_by_label_preserves_state_colors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = self.make_analyzer(out_dir=tmpdir)
            fake_fig = _FakeFigure()

            with mock.patch(
                "analysis.shared_utils.methyl_seg.methyl_seg.px.scatter",
                return_value=fake_fig,
            ) as scatter_mock:
                analyzer.plot_interactive_beta_by_label(show_plot=False)

            kwargs = scatter_mock.call_args.kwargs
            df_plot = kwargs["data_frame"]
            self.assertEqual(kwargs["color"], "kmeans_label")
            self.assertEqual(set(df_plot["kmeans_label"]), {"0", "1", "3"})
            self.assertEqual(
                set(kwargs["color_discrete_map"].keys()), {"0", "1", "3"}
            )
            self.assertEqual(
                kwargs["category_orders"]["kmeans_label"], ["0", "1", "3"]
            )
            self.assertEqual(fake_fig.layout_updates[-1]["legend_title_text"], "State")
            self.assertTrue(fake_fig.for_each_trace_called)
            self.assertTrue(
                fake_fig.write_html_path.endswith("interactive_beta_by_kmeans_label.html")
            )

    def test_segmenter_plot_interactive_beta_by_label_supports_pmr_only_colors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = self.make_segmenter(out_dir=tmpdir)
            fake_fig = _FakeFigure()
            meth_data = self.make_segmenter_meth_data()

            with mock.patch.object(
                segmenter,
                "segment_sample",
                return_value=(meth_data, None),
            ), mock.patch(
                "analysis.shared_utils.methyl_seg.methyl_seg.px.scatter",
                return_value=fake_fig,
            ) as scatter_mock:
                segmenter.plot_interactive_beta_by_label(
                    sample_info=SampleInfo(sample_id="seg_sample", meth_data=meth_data),
                    show_plot=False,
                    color_pmr_only=True,
                )

            kwargs = scatter_mock.call_args.kwargs
            df_plot = kwargs["data_frame"]
            self.assertEqual(kwargs["color"], "hmm_state_readable_pmr_status")
            self.assertEqual(
                kwargs["color_discrete_map"],
                {"PMR": "#d62728", "non-PMR": "#1f77b4"},
            )
            self.assertEqual(
                kwargs["category_orders"]["hmm_state_readable_pmr_status"],
                ["PMR", "non-PMR"],
            )
            self.assertEqual(
                set(df_plot["hmm_state_readable_pmr_status"]), {"PMR", "non-PMR"}
            )
            self.assertEqual(fake_fig.layout_updates[-1]["legend_title_text"], "PMR status")
            self.assertFalse(fake_fig.for_each_trace_called)
            self.assertTrue(
                fake_fig.write_html_path.endswith(
                    "interactive_beta_by_hmm_state_readable_pmr_only.html"
                )
            )

    def test_segmenter_plot_interactive_beta_by_label_preserves_state_colors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = self.make_segmenter(out_dir=tmpdir)
            fake_fig = _FakeFigure()
            meth_data = self.make_segmenter_meth_data()

            with mock.patch.object(
                segmenter,
                "segment_sample",
                return_value=(meth_data, None),
            ), mock.patch(
                "analysis.shared_utils.methyl_seg.methyl_seg.px.scatter",
                return_value=fake_fig,
            ) as scatter_mock:
                segmenter.plot_interactive_beta_by_label(
                    sample_info=SampleInfo(sample_id="seg_sample", meth_data=meth_data),
                    show_plot=False,
                )

            kwargs = scatter_mock.call_args.kwargs
            df_plot = kwargs["data_frame"]
            self.assertEqual(kwargs["color"], "hmm_state_readable")
            self.assertEqual(set(df_plot["hmm_state_readable"]), {"0", "1", "3"})
            self.assertEqual(
                set(kwargs["color_discrete_map"].keys()), {"0", "1", "3"}
            )
            self.assertEqual(
                kwargs["category_orders"]["hmm_state_readable"], ["0", "1", "3"]
            )
            self.assertEqual(fake_fig.layout_updates[-1]["legend_title_text"], "State")
            self.assertTrue(fake_fig.for_each_trace_called)
            self.assertTrue(
                fake_fig.write_html_path.endswith(
                    "interactive_beta_by_hmm_state_readable.html"
                )
            )


class SegmenterEncodingTests(unittest.TestCase):
    def make_sample_info(self) -> SampleInfo:
        meth_df = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr1"],
                "CpG_beg": [10, 20, 30],
                "CpG_end": [11, 21, 31],
                "beta": [0.1, 0.5, 0.9],
            }
        )
        return SampleInfo(sample_id="seg_sample", meth_data=meth_df)

    def test_segment_sample_compacts_sparse_state_codes_before_hmm(self):
        fit_inputs = []
        predict_inputs = []

        class _FakeHMM:
            def __init__(self):
                self.n_states = 3
                self.hmm_model = object()

            def create_model(self):
                return None

            def fit(self, emissions, sample_info=None, chrom=None):
                fit_inputs.append(np.asarray(emissions, dtype=int).copy())
                return None

            def predict(self, emissions):
                predict_inputs.append(np.asarray(emissions, dtype=int).copy())
                return np.asarray(emissions, dtype=int)

        assigner = mock.Mock()
        assigner.relabel_by_mean_emission.return_value = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.INTERMEDIATE,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        analyzer = mock.Mock()
        analyzer.assigner = assigner
        segmenter = MethylSegmenter(
            analyzer=analyzer,
            hmm_model=_FakeHMM(),
            out_dir=".",
        )
        sample_info = self.make_sample_info()
        meth_data = sample_info.meth_data.copy()
        meth_data["state"] = np.array([0, 2, 3], dtype=int)
        meth_data["state_readable"] = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.INTERMEDIATE,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        emissions_df = pd.DataFrame({"beta": meth_data["beta"]})

        def _assign_states_side_effect(sample_info, chrom):
            segmenter.meth_data = meth_data.copy()
            segmenter.emissions_df = emissions_df.copy()
            return segmenter.meth_data, segmenter.emissions_df

        segmenter.assign_states = mock.Mock(side_effect=_assign_states_side_effect)

        result_meth_data, _ = segmenter.segment_sample(sample_info=sample_info, chrom="chr1")

        np.testing.assert_array_equal(fit_inputs[0], np.array([0, 1, 2], dtype=int))
        np.testing.assert_array_equal(
            predict_inputs[0],
            np.array([0, 1, 2], dtype=int),
        )
        np.testing.assert_array_equal(
            result_meth_data["hmm_state"].to_numpy(dtype=int),
            np.array([0, 1, 2], dtype=int),
        )
        np.testing.assert_array_equal(
            result_meth_data["hmm_state_readable"].to_numpy(dtype=object),
            np.array(
                [
                    MethylationStates.LOW,
                    MethylationStates.INTERMEDIATE,
                    MethylationStates.HIGH,
                ],
                dtype=object,
            ),
        )

    def test_segment_sample_sets_regions_df_from_hmm_states(self):
        class _FakeHMM:
            def __init__(self):
                self.n_states = 2
                self.hmm_model = object()

            def create_model(self):
                return None

            def fit(self, emissions, sample_info=None, chrom=None):
                return None

            def predict(self, emissions):
                return np.array([0, 0, 1], dtype=int)

        assigner = mock.Mock()
        assigner.relabel_by_mean_emission.return_value = np.array(
            [
                MethylationStates.PMR,
                MethylationStates.PMR,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        analyzer = mock.Mock()
        analyzer.assigner = assigner
        segmenter = MethylSegmenter(
            analyzer=analyzer,
            hmm_model=_FakeHMM(),
            out_dir=".",
        )
        sample_info = self.make_sample_info()
        meth_data = sample_info.meth_data.copy()
        meth_data["state"] = np.array([0, 0, 3], dtype=int)
        meth_data["state_readable"] = np.array(
            [
                MethylationStates.PMR,
                MethylationStates.PMR,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        emissions_df = pd.DataFrame({"beta": meth_data["beta"]})

        def _assign_states_side_effect(sample_info, chrom):
            segmenter.meth_data = meth_data.copy()
            segmenter.emissions_df = emissions_df.copy()
            return segmenter.meth_data, segmenter.emissions_df

        segmenter.assign_states = mock.Mock(side_effect=_assign_states_side_effect)

        segmenter.segment_sample(sample_info=sample_info, chrom="chr1")

        self.assertTrue(hasattr(segmenter, "regions_df"))
        self.assertEqual(segmenter.regions_df["start"].tolist(), [10, 30])
        self.assertEqual(segmenter.regions_df["end"].tolist(), [21, 31])
        self.assertEqual(segmenter.regions_df["probe_count"].tolist(), [2, 1])
        np.testing.assert_array_equal(
            segmenter.regions_df["state"].to_numpy(dtype=object),
            np.array([MethylationStates.PMR, MethylationStates.HIGH], dtype=object),
        )


class GaussianObservationModeSegmenterTests(unittest.TestCase):
    class _FakeGaussianHMM:
        def __init__(self, predicted_states):
            self.n_states = 4
            self.hmm_model = object()
            self.predicted_states = np.asarray(predicted_states, dtype=int)
            self.init_calls = []
            self.fit_inputs = []
            self.predict_inputs = []

        def create_model(self):
            self.hmm_model = object()

        def initialize_from_kmeans(self, X_scaled, km_labels, lengths=None):
            self.init_calls.append(
                {
                    "X_scaled": np.asarray(X_scaled, dtype=float).copy(),
                    "km_labels": np.asarray(km_labels, dtype=int).copy(),
                    "lengths": None if lengths is None else list(lengths),
                }
            )

        def fit(self, emissions, sample_info=None, chrom=None):
            self.fit_inputs.append(np.asarray(emissions, dtype=float).copy())

        def predict(self, emissions):
            self.predict_inputs.append(np.asarray(emissions, dtype=float).copy())
            return self.predicted_states.copy()

    def make_sample_info(self, chroms=None) -> SampleInfo:
        if chroms is None:
            chroms = ["chr1", "chr1", "chr1", "chr1"]
        positions = [10, 20, 30, 40][: len(chroms)]
        meth_df = pd.DataFrame(
            {
                "CpG_chrm": chroms,
                "CpG_beg": positions,
                "CpG_end": [pos + 1 for pos in positions],
                "beta": np.linspace(0.1, 0.8, num=len(chroms)),
            }
        )
        return SampleInfo(sample_id="gaussian_sample", meth_data=meth_df)

    def make_emission_df(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "beta": np.linspace(0.1, 0.8, num=n_rows),
                "feat_a": np.linspace(1.0, 4.0, num=n_rows),
                "feat_b": np.linspace(10.0, 40.0, num=n_rows),
                "feat_count_n_cpg": np.arange(1, n_rows + 1, dtype=float),
            }
        )

    def make_segmenter(
        self,
        meth_data: pd.DataFrame,
        emission_df: pd.DataFrame,
        hmm_model,
    ):
        assigner = MethylStateAssigner(random_state=17, window_specs=[(20, "20bp")])
        assigner.prepare_sample_for_clustering = mock.Mock(
            return_value=(
                meth_data.copy(),
                emission_df.to_numpy(dtype=float, copy=True),
                emission_df.copy(),
            )
        )

        analyzer = mock.Mock()
        analyzer.assigner = assigner
        analyzer.state_cutoffs = None

        segmenter = MethylSegmenter(
            analyzer=analyzer,
            hmm_model=hmm_model,
            hmm_observation_mode=HMMObservationMode.GAUSSIAN_EMISSIONS,
            out_dir=".",
        )
        return segmenter, assigner

    def test_segment_sample_gaussian_mode_uses_scaled_2d_emissions(self):
        sample_info = self.make_sample_info()
        emission_df = self.make_emission_df(len(sample_info.meth_data))
        fake_hmm = self._FakeGaussianHMM(predicted_states=[0, 1, 1, 2])
        segmenter, assigner = self.make_segmenter(
            meth_data=sample_info.meth_data,
            emission_df=emission_df,
            hmm_model=fake_hmm,
        )
        raw_init_labels = np.array([3, 3, 1, 1], dtype=int)
        init_readable = np.array(
            [
                MethylationStates.HIGH,
                MethylationStates.HIGH,
                MethylationStates.INTERMEDIATE,
                MethylationStates.INTERMEDIATE,
            ],
            dtype=object,
        )
        final_readable = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.PMR,
                MethylationStates.PMR,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        assigner.model = object()
        assigner.apply_kmeans_to_emissions = mock.Mock(
            return_value=(None, None, raw_init_labels, None)
        )
        assigner.relabel_by_mean_emission = mock.Mock(
            side_effect=[init_readable, final_readable]
        )

        result_meth_data, _ = segmenter.segment_sample(sample_info=sample_info, chrom="chr1")

        self.assertEqual(len(fake_hmm.init_calls), 1)
        self.assertEqual(len(fake_hmm.fit_inputs), 1)
        self.assertEqual(len(fake_hmm.predict_inputs), 1)
        self.assertEqual(fake_hmm.init_calls[0]["X_scaled"].ndim, 2)
        self.assertEqual(fake_hmm.fit_inputs[0].ndim, 2)
        self.assertEqual(fake_hmm.predict_inputs[0].ndim, 2)
        self.assertEqual(
            fake_hmm.fit_inputs[0].shape[1],
            len(emission_df.columns),
        )
        np.testing.assert_array_equal(
            fake_hmm.init_calls[0]["km_labels"],
            raw_init_labels,
        )
        np.testing.assert_array_equal(
            result_meth_data["state"].to_numpy(dtype=int),
            MethylationStates.convert_to_numeric(init_readable),
        )
        np.testing.assert_array_equal(
            result_meth_data["state_readable"].to_numpy(dtype=object),
            init_readable,
        )
        np.testing.assert_array_equal(
            result_meth_data["hmm_state"].to_numpy(dtype=int),
            np.array([0, 1, 1, 2], dtype=int),
        )
        np.testing.assert_array_equal(
            result_meth_data["hmm_state_readable"].to_numpy(dtype=object),
            final_readable,
        )

    def test_gaussian_mode_reuses_pretrained_kmeans_labels_when_available(self):
        sample_info = self.make_sample_info()
        emission_df = self.make_emission_df(len(sample_info.meth_data))
        fake_hmm = self._FakeGaussianHMM(predicted_states=[0, 0, 1, 1])
        segmenter, assigner = self.make_segmenter(
            meth_data=sample_info.meth_data,
            emission_df=emission_df,
            hmm_model=fake_hmm,
        )
        raw_init_labels = np.array([2, 2, 1, 1], dtype=int)
        assigner.model = object()
        assigner.apply_kmeans_to_emissions = mock.Mock(
            return_value=(None, None, raw_init_labels, None)
        )
        assigner.relabel_by_mean_emission = mock.Mock(
            side_effect=[
                np.array([MethylationStates.HIGH] * 4, dtype=object),
                np.array([MethylationStates.INTERMEDIATE] * 4, dtype=object),
            ]
        )

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.KMeans",
            side_effect=AssertionError("fallback KMeans should not be used"),
        ):
            segmenter.segment_sample(sample_info=sample_info, chrom="chr1")

        assigner.apply_kmeans_to_emissions.assert_called_once()
        np.testing.assert_array_equal(
            fake_hmm.init_calls[0]["km_labels"],
            raw_init_labels,
        )

    def test_gaussian_mode_falls_back_to_kmeans_when_no_pretrained_model(self):
        sample_info = self.make_sample_info()
        emission_df = self.make_emission_df(len(sample_info.meth_data))
        fake_hmm = self._FakeGaussianHMM(predicted_states=[0, 1, 2, 3])
        segmenter, assigner = self.make_segmenter(
            meth_data=sample_info.meth_data,
            emission_df=emission_df,
            hmm_model=fake_hmm,
        )
        assigner.model = None
        assigner.apply_kmeans_to_emissions = mock.Mock(
            side_effect=AssertionError("pretrained KMeans should not be used")
        )
        assigner.relabel_by_mean_emission = mock.Mock(
            side_effect=[
                np.array([MethylationStates.LOW] * 4, dtype=object),
                np.array([MethylationStates.HIGH] * 4, dtype=object),
            ]
        )
        kmeans_instance = mock.Mock()
        kmeans_instance.fit_predict.return_value = np.array([0, 0, 1, 1], dtype=int)

        with mock.patch(
            "analysis.shared_utils.methyl_seg.methyl_seg.KMeans",
            return_value=kmeans_instance,
        ) as kmeans_cls:
            segmenter.segment_sample(sample_info=sample_info, chrom="chr1")

        kmeans_cls.assert_called_once_with(
            n_clusters=fake_hmm.n_states,
            n_init=10,
            random_state=segmenter.random_state,
        )
        kmeans_instance.fit_predict.assert_called_once()
        self.assertEqual(kmeans_instance.fit_predict.call_args[0][0].ndim, 2)

    def test_gaussian_mode_derives_sequence_lengths_for_multi_chrom_sample(self):
        sample_info = self.make_sample_info(chroms=["chr1", "chr1", "chr2", "chr2"])
        emission_df = self.make_emission_df(len(sample_info.meth_data))
        fake_hmm = self._FakeGaussianHMM(predicted_states=[0, 0, 1, 1])
        segmenter, assigner = self.make_segmenter(
            meth_data=sample_info.meth_data,
            emission_df=emission_df,
            hmm_model=fake_hmm,
        )
        assigner.model = object()
        assigner.apply_kmeans_to_emissions = mock.Mock(
            return_value=(None, None, np.array([0, 0, 1, 1], dtype=int), None)
        )
        assigner.relabel_by_mean_emission = mock.Mock(
            side_effect=[
                np.array([MethylationStates.LOW] * 4, dtype=object),
                np.array([MethylationStates.HIGH] * 4, dtype=object),
            ]
        )

        segmenter.segment_sample(sample_info=sample_info, chrom=None)

        self.assertEqual(fake_hmm.init_calls[0]["lengths"], [2, 2])


class DiscreteObservationModeSegmenterTests(unittest.TestCase):
    class _FakeDiscreteHMM:
        def __init__(self):
            self.n_states = 4
            self.hmm_model = object()
            self.lengths = None
            self.fit_inputs = []
            self.fit_lengths = []
            self.predict_inputs = []
            self.predict_lengths = []

        def create_model(self):
            self.hmm_model = object()

        def fit(self, emissions, sample_info=None, chrom=None):
            self.fit_inputs.append(np.asarray(emissions, dtype=int).copy())
            self.fit_lengths.append(
                None if self.lengths is None else list(self.lengths)
            )

        def predict(self, emissions):
            emissions = np.asarray(emissions, dtype=int)
            self.predict_inputs.append(emissions.copy())
            self.predict_lengths.append(
                None if self.lengths is None else list(self.lengths)
            )
            return emissions.copy()

    def make_sample_info(self) -> SampleInfo:
        meth_df = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr2", "chr2"],
                "CpG_beg": [10, 20, 30, 40],
                "CpG_end": [11, 21, 31, 41],
                "beta": [0.1, 0.4, 0.6, 0.9],
            }
        )
        return SampleInfo(sample_id="discrete_sample", meth_data=meth_df)

    def test_segment_sample_multi_chrom_sets_lengths_for_discrete_hmm(self):
        fake_hmm = self._FakeDiscreteHMM()
        sample_info = self.make_sample_info()
        meth_data = sample_info.meth_data.copy()
        meth_data["state"] = np.array([0, 2, 3, 3], dtype=int)
        meth_data["state_readable"] = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.INTERMEDIATE,
                MethylationStates.HIGH,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        emissions_df = pd.DataFrame({"beta": meth_data["beta"]})

        assigner = mock.Mock()
        assigner.relabel_by_mean_emission.return_value = np.array(
            [
                MethylationStates.LOW,
                MethylationStates.INTERMEDIATE,
                MethylationStates.HIGH,
                MethylationStates.HIGH,
            ],
            dtype=object,
        )
        analyzer = mock.Mock()
        analyzer.assigner = assigner
        analyzer.state_cutoffs = None

        segmenter = MethylSegmenter(
            analyzer=analyzer,
            hmm_model=fake_hmm,
            out_dir=".",
        )

        def _assign_states_side_effect(sample_info, chrom):
            segmenter.meth_data = meth_data.copy()
            segmenter.emissions_df = emissions_df.copy()
            return segmenter.meth_data, segmenter.emissions_df

        segmenter.assign_states = mock.Mock(side_effect=_assign_states_side_effect)

        result_meth_data, _ = segmenter.segment_sample(sample_info=sample_info, chrom=None)

        np.testing.assert_array_equal(
            fake_hmm.fit_inputs[0],
            np.array([0, 1, 2, 2], dtype=int),
        )
        np.testing.assert_array_equal(
            fake_hmm.predict_inputs[0],
            np.array([0, 1, 2, 2], dtype=int),
        )
        self.assertEqual(fake_hmm.fit_lengths[0], [2, 2])
        self.assertEqual(fake_hmm.predict_lengths[0], [2, 2])
        np.testing.assert_array_equal(
            result_meth_data["hmm_state"].to_numpy(dtype=int),
            np.array([0, 1, 2, 2], dtype=int),
        )


class RunOnAllChromsPathwayTests(unittest.TestCase):
    def make_sample_info(self) -> SampleInfo:
        meth_df = pd.DataFrame(
            {
                "CpG_chrm": ["chr1", "chr1", "chr2", "chr2"],
                "CpG_beg": [10, 20, 100, 120],
                "CpG_end": [11, 21, 101, 121],
                "beta": [0.1, 0.2, 0.8, 0.9],
            }
        )
        return SampleInfo(sample_id="joint_sample", meth_data=meth_df)

    def test_run_on_all_chroms_joint_hmm_writes_per_chrom_state_beds(self):
        sample_info = self.make_sample_info()
        joint_regions = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr2",
                    "start": 100,
                    "end": 121,
                    "avg_beta": 0.85,
                    "probe_count": 2,
                    "state": MethylationStates.HIGH,
                },
                {
                    "CpG_chrm": "chr1",
                    "start": 10,
                    "end": 21,
                    "avg_beta": 0.15,
                    "probe_count": 2,
                    "state": MethylationStates.LOW,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pathway = MethylSegPathway(
                train_sample_info=sample_info,
                hmm_type="sticky",
                hmm_params={},
                out_dir=tmpdir,
            )
            pathway.segmentor.segment_sample = mock.Mock(
                return_value=(sample_info.meth_data.copy(), object())
            )
            pathway.segmentor.create_regions = mock.Mock(return_value=joint_regions.copy())

            result = pathway.run_on_all_chroms(
                sample_info=sample_info,
                chroms=["chr2", "chr1"],
                min_probes=5,
            )

            called_sample_info = pathway.segmentor.segment_sample.call_args.kwargs[
                "sample_info"
            ]
            self.assertEqual(
                called_sample_info.meth_data["CpG_chrm"].tolist(),
                ["chr2", "chr2", "chr1", "chr1"],
            )
            pathway.segmentor.segment_sample.assert_called_once_with(
                sample_info=mock.ANY,
                chrom=None,
                force_resegment=False,
            )
            pathway.segmentor.create_regions.assert_called_once_with(
                state_col="hmm_state_readable",
                region_min_probes=5,
            )
            pd.testing.assert_frame_equal(result.reset_index(drop=True), joint_regions)

            out_dir = Path(tmpdir)
            self.assertEqual(
                (out_dir / "segments_chr2_joint_sample_HIGH.bed").read_text().strip(),
                "chr2\t100\t121\tHIGH",
            )
            self.assertEqual(
                (out_dir / "segments_chr1_joint_sample_LOW.bed").read_text().strip(),
                "chr1\t10\t21\tLOW",
            )
            self.assertTrue((out_dir / "segments_chr1_joint_sample_PMR.bed").exists())
            self.assertEqual(
                (out_dir / "segments_chr1_joint_sample_PMR.bed").read_text(),
                "",
            )

    def test_run_on_all_chroms_ct_falls_back_to_per_chrom_generation(self):
        sample_info = self.make_sample_info()
        region_by_chrom = {
            "chr2": pd.DataFrame(
                [
                    {
                        "CpG_chrm": "chr2",
                        "start": 100,
                        "end": 121,
                        "avg_beta": 0.85,
                        "probe_count": 2,
                        "state": MethylationStates.HIGH,
                    }
                ]
            ),
            "chr1": pd.DataFrame(
                [
                    {
                        "CpG_chrm": "chr1",
                        "start": 10,
                        "end": 21,
                        "avg_beta": 0.15,
                        "probe_count": 2,
                        "state": MethylationStates.LOW,
                    }
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            pathway = MethylSegPathway(
                train_sample_info=sample_info,
                hmm_type="ct",
                hmm_params={
                    "n_emissions": 4,
                    "holding_time_guess": 100.0,
                },
                out_dir=tmpdir,
            )
            pathway.generate_regions = mock.Mock(
                side_effect=lambda sample_info, chrom, min_probes, force_resegment: region_by_chrom[
                    chrom
                ].copy()
            )

            result = pathway.run_on_all_chroms(
                sample_info=sample_info,
                chroms=["chr2", "chr1"],
                min_probes=4,
                force_resegment=True,
            )

            self.assertEqual(
                [call.kwargs["chrom"] for call in pathway.generate_regions.call_args_list],
                ["chr2", "chr1"],
            )
            for call in pathway.generate_regions.call_args_list:
                self.assertEqual(call.kwargs["sample_info"].sample_id, sample_info.sample_id)
                self.assertTrue(call.kwargs["force_resegment"])
                self.assertEqual(call.kwargs["min_probes"], 4)

            expected = pd.concat(
                [region_by_chrom["chr2"], region_by_chrom["chr1"]],
                ignore_index=True,
            )
            pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


class PreprocessingBehaviorTests(unittest.TestCase):
    def test_preprocess_emission_features_fit_uses_median_imputation_and_log1p(self):
        assigner = MethylStateAssigner(random_state=17)
        emission_df = pd.DataFrame(
            {
                "feat_a": [1.0, np.nan, 5.0],
                "feat_b": [2.0, 4.0, 6.0],
                "25kb_n_cpg": [0.0, 9.0, 99.0],
            }
        )
        feature_cols = ["feat_a", "feat_b", "25kb_n_cpg"]

        scaled_values, imputer, scaler = assigner._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=feature_cols,
            fit=True,
        )

        expected = emission_df.copy()
        expected["25kb_n_cpg"] = np.log1p(expected["25kb_n_cpg"])
        expected.loc[1, "feat_a"] = 3.0
        expected_values = expected.to_numpy(dtype=float)
        expected_mean = expected_values.mean(axis=0)
        expected_std = expected_values.std(axis=0, ddof=0)
        expected_scaled = (expected_values - expected_mean) / expected_std

        np.testing.assert_allclose(scaled_values, expected_scaled)
        np.testing.assert_allclose(
            imputer.statistics_,
            np.array([3.0, 4.0, np.log1p(9.0)]),
        )
        np.testing.assert_allclose(scaler.mean_, expected_mean)

    def test_preprocess_emission_features_fit_skips_imputer_without_nans(self):
        assigner = MethylStateAssigner(random_state=17)
        emission_df = pd.DataFrame(
            {
                "feat_a": [1.0, 3.0, 5.0],
                "25kb_n_cpg": [0.0, 9.0, 99.0],
            }
        )

        scaled_values, imputer, scaler = assigner._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=["feat_a", "25kb_n_cpg"],
            fit=True,
        )

        expected = emission_df.copy()
        expected["25kb_n_cpg"] = np.log1p(expected["25kb_n_cpg"])
        expected_values = expected.to_numpy(dtype=float)
        expected_mean = expected_values.mean(axis=0)
        expected_std = expected_values.std(axis=0, ddof=0)
        expected_scaled = (expected_values - expected_mean) / expected_std

        self.assertIsNone(imputer)
        np.testing.assert_allclose(scaled_values, expected_scaled)
        np.testing.assert_allclose(scaler.mean_, expected_mean)

    def test_preprocess_emission_features_apply_reuses_model_imputer_and_scaler(self):
        imputer = mock.Mock()
        imputed_values = np.array([[3.0, np.log1p(9.0)]])
        imputer.transform.return_value = imputed_values
        scaler = mock.Mock()
        scaled_values = np.array([[0.25, -0.75]])
        scaler.transform.return_value = scaled_values
        assigner = MethylStateAssigner(random_state=17)
        assigner.model = KMeansMethylationModel(
            kmeans=mock.Mock(),
            scaler=scaler,
            imputer=imputer,
            pca=None,
            feature_cols=["feat_a", "25kb_n_cpg"],
            n_states=assigner.n_states,
        )

        emission_df = pd.DataFrame(
            {
                "feat_a": [np.nan],
                "25kb_n_cpg": [9.0],
            }
        )

        actual = assigner._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=assigner.model.feature_cols,
            fit=False,
        )

        np.testing.assert_allclose(
            imputer.transform.call_args.args[0],
            np.array([[np.nan, np.log1p(9.0)]]),
            equal_nan=True,
        )
        np.testing.assert_allclose(scaler.transform.call_args.args[0], imputed_values)
        np.testing.assert_allclose(actual, scaled_values)

    def test_preprocess_emission_features_apply_skips_imputer_without_nans(self):
        imputer = mock.Mock()
        scaler = mock.Mock()
        scaled_values = np.array([[0.25, -0.75]])
        scaler.transform.return_value = scaled_values
        assigner = MethylStateAssigner(random_state=17)
        assigner.model = KMeansMethylationModel(
            kmeans=mock.Mock(),
            scaler=scaler,
            imputer=imputer,
            pca=None,
            feature_cols=["feat_a", "25kb_n_cpg"],
            n_states=assigner.n_states,
        )

        emission_df = pd.DataFrame(
            {
                "feat_a": [3.0],
                "25kb_n_cpg": [9.0],
            }
        )

        actual = assigner._preprocess_emission_features(
            emission_df=emission_df,
            feature_cols=assigner.model.feature_cols,
            fit=False,
        )

        imputer.transform.assert_not_called()
        np.testing.assert_allclose(
            scaler.transform.call_args.args[0],
            np.array([[3.0, np.log1p(9.0)]]),
        )
        np.testing.assert_allclose(actual, scaled_values)

    def test_preprocess_emission_features_apply_without_imputer_rejects_nans(self):
        scaler = mock.Mock()
        assigner = MethylStateAssigner(random_state=17)
        assigner.model = KMeansMethylationModel(
            kmeans=mock.Mock(),
            scaler=scaler,
            imputer=None,
            pca=None,
            feature_cols=["feat_a", "25kb_n_cpg"],
            n_states=assigner.n_states,
        )

        emission_df = pd.DataFrame(
            {
                "feat_a": [np.nan],
                "25kb_n_cpg": [9.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "fit without an imputer"):
            assigner._preprocess_emission_features(
                emission_df=emission_df,
                feature_cols=assigner.model.feature_cols,
                fit=False,
            )


if __name__ == "__main__":
    unittest.main()
