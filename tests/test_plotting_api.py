from types import MethodType, SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.modules.setdefault("gdown", SimpleNamespace())

from analysis.shared_utils.methylseg.methylseg import (  # noqa: E402
    HMMObservationMode,
    MethylSegPathway,
    MethylSegmentor,
    MethylStateAnalyzer,
    MethylStateAssigner,
    MethylationStates,
    SampleInfo,
)
from analysis.shared_utils.methylseg.methylseg.utils import (  # noqa: E402
    plot_interactive_beta_scatter,
)


def _sample_info(sample_id: str = "sample1") -> SampleInfo:
    return SampleInfo(
        sample_id=sample_id,
        meth_data=pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "CpG_beg": 10, "CpG_end": 11, "beta": 0.2},
                {"CpG_chrm": "chr1", "CpG_beg": 20, "CpG_end": 21, "beta": 0.3},
                {"CpG_chrm": "chr2", "CpG_beg": 30, "CpG_end": 31, "beta": 0.8},
            ]
        ),
    )


class PathwayPlottingTests(unittest.TestCase):
    def test_plot_labels_defaults_to_train_sample_for_hmm_and_clean_overlay(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        train_sample = _sample_info("train")
        pathway.train_sample_info = train_sample

        raw_regions = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 10,
                    "end": 30,
                    "avg_beta": 0.25,
                    "probe_count": 2,
                    "state": MethylationStates.PMD,
                }
            ]
        )
        clean_regions = raw_regions.copy()

        pathway.generate_regions = MethodType(
            lambda self, sample_info=None, chrom="chr1", min_probes=3, force_resegment=False, **kwargs: raw_regions.copy(),
            pathway,
        )
        pathway.get_clean_regions = MethodType(
            lambda self, regions_df=None, state=MethylationStates.PMD, **kwargs: clean_regions.copy(),
            pathway,
        )

        captured = {}

        def fake_plot_labels(self, **kwargs):
            captured.update(kwargs)
            return "hmm-figure"

        pathway.segmentor = SimpleNamespace(plot_labels=MethodType(fake_plot_labels, SimpleNamespace()))
        pathway.analyzer = SimpleNamespace()

        result = pathway.plot_labels(
            label_source="hmm",
            chrom="chr1",
            clean_regions=True,
            overlay_state="PMD",
            show_plot=False,
        )

        self.assertEqual(result, "hmm-figure")
        self.assertEqual(captured["sample_info"].sample_id, "train")
        self.assertTrue(captured["overlay_regions_df"].equals(clean_regions))
        self.assertEqual(captured["overlay_style"], "state")
        self.assertEqual(captured["region_start"], None)

    def test_plot_labels_preserves_state_overlay_when_region_is_selected(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.train_sample_info = _sample_info("train")

        clean_regions = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 95,
                    "end": 205,
                    "avg_beta": 0.25,
                    "probe_count": 2,
                    "state": MethylationStates.PMD,
                    "length": 110,
                }
            ]
        )
        pathway.get_clean_regions = MethodType(
            lambda self, regions_df=None, state=MethylationStates.PMD, **kwargs: clean_regions.copy(),
            pathway,
        )
        pathway.generate_regions = MethodType(
            lambda self, sample_info=None, chrom="chr1", min_probes=3, force_resegment=False, **kwargs: clean_regions.copy(),
            pathway,
        )
        pathway.segmentor = SimpleNamespace()

        captured = {}

        def fake_plot_labels(self, **kwargs):
            captured.update(kwargs)
            return "kmeans-figure"

        pathway.analyzer = SimpleNamespace(plot_labels=MethodType(fake_plot_labels, SimpleNamespace()))

        result = pathway.plot_labels(
            label_source="kmeans",
            chrom="chr1",
            clean_regions=True,
            region_start=100,
            region_end=200,
            region_chrom="chr1",
            show_plot=False,
        )

        self.assertEqual(result, "kmeans-figure")
        self.assertEqual(captured["overlay_style"], "state")
        self.assertEqual(captured["overlay_regions_df"].iloc[0]["start"], 95)
        self.assertEqual(captured["overlay_regions_df"].iloc[0]["end"], 205)
        self.assertEqual(captured["region_start"], 100)
        self.assertEqual(captured["region_end"], 200)
        self.assertEqual(captured["region_chrom"], "chr1")

    def test_plot_labels_routes_clean_overlay_reuse_through_get_clean_regions(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.train_sample_info = _sample_info("train")
        pathway.segmentor = SimpleNamespace()
        clean_regions = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 10,
                    "end": 20,
                    "avg_beta": 0.2,
                    "probe_count": 2,
                    "state": MethylationStates.PMD,
                    "length": 10,
                }
            ]
        )
        captured_get_clean_regions = {}

        def fake_get_clean_regions(self, regions_df=None, state=MethylationStates.PMD, **kwargs):
            captured_get_clean_regions.update(kwargs)
            return clean_regions.copy()

        pathway.get_clean_regions = MethodType(fake_get_clean_regions, pathway)
        pathway.generate_regions = MethodType(
            lambda self, sample_info=None, chrom="chr1", min_probes=3, force_resegment=False, **kwargs: clean_regions.copy(),
            pathway,
        )

        captured_plot = {}

        def fake_plot_labels(self, **kwargs):
            captured_plot.update(kwargs)
            return "kmeans-figure"

        pathway.analyzer = SimpleNamespace(plot_labels=MethodType(fake_plot_labels, SimpleNamespace()))

        result = pathway.plot_labels(
            label_source="kmeans",
            chrom="chr1",
            clean_regions=True,
            show_plot=False,
        )

        self.assertEqual(result, "kmeans-figure")
        self.assertEqual(captured_get_clean_regions["sample_id"], "train")
        self.assertEqual(captured_get_clean_regions["chrom"], "chr1")
        self.assertFalse(captured_get_clean_regions["force_resegment"])
        self.assertTrue(captured_plot["overlay_regions_df"].equals(clean_regions))

    def test_plot_embedding_uses_training_kmeans_by_default(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.train_sample_info = _sample_info("train")

        captured = {}

        def fake_plot_training_embedding(self, **kwargs):
            captured.update(kwargs)
            return "training-embedding"

        pathway.assigner = SimpleNamespace(
            plot_training_embedding=MethodType(
                fake_plot_training_embedding, SimpleNamespace()
            )
        )

        result = pathway.plot_embedding(label_source="kmeans", show_plot=False)

        self.assertEqual(result, "training-embedding")
        self.assertFalse(captured["show_plot"])


class LowLevelPlottingTests(unittest.TestCase):
    def test_segmentor_plot_labels_uses_default_sample_and_highlight_overlay(self):
        segmentor = MethylSegmentor.__new__(MethylSegmentor)
        default_sample = _sample_info("default")
        segmentor.default_sample_info = default_sample
        segmentor.out_dir = "."

        meth_data = default_sample.meth_data.copy()
        meth_data["hmm_state_readable"] = np.array(
            [MethylationStates.PMD, MethylationStates.PMD, MethylationStates.HIGH],
            dtype=object,
        )
        segmentor.segment_sample = MethodType(
            lambda self, sample_info=None, chrom=None, force_resegment=False: (
                meth_data.copy(),
                None,
            ),
            segmentor,
        )

        with patch(
            "analysis.shared_utils.methylseg.methylseg.methyl_segmentor.plot_interactive_beta_scatter"
        ) as mock_plot:
            mock_plot.return_value = "segmentor-figure"
            result = segmentor.plot_labels(
                chrom="chr1",
                region_start=10,
                region_end=25,
                region_chrom="chr1",
                show_plot=False,
            )

        self.assertEqual(result, "segmentor-figure")
        self.assertEqual(
            mock_plot.call_args.kwargs["sample_info"].sample_id, default_sample.sample_id
        )
        self.assertEqual(mock_plot.call_args.kwargs["overlay_style"], "highlight")
        self.assertEqual(mock_plot.call_args.kwargs["region_start"], 10)
        self.assertEqual(mock_plot.call_args.kwargs["region_end"], 25)

    def test_analyzer_plot_labels_supports_rule_based_labels(self):
        analyzer = MethylStateAnalyzer.__new__(MethylStateAnalyzer)
        analyzer.out_dir = "."
        analyzer.assigner = SimpleNamespace(
            apply_kmeans_to_sample=lambda sample_info, chrom=None: (
                sample_info.meth_data.copy(),
                pd.DataFrame({"beta": sample_info.meth_data["beta"]}),
                None,
                None,
                None,
                np.array([MethylationStates.PMD, MethylationStates.HIGH, MethylationStates.HIGH], dtype=object),
            ),
            train_sample_info=None,
        )
        analyzer.define_states_by_rules = MethodType(
            lambda self, sample_info, chrom=None, sample_emissions=None: np.array(
                [MethylationStates.PMD, MethylationStates.INTERMEDIATE, MethylationStates.HIGH],
                dtype=object,
            ),
            analyzer,
        )

        with patch(
            "analysis.shared_utils.methylseg.methylseg.methyl_state_analyzer.plot_interactive_beta_scatter"
        ) as mock_plot:
            mock_plot.return_value = "analyzer-figure"
            result = analyzer.plot_labels(
                sample_info=_sample_info("eval"),
                chrom="chr1",
                label_source="rule_based",
                show_plot=False,
            )

        self.assertEqual(result, "analyzer-figure")
        self.assertEqual(
            mock_plot.call_args.kwargs["label_col"],
            "rule_based_label",
        )

    def test_analyzer_histograms_use_relabeled_kmeans_state_names(self):
        analyzer = MethylStateAnalyzer.__new__(MethylStateAnalyzer)
        analyzer.out_dir = "."
        analyzer.train_joint = None
        analyzer.assigner = SimpleNamespace(
            model=SimpleNamespace(),
            train_meth=pd.DataFrame({"CpG_beg": [10, 20], "beta": [0.2, 0.8]}),
            train_emission_df=pd.DataFrame({"feat1": [1.0, 2.0]}),
            train_labels=np.array([0, 1]),
            int_low_cutoff=0.2,
            int_high_cutoff=0.7,
            window_specs=[(40_000, "40kb"), (450_000, "450kb")],
            apply_kmeans_to_emissions=lambda emission_df: (
                None,
                None,
                np.array([0, 1]),
                np.array([MethylationStates.PMD, MethylationStates.HIGH], dtype=object),
            ),
            get_pca_loadings=lambda: pd.DataFrame({"PC2": [1.0]}, index=["feat1"]),
        )

        hist_labels = []
        title_values = []

        with patch(
            "analysis.shared_utils.methylseg.methylseg.methyl_state_analyzer.relabel_by_mean_emission"
        ) as mock_relabel, patch(
            "pandas.Series.hist"
        ) as mock_hist, patch(
            "matplotlib.pyplot.title"
        ) as mock_title:
            mock_relabel.return_value = np.array(
                [MethylationStates.PMD, MethylationStates.HIGH], dtype=object
            )
            mock_hist.side_effect = lambda *args, **kwargs: hist_labels.append(
                kwargs["label"]
            )
            mock_title.side_effect = lambda value: title_values.append(value)
            analyzer.plot_feature_distributions_by_kmeans_state(show_plots=False)

        self.assertCountEqual(hist_labels, ["PMD", "HIGH"])
        self.assertTrue(any("KMeans State" in value for value in title_values))
        self.assertTrue(mock_relabel.called)

    def test_analyzer_backfills_kmeans_state_display_on_existing_train_joint(self):
        analyzer = MethylStateAnalyzer.__new__(MethylStateAnalyzer)
        analyzer.out_dir = "."
        analyzer.train_joint = pd.DataFrame(
            {
                "CpG_beg": [10, 20],
                "beta": [0.2, 0.8],
                "feat1": [1.0, 2.0],
                "kmeans_label": [0, 1],
            }
        )
        analyzer.assigner = SimpleNamespace(
            model=SimpleNamespace(),
            train_labels=np.array([0, 1]),
            train_emission_df=pd.DataFrame(
                {
                    "beta": [0.2, 0.8],
                    "40kb_int_pct": [0.5, 0.1],
                    "40kb_high_pct": [0.1, 0.9],
                    "40kb_low_pct": [0.7, 0.0],
                    "450kb_int_pct": [0.6, 0.2],
                    "450kb_high_pct": [0.1, 0.8],
                    "450kb_low_pct": [0.6, 0.0],
                }
            ),
            int_low_cutoff=0.2,
            int_high_cutoff=0.7,
            window_specs=[(40_000, "40kb"), (450_000, "450kb")],
            apply_kmeans_to_emissions=lambda emission_df: (
                None,
                None,
                np.array([0, 1]),
                np.array([MethylationStates.PMD, MethylationStates.HIGH], dtype=object),
            ),
        )

        analyzer._build_train_joint()

        self.assertIn("kmeans_state_display", analyzer.train_joint.columns)
        self.assertCountEqual(
            analyzer.train_joint["kmeans_state_display"].tolist(),
            ["PMD", "HIGH"],
        )

    def test_interactive_plot_preserves_state_colors_and_adds_region_guides(self):
        sample_info = _sample_info("eval")
        df_plot = sample_info.meth_data.copy()
        df_plot["hmm_state_readable"] = np.array(
            [MethylationStates.PMD, MethylationStates.PMD, MethylationStates.HIGH],
            dtype=object,
        )
        overlay_regions_df = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 10,
                    "end": 25,
                    "state": MethylationStates.PMD,
                },
                {
                    "CpG_chrm": "chr2",
                    "start": 25,
                    "end": 35,
                    "state": MethylationStates.HIGH,
                },
            ]
        )

        fig = plot_interactive_beta_scatter(
            df_plot=df_plot,
            sample_info=sample_info,
            sample_info_removed=None,
            chrom="chr1",
            out_dir=None,
            label_col="hmm_state_readable",
            show_plot=False,
            overlay_regions_df=overlay_regions_df,
            overlay_style="state",
            region_start=10,
            region_end=25,
            region_chrom="chr1",
        )

        self.assertIn("PMD", [trace.name for trace in fig.data])
        self.assertEqual(len(fig.layout.shapes), 2)
        self.assertEqual(list(fig.layout.xaxis.range), [-990, 1025])

    def test_assigner_plot_embedding_routes_to_region_highlight_path(self):
        assigner = MethylStateAssigner.__new__(MethylStateAssigner)
        meth_data = _sample_info("eval").meth_data.copy()
        emission_df = pd.DataFrame({"beta": [0.2, 0.3, 0.8]})
        labels = np.array([MethylationStates.PMD, MethylationStates.PMD, MethylationStates.HIGH], dtype=object)

        captured = {}

        def fake_region_plot(self, **kwargs):
            captured.update(kwargs)
            return "pca-region-figure"

        assigner.plot_pca_clusters_with_region = MethodType(fake_region_plot, assigner)

        result = assigner.plot_embedding(
            emission_df=emission_df,
            labels=labels,
            meth_data=meth_data,
            method="pca",
            sample_info=_sample_info("eval"),
            chrom="chr1",
            region_start=10,
            region_end=25,
            region_chrom="chr1",
            show_plot=False,
        )

        self.assertEqual(result, "pca-region-figure")
        self.assertEqual(captured["region_start"], 10)
        self.assertEqual(captured["show_plot"], False)


class CleanRegionCachingTests(unittest.TestCase):
    def test_get_clean_regions_writes_and_reuses_chromosome_specific_cache(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 100
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        regions_df = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 10,
                    "end": 20,
                    "avg_beta": 0.2,
                    "probe_count": 2,
                    "state": MethylationStates.PMD,
                }
            ]
        )

        clean_df = pathway.get_clean_regions(
            regions_df=regions_df,
            state=MethylationStates.PMD,
            sample_id="sample1",
            chrom="chr1",
        )
        cache_path = pathway._clean_region_cache_path(
            sample_id="sample1",
            chrom="chr1",
            state=MethylationStates.PMD,
            merge_gap_bp=100,
            min_region_length=0,
            min_cpgs=1,
        )

        self.assertTrue(cache_path.exists())
        self.assertEqual(clean_df.iloc[0]["start"], 10)

        reused_df = pathway.get_clean_regions(
            regions_df=None,
            state=MethylationStates.PMD,
            sample_id="sample1",
            chrom="chr1",
        )

        self.assertTrue(reused_df.equals(clean_df))

    def test_get_clean_regions_force_resegment_rewrites_cache(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 100
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        first_regions = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 10,
                    "end": 20,
                    "avg_beta": 0.2,
                    "probe_count": 2,
                    "state": MethylationStates.PMD,
                }
            ]
        )
        second_regions = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 30,
                    "end": 40,
                    "avg_beta": 0.4,
                    "probe_count": 3,
                    "state": MethylationStates.PMD,
                }
            ]
        )

        pathway.get_clean_regions(
            regions_df=first_regions,
            state=MethylationStates.PMD,
            sample_id="sample1",
            chrom="chr1",
        )
        rewritten_df = pathway.get_clean_regions(
            regions_df=second_regions,
            state=MethylationStates.PMD,
            sample_id="sample1",
            chrom="chr1",
            force_resegment=True,
        )

        self.assertEqual(rewritten_df.iloc[0]["start"], 30)
        self.assertEqual(rewritten_df.iloc[0]["probe_count"], 3)

    def test_get_clean_regions_does_not_overwrite_summary_file(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 100
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        summary_dir = Path(pathway.out_dir) / "summary_files"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "segments_cleaned_PMD.bed"
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("sentinel\n")

        regions_df = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 10,
                    "end": 20,
                    "avg_beta": 0.2,
                    "probe_count": 2,
                    "state": MethylationStates.PMD,
                }
            ]
        )
        pathway.get_clean_regions(
            regions_df=regions_df,
            state=MethylationStates.PMD,
            sample_id="sample1",
            chrom="chr1",
        )

        with open(summary_path, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "sentinel\n")


if __name__ == "__main__":
    unittest.main()
