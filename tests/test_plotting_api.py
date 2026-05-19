from types import MethodType, SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from methylseg.helper_classes import HMMObservationMode, MethylationStates, SampleInfo
from methylseg.methyl_segmentor import MethylSegmentor
from methylseg.methyl_state_analyzer import MethylStateAnalyzer
from methylseg.methyl_state_assigner import MethylStateAssigner
from methylseg.methylseg_pathway import MethylSegPathway
from methylseg.utils import plot_interactive_beta_scatter


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
        pathway.out_dir = tempfile.mkdtemp()

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
        _write_clean_outputs(pathway.out_dir, {MethylationStates.PMD: clean_regions.copy()})

        captured = {}

        def fake_plot_labels(self, **kwargs):
            captured.update(kwargs)
            return "hmm-figure"

        pathway.segmentor = SimpleNamespace(plot_labels=MethodType(fake_plot_labels, SimpleNamespace()))
        pathway.analyzer = SimpleNamespace()

        result = pathway.plot_labels(
            label_source="hmm",
            chrom="chr1",
            use_cleaned_regions=True,
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
        pathway.out_dir = tempfile.mkdtemp()

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
        _write_clean_outputs(pathway.out_dir, {MethylationStates.PMD: clean_regions.copy()})
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
            use_cleaned_regions=True,
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
        pathway.out_dir = tempfile.mkdtemp()
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
        _write_clean_outputs(pathway.out_dir, {MethylationStates.PMD: clean_regions.copy()})
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
            use_cleaned_regions=True,
            show_plot=False,
        )

        self.assertEqual(result, "kmeans-figure")
        self.assertTrue(captured_plot["overlay_regions_df"].equals(clean_regions))

    def test_plot_labels_use_cleaned_regions_requires_prebuilt_outputs(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.train_sample_info = _sample_info("train")
        pathway.out_dir = tempfile.mkdtemp()
        pathway.segmentor = SimpleNamespace()
        pathway.analyzer = SimpleNamespace(plot_labels=MethodType(lambda self, **kwargs: "unused", SimpleNamespace()))

        with self.assertRaisesRegex(
            ValueError,
            "Clean regions must be generated first using get_clean_regions",
        ):
            pathway.plot_labels(
                label_source="kmeans",
                chrom="chr1",
                use_cleaned_regions=True,
                show_plot=False,
            )

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


def _write_clean_outputs(
    out_dir: str,
    data_by_state: dict[MethylationStates, pd.DataFrame],
    chrom: str = "chr1",
    sample_id: str = "train",
) -> dict[MethylationStates, Path]:
    clean_dir = Path(out_dir) / "clean_regions"
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for state in MethylationStates:
        bed_path = clean_dir / f"segments_cleaned_{chrom}_{sample_id}_{state.name}.bed"
        metadata_path = (
            clean_dir / f"metadata_cleaned_{chrom}_{sample_id}_{state.name}.tsv"
        )
        clean_df = data_by_state.get(state)
        if clean_df is None:
            clean_df = pd.DataFrame(
                columns=[
                    "CpG_chrm",
                    "start",
                    "end",
                    "avg_beta",
                    "probe_count",
                    "state",
                    "length",
                    "contains_intermediate",
                    "n_segments",
                    "n_pmd_segments",
                    "n_intermediate_segments",
                ]
            )
        clean_df.to_csv(metadata_path, sep="\t", index=False)
        clean_df.loc[:, ["CpG_chrm", "start", "end", "state"]].to_csv(
            bed_path, sep="\t", header=False, index=False
        )
        output_paths[state] = bed_path
    return output_paths


class CleanRegionCachingTests(unittest.TestCase):
    def test_get_clean_regions_writes_chr_specific_outputs(self):
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

        summary_paths, clean_dir = pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom="chr1",
        )

        pmd_metadata = clean_dir / "metadata_cleaned_chr1_sample1_PMD.tsv"
        pmd_bed = clean_dir / "segments_cleaned_chr1_sample1_PMD.bed"

        self.assertEqual(summary_paths, {})
        self.assertTrue(pmd_metadata.exists())
        self.assertTrue(pmd_bed.exists())
        self.assertEqual(pd.read_csv(pmd_metadata, sep="\t").iloc[0]["start"], 10)

    def test_get_clean_regions_force_resegment_rewrites_outputs(self):
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

        _, clean_dir = pathway.get_clean_regions(
            regions_df=first_regions,
            sample_id="sample1",
            chrom="chr1",
        )
        pathway.get_clean_regions(
            regions_df=second_regions,
            sample_id="sample1",
            chrom="chr1",
            force_resegment=True,
        )
        rewritten_df = pd.read_csv(
            clean_dir / "metadata_cleaned_chr1_sample1_PMD.tsv",
            sep="\t",
        )

        self.assertEqual(rewritten_df.iloc[0]["start"], 30)
        self.assertEqual(rewritten_df.iloc[0]["probe_count"], 3)

    def test_get_clean_regions_writes_metadata_and_summary_outputs(self):
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
        summary_paths, clean_dir = pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom=None,
            generate_summary_files=True,
        )

        self.assertTrue((clean_dir / "metadata_cleaned_chr1_sample1_PMD.tsv").exists())
        self.assertTrue(summary_paths[MethylationStates.PMD].exists())
        self.assertTrue(
            (
                Path(pathway.out_dir) / "summary_files" / "metadata_cleaned_PMD.tsv"
            ).exists()
        )

    def test_get_clean_regions_pmd_expansion_respects_boundaries_and_gap(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 0
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        regions_df = pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "start": 10, "end": 20, "avg_beta": 0.2, "probe_count": 2, "state": MethylationStates.PMD},
                {"CpG_chrm": "chr1", "start": 20, "end": 30, "avg_beta": 0.4, "probe_count": 2, "state": MethylationStates.INTERMEDIATE},
                {"CpG_chrm": "chr1", "start": 30, "end": 40, "avg_beta": 0.3, "probe_count": 2, "state": MethylationStates.PMD},
                {"CpG_chrm": "chr1", "start": 40, "end": 50, "avg_beta": 0.1, "probe_count": 2, "state": MethylationStates.LOW},
                {"CpG_chrm": "chr1", "start": 50, "end": 60, "avg_beta": 0.2, "probe_count": 2, "state": MethylationStates.PMD},
            ]
        )
        pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom="chr1",
            allow_pmd_expansion=True,
            expansion_merge_bp=0,
        )
        pmd_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_PMD.tsv",
            sep="\t",
        )
        self.assertEqual(pmd_df["start"].tolist(), [10, 50])
        self.assertEqual(pmd_df["end"].tolist(), [40, 60])
        self.assertEqual(pmd_df["contains_intermediate"].tolist(), [True, False])
        self.assertTrue((pmd_df["n_pmd_segments"] > 0).all())

    def test_get_clean_regions_isolated_intermediate_does_not_seed_pmd(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 0
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        regions_df = pd.DataFrame(
            [
                {
                    "CpG_chrm": "chr1",
                    "start": 20,
                    "end": 30,
                    "avg_beta": 0.4,
                    "probe_count": 2,
                    "state": MethylationStates.INTERMEDIATE,
                }
            ]
        )
        pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom="chr1",
            allow_pmd_expansion=True,
            expansion_merge_bp=0,
        )
        pmd_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_PMD.tsv",
            sep="\t",
        )
        intermediate_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_INTERMEDIATE.tsv",
            sep="\t",
        )
        self.assertTrue(pmd_df.empty)
        self.assertEqual(intermediate_df["start"].tolist(), [20])
        self.assertEqual(intermediate_df["end"].tolist(), [30])

    def test_get_clean_regions_high_intermediate_high_stays_non_pmd(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 0
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        regions_df = pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "start": 10, "end": 20, "avg_beta": 0.8, "probe_count": 2, "state": MethylationStates.HIGH},
                {"CpG_chrm": "chr1", "start": 20, "end": 30, "avg_beta": 0.4, "probe_count": 2, "state": MethylationStates.INTERMEDIATE},
                {"CpG_chrm": "chr1", "start": 30, "end": 40, "avg_beta": 0.9, "probe_count": 2, "state": MethylationStates.HIGH},
            ]
        )
        pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom="chr1",
            allow_pmd_expansion=True,
            expansion_merge_bp=0,
        )
        pmd_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_PMD.tsv",
            sep="\t",
        )
        intermediate_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_INTERMEDIATE.tsv",
            sep="\t",
        )
        self.assertTrue(pmd_df.empty)
        self.assertEqual(intermediate_df["start"].tolist(), [20])
        self.assertEqual(intermediate_df["end"].tolist(), [30])
        self.assertNotIn("contains_intermediate", intermediate_df.columns)
        self.assertNotIn("n_pmd_segments", intermediate_df.columns)
        self.assertNotIn("n_intermediate_segments", intermediate_df.columns)

    def test_get_clean_regions_no_expansion_keeps_intermediate_split(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 0
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        regions_df = pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "start": 10, "end": 20, "avg_beta": 0.2, "probe_count": 2, "state": MethylationStates.PMD},
                {"CpG_chrm": "chr1", "start": 20, "end": 30, "avg_beta": 0.4, "probe_count": 2, "state": MethylationStates.INTERMEDIATE},
                {"CpG_chrm": "chr1", "start": 30, "end": 40, "avg_beta": 0.3, "probe_count": 2, "state": MethylationStates.PMD},
            ]
        )
        pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom="chr1",
            allow_pmd_expansion=False,
        )
        pmd_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_PMD.tsv",
            sep="\t",
        )
        self.assertEqual(pmd_df["start"].tolist(), [10, 30])

    def test_get_clean_regions_intermediate_pmd_intermediate_merges_when_anchored(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 0
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        regions_df = pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "start": 10, "end": 20, "avg_beta": 0.4, "probe_count": 2, "state": MethylationStates.INTERMEDIATE},
                {"CpG_chrm": "chr1", "start": 20, "end": 30, "avg_beta": 0.2, "probe_count": 2, "state": MethylationStates.PMD},
                {"CpG_chrm": "chr1", "start": 30, "end": 40, "avg_beta": 0.4, "probe_count": 2, "state": MethylationStates.INTERMEDIATE},
            ]
        )
        pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom="chr1",
            allow_pmd_expansion=True,
            expansion_merge_bp=0,
        )
        pmd_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_PMD.tsv",
            sep="\t",
        )
        self.assertEqual(pmd_df["start"].tolist(), [10])
        self.assertEqual(pmd_df["end"].tolist(), [40])
        self.assertEqual(pmd_df["contains_intermediate"].tolist(), [True])
        self.assertEqual(pmd_df["n_pmd_segments"].tolist(), [1])
        self.assertEqual(pmd_df["n_intermediate_segments"].tolist(), [2])

    def test_get_clean_regions_post_expansion_merge_uses_standard_gap(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 15
        pathway.min_region_length = 0
        pathway.min_region_cpgs = 1
        pathway.segmentor = SimpleNamespace(regions_df=None)

        regions_df = pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "start": 10, "end": 20, "avg_beta": 0.4, "probe_count": 2, "state": MethylationStates.INTERMEDIATE},
                {"CpG_chrm": "chr1", "start": 20, "end": 30, "avg_beta": 0.2, "probe_count": 2, "state": MethylationStates.PMD},
                {"CpG_chrm": "chr1", "start": 30, "end": 40, "avg_beta": 0.4, "probe_count": 2, "state": MethylationStates.INTERMEDIATE},
                {"CpG_chrm": "chr1", "start": 42, "end": 45, "avg_beta": 0.1, "probe_count": 2, "state": MethylationStates.LOW},
                {"CpG_chrm": "chr1", "start": 50, "end": 60, "avg_beta": 0.3, "probe_count": 2, "state": MethylationStates.PMD},
            ]
        )
        pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom="chr1",
            allow_pmd_expansion=True,
            expansion_merge_bp=0,
        )
        pmd_df = pd.read_csv(
            Path(pathway.out_dir)
            / "clean_regions"
            / "metadata_cleaned_chr1_sample1_PMD.tsv",
            sep="\t",
        )
        self.assertEqual(pmd_df["start"].tolist(), [10])
        self.assertEqual(pmd_df["end"].tolist(), [60])
        self.assertEqual(pmd_df["contains_intermediate"].tolist(), [True])
        self.assertEqual(pmd_df["n_segments"].tolist(), [4])
        self.assertEqual(pmd_df["n_pmd_segments"].tolist(), [2])
        self.assertEqual(pmd_df["n_intermediate_segments"].tolist(), [2])

    def test_get_clean_regions_skips_summary_files_when_requested(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.out_dir = tempfile.mkdtemp()
        pathway.merge_gap_bp = 0
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
        summary_paths, clean_dir = pathway.get_clean_regions(
            regions_df=regions_df,
            sample_id="sample1",
            chrom=None,
            generate_summary_files=False,
        )
        self.assertEqual(summary_paths, {})
        self.assertTrue((clean_dir / "metadata_cleaned_chr1_sample1_PMD.tsv").exists())
        self.assertFalse(
            (Path(pathway.out_dir) / "summary_files" / "metadata_cleaned_PMD.tsv").exists()
        )


if __name__ == "__main__":
    unittest.main()
