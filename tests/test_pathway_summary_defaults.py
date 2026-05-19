from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from methylseg.helper_classes import HMMObservationMode, HMMType, MethylationStates, SampleInfo
from methylseg.methyl_segmentor import MethylSegmentor
from methylseg.methylseg_pathway import MethylSegPathway


def _sample_info(sample_id: str = "train_sample") -> SampleInfo:
    return SampleInfo(
        sample_id=sample_id,
        meth_data=pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "CpG_beg": 10, "CpG_end": 11, "beta": 0.20},
                {"CpG_chrm": "chr1", "CpG_beg": 20, "CpG_end": 21, "beta": 0.25},
                {"CpG_chrm": "chr2", "CpG_beg": 30, "CpG_end": 31, "beta": 0.80},
                {"CpG_chrm": "chr2", "CpG_beg": 40, "CpG_end": 41, "beta": 0.85},
            ]
        ),
    )


class DummySegmentor:
    def __init__(self, regions_df: pd.DataFrame):
        self._regions_df = regions_df
        self.regions_df = pd.DataFrame()
        self.default_sample_info = None
        self.segment_sample_calls = []
        self.regions_to_bed_calls = []

    def segment_sample(self, sample_info=None, chrom=None, force_resegment=False):
        self.segment_sample_calls.append((sample_info, chrom, force_resegment))
        return sample_info.meth_data.copy(), None

    def create_regions(self, state_col="hmm_state_readable", region_min_probes=1):
        self.regions_df = self._regions_df.copy()
        return self.regions_df.copy()

    def regions_to_bed(self, bed_path: str, separate_beds_by_state: bool = False):
        self.regions_to_bed_calls.append((bed_path, separate_beds_by_state))


def _pathway_stub(tmp_path: Path, train_sample_info: SampleInfo) -> MethylSegPathway:
    pathway = MethylSegPathway.__new__(MethylSegPathway)
    pathway.train_sample_info = train_sample_info
    pathway.train_chroms = None
    pathway.out_dir = str(tmp_path)
    pathway.min_region_length = 0
    pathway.min_region_cpgs = 1
    pathway.merge_gap_bp = 0
    pathway.hmm_type = HMMType.CT
    pathway.segmentor = SimpleNamespace(
        regions_df=pd.DataFrame(),
        default_sample_info=train_sample_info,
    )
    return pathway


class PathwaySummaryDefaultTests(unittest.TestCase):
    def test_generate_regions_uses_train_sample_by_default(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_sample = _sample_info()
            raw_regions = pd.DataFrame(
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

            pathway = _pathway_stub(tmp_path, train_sample)
            pathway.segmentor = DummySegmentor(raw_regions)
            pathway.segmentor.default_sample_info = train_sample

            result_df = pathway.generate_regions(chrom="chr1")

            self.assertEqual(
                pathway.segmentor.segment_sample_calls[0][0].sample_id,
                train_sample.sample_id,
            )
            self.assertEqual(result_df["start"].tolist(), [10])
            self.assertEqual(
                pathway.segmentor.regions_to_bed_calls,
                [(f"{tmp_path}/segments_chr1_{train_sample.sample_id}.bed", True)],
            )

    def test_run_on_all_chroms_writes_raw_and_optional_cleaned_summaries(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_sample = _sample_info()
            pathway = _pathway_stub(tmp_path, train_sample)

            chrom_regions = {
                "chr1": pd.DataFrame(
                    [
                        {
                            "CpG_chrm": "chr1",
                            "start": 10,
                            "end": 20,
                            "avg_beta": 0.2,
                            "probe_count": 2,
                            "state": MethylationStates.PMD,
                        },
                        {
                            "CpG_chrm": "chr1",
                            "start": 30,
                            "end": 40,
                            "avg_beta": 0.8,
                            "probe_count": 2,
                            "state": MethylationStates.HIGH,
                        },
                    ]
                ),
                "chr2": pd.DataFrame(
                    [
                        {
                            "CpG_chrm": "chr2",
                            "start": 50,
                            "end": 60,
                            "avg_beta": 0.1,
                            "probe_count": 2,
                            "state": MethylationStates.LOW,
                        }
                    ]
                ),
            }

            def fake_generate_regions(
                self,
                sample_info=None,
                chrom="chr1",
                min_probes=3,
                sample_name=None,
                sample_file=None,
                force_resegment=False,
            ):
                self._last_generate_sample = sample_info
                return chrom_regions[chrom].copy()

            pathway.generate_regions = MethodType(fake_generate_regions, pathway)

            raw_paths = pathway.run_on_all_chroms(
                chroms=["chr1", "chr2"],
                clean_regions=False,
            )
            expected_raw = [
                str(tmp_path / "summary_files" / "segments_raw_LOW.bed"),
                str(tmp_path / "summary_files" / "segments_raw_PMD.bed"),
                str(tmp_path / "summary_files" / "segments_raw_INTERMEDIATE.bed"),
                str(tmp_path / "summary_files" / "segments_raw_HIGH.bed"),
            ]
            self.assertEqual(raw_paths, expected_raw)
            for path in expected_raw:
                self.assertTrue(Path(path).exists(), path)
            self.assertFalse(
                (tmp_path / "summary_files" / "segments_cleaned_PMD.bed").exists()
            )
            self.assertEqual(pathway._last_generate_sample.sample_id, train_sample.sample_id)

            cleaned_paths = pathway.run_on_all_chroms(
                chroms=["chr1", "chr2"],
                clean_regions=True,
            )
            self.assertEqual(len(cleaned_paths), 8)
            self.assertIn(
                str(tmp_path / "summary_files" / "segments_cleaned_PMD.bed"),
                cleaned_paths,
            )
            self.assertTrue(
                (tmp_path / "summary_files" / "segments_cleaned_PMD.bed").exists()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "clean_regions"
                    / f"metadata_cleaned_chr1_{train_sample.sample_id}_PMD.tsv"
                ).exists()
            )

    def test_run_pathway_honors_explicit_sample_override(self):
        pathway = MethylSegPathway.__new__(MethylSegPathway)
        train_sample = _sample_info("train")
        eval_sample = _sample_info("eval")
        pathway.train_sample_info = train_sample
        pathway.fit_pathway = MethodType(
            lambda self, force_optimize_rules=False: None,
            pathway,
        )

        captured = {}

        def fake_run_on_all_chroms(
            self,
            sample_info=None,
            chroms=None,
            min_probes=3,
            force_resegment=False,
            clean_regions=True,
        ):
            captured["sample_id"] = sample_info.sample_id
            captured["clean_regions"] = clean_regions
            return ["ok"]

        pathway.run_on_all_chroms = MethodType(fake_run_on_all_chroms, pathway)

        result = pathway.run_pathway(sample_info=eval_sample, clean_regions=False)

        self.assertEqual(result, ["ok"])
        self.assertEqual(captured["sample_id"], "eval")
        self.assertFalse(captured["clean_regions"])


class SegmentorDefaultSampleTests(unittest.TestCase):
    def test_segment_sample_uses_default_sample_info(self):
        segmentor = MethylSegmentor.__new__(MethylSegmentor)
        default_sample = _sample_info()
        segmentor.default_sample_info = default_sample
        segmentor.segment_results = {}
        segmentor.hmm_observation_mode = HMMObservationMode.DISCRETE_STATES
        segmentor.hmm_model = SimpleNamespace(hmm_model="dummy")

        def fake_discrete(self, sample_info, chrom=None):
            self.meth_data = sample_info.meth_data.copy()
            self.emissions_df = pd.DataFrame({"feature": np.arange(len(self.meth_data))})
            return np.zeros(len(self.meth_data), dtype=int), np.zeros(
                len(self.meth_data), dtype=int
            )

        def fake_create_regions(self, state_col="hmm_state_readable", region_min_probes=1):
            return pd.DataFrame(
                [
                    {
                        "CpG_chrm": "chr1",
                        "start": 10,
                        "end": 20,
                        "avg_beta": 0.2,
                        "probe_count": 2,
                        "state": 0,
                    }
                ]
            )

        segmentor._segment_sample_discrete_states = MethodType(fake_discrete, segmentor)
        segmentor.create_regions = MethodType(fake_create_regions, segmentor)

        meth_data, hmm_model = segmentor.segment_sample(chrom="chr1")

        self.assertEqual(meth_data["hmm_state"].tolist(), [0, 0, 0, 0])
        self.assertEqual(hmm_model, "dummy")
        self.assertIn(default_sample.sample_id, segmentor.segment_results)


if __name__ == "__main__":
    unittest.main()
