from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from analysis.shared_utils.methyl_seg.methyl_seg import (
    MethylSegPathway,
    MethylationStates,
    SampleInfo,
)


def _pathway_stub(tmp_path: Path) -> MethylSegPathway:
    pathway = MethylSegPathway.__new__(MethylSegPathway)
    pathway.min_region_length = 0
    pathway.min_region_cpgs = 1
    pathway.merge_gap_bp = 0
    pathway.out_dir = str(tmp_path)
    pathway.segmentor = SimpleNamespace(regions_df=pd.DataFrame())
    return pathway


def test_get_clean_regions_preserves_raw_regions_when_merge_disabled(tmp_path):
    pathway = _pathway_stub(tmp_path)
    regions_df = pd.DataFrame(
        [
            {
                "CpG_chrm": "chr1",
                "start": 10,
                "end": 20,
                "avg_beta": 0.2,
                "probe_count": 2,
                "state": MethylationStates.PMR,
            },
            {
                "CpG_chrm": "chr1",
                "start": 25,
                "end": 35,
                "avg_beta": 0.8,
                "probe_count": 4,
                "state": MethylationStates.PMR,
            },
            {
                "CpG_chrm": "chr1",
                "start": 40,
                "end": 50,
                "avg_beta": 0.5,
                "probe_count": 3,
                "state": MethylationStates.HIGH,
            },
        ]
    )

    clean_df = pathway.get_clean_regions(
        regions_df=regions_df,
        state="PMR",
        merge_gap_bp=0,
        min_region_length=0,
        min_cpgs=1,
    )

    assert clean_df["start"].tolist() == [10, 25]
    assert clean_df["end"].tolist() == [20, 35]
    assert clean_df["probe_count"].tolist() == [2, 4]
    assert clean_df["length"].tolist() == [10, 10]


def test_get_clean_regions_merges_and_recomputes_weighted_metadata(tmp_path):
    pathway = _pathway_stub(tmp_path)
    regions_df = pd.DataFrame(
        [
            {
                "CpG_chrm": "chr1",
                "start": 10,
                "end": 20,
                "avg_beta": 0.2,
                "probe_count": 2,
                "state": MethylationStates.PMR,
            },
            {
                "CpG_chrm": "chr1",
                "start": 25,
                "end": 45,
                "avg_beta": 0.8,
                "probe_count": 4,
                "state": MethylationStates.PMR,
            },
        ]
    )

    clean_df = pathway.get_clean_regions(
        regions_df=regions_df,
        state=MethylationStates.PMR,
        merge_gap_bp=10,
        min_region_length=0,
        min_cpgs=1,
    )

    assert len(clean_df) == 1
    assert clean_df.iloc[0]["start"] == 10
    assert clean_df.iloc[0]["end"] == 45
    assert clean_df.iloc[0]["probe_count"] == 6
    assert clean_df.iloc[0]["length"] == 35
    assert clean_df.iloc[0]["avg_beta"] == 0.6


def test_get_clean_regions_applies_post_merge_filters(tmp_path):
    pathway = _pathway_stub(tmp_path)
    pathway.min_region_length = 30
    pathway.min_region_cpgs = 5
    regions_df = pd.DataFrame(
        [
            {
                "CpG_chrm": "chr1",
                "start": 10,
                "end": 20,
                "avg_beta": 0.2,
                "probe_count": 2,
                "state": MethylationStates.PMR,
            },
            {
                "CpG_chrm": "chr1",
                "start": 25,
                "end": 45,
                "avg_beta": 0.8,
                "probe_count": 4,
                "state": MethylationStates.PMR,
            },
        ]
    )

    clean_df = pathway.get_clean_regions(
        regions_df=regions_df,
        state="PMR",
        merge_gap_bp=10,
    )

    assert len(clean_df) == 1
    assert clean_df.iloc[0]["probe_count"] == 6
    assert clean_df.iloc[0]["length"] == 35


def test_generate_regions_filters_by_length_and_writes_raw_state_beds(tmp_path):
    class DummySegmentor:
        def __init__(self, regions_df):
            self._regions_df = regions_df
            self.regions_df = pd.DataFrame()
            self.regions_to_bed_calls = []

        def segment_sample(self, sample_info, chrom, force_resegment=False):
            return sample_info.meth_data, None

        def create_regions(self, state_col="hmm_state_readable", region_min_probes=1):
            self.regions_df = self._regions_df.copy()
            return self.regions_df.copy()

        def regions_to_bed(self, bed_path: str, separate_beds_by_state: bool = False):
            self.regions_to_bed_calls.append((bed_path, separate_beds_by_state))

    raw_regions = pd.DataFrame(
        [
            {
                "CpG_chrm": "chr1",
                "start": 10,
                "end": 20,
                "avg_beta": 0.2,
                "probe_count": 2,
                "state": MethylationStates.PMR,
            },
            {
                "CpG_chrm": "chr1",
                "start": 25,
                "end": 45,
                "avg_beta": 0.8,
                "probe_count": 4,
                "state": MethylationStates.PMR,
            },
            {
                "CpG_chrm": "chr1",
                "start": 60,
                "end": 75,
                "avg_beta": 0.7,
                "probe_count": 3,
                "state": MethylationStates.HIGH,
            },
        ]
    )

    pathway = _pathway_stub(tmp_path)
    pathway.min_region_length = 20
    pathway.segmentor = DummySegmentor(raw_regions)

    sample_info = SampleInfo(
        sample_id="sample1",
        meth_data=pd.DataFrame(
            [{"CpG_chrm": "chr1", "CpG_beg": 10, "CpG_end": 11, "beta": 0.5}]
        ),
    )

    result_df = pathway.generate_regions(sample_info=sample_info, chrom="chr1")

    assert result_df["start"].tolist() == [25]
    assert result_df["end"].tolist() == [45]
    assert result_df["length"].tolist() == [20]
    assert pathway.segmentor.regions_to_bed_calls == [
        (f"{tmp_path}/segments_chr1_sample1.bed", True)
    ]
