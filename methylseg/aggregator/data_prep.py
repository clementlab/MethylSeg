"""Prepare streaming sample inputs and canonical manifests for aggregation."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..helper_classes import SampleInfo

class DataPrep:
    def __init__(self, sample_infos: Iterable[SampleInfo], output_dir: str | Path = "aggregation"):
        self.sample_infos = sample_infos
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def create_aggregation_manifest(self):
        manifest = []
        for sample in self.sample_infos:
            sample_id = sample.sample_id
            meth_data_df = sample.meth_data
            meth_data_path = self.output_dir / f"{sample_id}_meth_data.csv"
            meth_data_df.to_csv(meth_data_path, index=False)
            resolution = sample.resolution
            manifest.append((sample_id, str(meth_data_path), resolution))
        manifest_df = pd.DataFrame(manifest, columns=["sample_id", "meth_data_path", "resolution"])
        manifest_df.to_csv(self.output_dir / "aggregation_manifest.csv", index=False)
        return manifest_df
