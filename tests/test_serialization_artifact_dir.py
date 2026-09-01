from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from methylseg.helper_classes import (
    HMMObservationMode,
    KMeansMethylationModel,
    MethylStateAssignmentMethod,
    MethylationStates,
    SampleInfo,
)
from methylseg.methylseg_pathway import MethylSegPathway


class SerializationArtifactDirTests(unittest.TestCase):
    def test_to_yaml_can_write_artifacts_to_dedicated_sibling_directory(self):
        rng = np.random.default_rng(0)
        raw_features = rng.normal(size=(12, 3))
        scaler = StandardScaler().fit(raw_features)
        scaled = scaler.transform(raw_features)
        pca = PCA(n_components=2, random_state=0).fit(scaled)
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=0).fit(
            pca.transform(scaled)
        )

        feature_cols = ["beta", "40kb_avg_meth", "40kb_std"]
        model = KMeansMethylationModel(
            kmeans=kmeans,
            scaler=scaler,
            imputer=None,
            pca=pca,
            feature_cols=feature_cols,
            n_states=4,
            cluster_space="pca",
            n_pca=2,
        )

        train_meth = pd.DataFrame(
            [
                {"CpG_chrm": "chr1", "CpG_beg": 10, "CpG_end": 11, "beta": 0.1},
                {"CpG_chrm": "chr1", "CpG_beg": 20, "CpG_end": 21, "beta": 0.3},
                {"CpG_chrm": "chr1", "CpG_beg": 30, "CpG_end": 31, "beta": 0.8},
            ]
        )
        train_emission_df = pd.DataFrame(
            raw_features[:3],
            columns=feature_cols,
        )
        train_joint = train_meth.assign(
            kmeans_label=["PMD", "LOW", "HIGH"],
            rule_based_label=["PMD", "LOW", "HIGH"],
        )

        pathway = MethylSegPathway.__new__(MethylSegPathway)
        pathway.data_path = None
        pathway.meth_ref_path = None
        pathway.samples_info_path = None
        pathway.out_dir = "."
        pathway.train_sample_name = "train-sample"
        pathway.train_sample_file = None
        pathway.train_chroms = None
        pathway.max_cpg_per_chrom = 50_000
        pathway.random_state = 42
        pathway.cluster_space = "pca"
        pathway.n_pca = 2
        pathway.min_region_length = 5_000
        pathway.min_region_cpgs = 6
        pathway.merge_gap_bp = 100_000
        pathway.hmm_params = {"stay_prob": 0.99995}
        pathway.hmm_type = "sticky"
        pathway.hmm_observation_mode = HMMObservationMode.DISCRETE_STATES
        pathway.train_sample_info = SampleInfo("train-sample", train_meth.copy())
        pathway.assigner = SimpleNamespace(
            window_specs=[(40_000, "40kb")],
            n_states=4,
            int_low_cutoff=0.2,
            int_high_cutoff=0.7,
            high_cutoff=0.7,
            model=model,
            train_meth=train_meth.copy(),
            train_emission_df=train_emission_df.copy(),
            training_summary_df=pd.DataFrame(
                [{"state": "PMD", "count": 1, "fraction": 0.33}]
            ),
            train_labels=np.array(
                [
                    MethylationStates.PMD,
                    MethylationStates.LOW,
                    MethylationStates.HIGH,
                ],
                dtype=object,
            ),
            train_pca_scores=pca.transform(scaled[:3]),
        )
        pathway.analyzer = SimpleNamespace(
            state_cutoffs={
                "beta_low_max": 0.2,
                "beta_high_min": 0.7,
                "pmd_cutoffs": {
                    "40kb": {
                        "int_min": 0.5,
                        "std_max": 0.3,
                        "high_max": 0.2,
                        "low_max": 0.2,
                    }
                },
            },
            cutoffs_set_manually=False,
            train_joint=train_joint.copy(),
        )
        pathway.segmentor = SimpleNamespace(
            state_assignment_method=MethylStateAssignmentMethod.KMEANS,
            hmm_observation_mode=HMMObservationMode.DISCRETE_STATES,
            regions_df=pd.DataFrame(
                [
                    {
                        "CpG_chrm": "chr1",
                        "start": 10,
                        "end": 31,
                        "avg_beta": 0.4,
                        "probe_count": 3,
                        "state": MethylationStates.PMD,
                    }
                ]
            ),
        )

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            yaml_path = tmpdir / "reference_files" / "toy_model.yaml"
            artifact_dir = "toy_model_artifacts"

            pathway.to_yaml(
                yaml_path=yaml_path,
                artifact_dir=artifact_dir,
            )

            saved_cfg = yaml.safe_load(yaml_path.read_text())
            self.assertEqual(
                saved_cfg["models"]["scaler"],
                "toy_model_artifacts/models/scaler.joblib",
            )
            self.assertEqual(
                saved_cfg["training_artifacts"]["train_emission_df"],
                "toy_model_artifacts/training_artifacts/train_emission_df.feather",
            )
            self.assertEqual(
                saved_cfg["train_sample_info"]["meth_data_path"],
                "toy_model_artifacts/train_sample_meth.feather",
            )

            restored = MethylSegPathway.from_yaml(yaml_path)
            self.assertEqual(
                restored.assigner.model.scaler.n_features_in_,
                len(feature_cols),
            )
            self.assertEqual(
                restored.assigner.model.feature_cols,
                feature_cols,
            )


if __name__ == "__main__":
    unittest.main()
