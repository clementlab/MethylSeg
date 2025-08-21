import os
from typing import Union

import joblib
import pandas as pd
from .data_prep import DataPrep
from .meth_seg import MethSeg, MethSegMethod

# # %%
# FILTER_SAMPLES = False
# NUM_FILTER = 20
# N_JOBS = 50


class GenerateMethylationRegions:

    def __init__(
        self,
        data_prep: DataPrep,
        seg_method: MethSegMethod,
        selected_samples_path,
        filter_samples: Union[bool, list] = None,
        num_filter=20,
        n_jobs=50,
        out_dir=".",
        n_states=3,
        disable_loading_from_cache=False,
    ):
        self.data_prep = data_prep
        self.seg_method = seg_method
        self.filter_samples = filter_samples
        self.num_filter = num_filter
        self.n_jobs = n_jobs
        self.out_dir = out_dir
        self.selected_samples_path = selected_samples_path

        if self.files_exist() and not disable_loading_from_cache:
            self.predicted_states = pd.read_csv(
                f"{self.out_dir}/methylation_regions.csv"
            )
            self.segmentor = joblib.load(
                f"{self.out_dir}/methylation_segmentor_{self.seg_method.name}.joblib"
            )
            self.samples_info = pd.read_csv(self.selected_samples_path, sep="\t")

    def load_data(self):
        self.all_methylation_data = self.data_prep.all_methylation_data
        self.all_methylation_count = self.data_prep.all_methylation_count
        self.prepped_methylation_data = self.data_prep.test_data
        self.samples_info = pd.read_csv(self.selected_samples_path, sep="\t")

        self.cancer_samples = self.samples_info[
            self.samples_info["sample_type"] != "Solid Tissue Normal"
        ]["sample"].values

        self.samples = self.all_methylation_data.columns[3:]

        if self.filter_samples:
            if isinstance(self.filter_samples, list):
                self.cancer_samples = [
                    sample
                    for sample in self.cancer_samples
                    if sample in self.filter_samples
                ]
            elif isinstance(self.filter_samples, bool) and self.filter_samples:
                self.cancer_samples = self.cancer_samples[: self.num_filter]
        self.all_methylation_data_filtered = self.all_methylation_data[
            list(self.all_methylation_data.columns[:3]) + list(self.cancer_samples)
        ]
        self.all_methylation_count_filtered = self.all_methylation_count[
            list(self.all_methylation_count.columns[:3]) + list(self.cancer_samples)
        ]
        self.prepped_methylation_data_filtered = self.prepped_methylation_data[
            list(self.prepped_methylation_data.columns[:3]) + list(self.cancer_samples)
        ]

    def prep_segmentors(self, summary_measurement="pct"):
        n_components = 2 if summary_measurement == "pct" else 3

        if self.seg_method == MethSegMethod.CT_HMM:
            holding_time_guess = 100000

            n_emissions = len(self.prepped_methylation_data_filtered)
            ct_segmentor = MethSeg(
                segmentation_method=MethSegMethod.CT_HMM,
                aggregation_method=MethSegMethod.GAUSSIAN_HMM,
                methylation_data=self.prepped_methylation_data_filtered,
                cpg_count_data=self.all_methylation_count_filtered,
                random_state=14,
                model_args={
                    "holding_time": holding_time_guess,
                    "n_emissions": n_emissions,
                    "n_states": n_components,
                },
            )
            ct_fit_args = {
                "fit_startprob": True,
                "verbose": False,
                "max_iter": 10,
            }
            self.segmentor, self.segmentor_args = ct_segmentor, ct_fit_args
        elif self.seg_method == MethSegMethod.GAUSSIAN_HMM:

            dt_segmentor = MethSeg(
                segmentation_method=MethSegMethod.GAUSSIAN_HMM,
                aggregation_method=MethSegMethod.GAUSSIAN_HMM,
                methylation_data=self.prepped_methylation_data_filtered,
                cpg_count_data=self.all_methylation_count_filtered,
                random_state=42,
                model_args={
                    "covariance_type": "full",
                    "n_components": n_components,
                },
            )

            self.segmentor, self.segmentor_args = dt_segmentor, None

        elif self.seg_method == MethSegMethod.GMM_HMM:

            gmm_segmentor = MethSeg(
                segmentation_method=MethSegMethod.GMM_HMM,
                aggregation_method=MethSegMethod.GAUSSIAN_HMM,
                methylation_data=self.prepped_methylation_data_filtered,
                cpg_count_data=self.all_methylation_count_filtered,
                random_state=3,
                model_args={
                    "n_components": n_components,
                    "n_mix": 1,
                },
            )

            self.segmentor, self.segmentor_args = gmm_segmentor, None

        elif self.seg_method == MethSegMethod.POISSON_HMM:

            poisson_segmentor = MethSeg(
                segmentation_method=MethSegMethod.POISSON_HMM,
                aggregation_method=MethSegMethod.GAUSSIAN_HMM,
                methylation_data=self.prepped_methylation_data_filtered,
                cpg_count_data=self.all_methylation_count_filtered,
                random_state=3,
                model_args={
                    "n_components": n_components,
                },
            )
            self.segmentor, self.segmentor_args = poisson_segmentor, None

        elif self.seg_method == MethSegMethod.WINDOW:

            window_segmentor = MethSeg(
                segmentation_method=MethSegMethod.WINDOW,
                aggregation_method=MethSegMethod.GAUSSIAN_HMM,
                methylation_data=self.prepped_methylation_data_filtered,
                cpg_count_data=self.all_methylation_count_filtered,
                random_state=3,
            )

            self.segmentor, self.segmentor_args = window_segmentor, None

    def init(self):
        self.load_data()
        self.prep_segmentors()

    def run(self, show_plots=False):
        self.init()
        self.predicted_states = self.segmentor.aggregate_genomic_regions(
            n_jobs=self.n_jobs, fit_args=self.segmentor_args
        )

        self.segmentor.plot_state_distribution(
            f"{self.segmentor.segmentation_method.name} Median Methylation",
            "avg_summary_stat",
            bins=30,
            density=True,
            scale="linear",
            base=None,
            save_file=f"{self.out_dir}/methylation_distribution.png",
            show_plots=show_plots,
        )
        self.save_results()

    def save_results(self):
        # joblib.dump(
        #     self.segmentor,
        #     f"{self.out_dir}/methylation_segmentor_{self.seg_method.name}.joblib",
        # )
        self.predicted_states.to_csv(
            f"{self.out_dir}/methylation_regions.csv", index=False
        )

    def files_exist(self):
        return all(
            [
                os.path.exists(f"{self.out_dir}/methylation_regions.csv"),
                os.path.exists(
                    f"{self.out_dir}/methylation_segmentor_{self.seg_method.name}.joblib"
                ),
            ]
        )
