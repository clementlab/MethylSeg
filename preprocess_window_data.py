import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from data_preprocessor import (
    DataPreprocessor,
)
from windowed_methylation_preprocessor import (
    WindowedMethylationPreProcessor,
)


class WindowPreprocessor(DataPreprocessor):

    def __init__(
        self,
        meth_ref_file,
        samples_file,
        selected_samples_file,
        window_size,
        step_size,
        genome_file,
        tmp_dir="/tmp",
        out_dir=".",
        disable_loading_from_cache=False,
        n_jobs=1,
    ):
        self.windowed_methylation_preprocessor = WindowedMethylationPreProcessor(
            meth_ref_file=meth_ref_file,
            samples_file=samples_file,
            window_size=window_size,
            step_size=step_size,
            genome_file=genome_file,
            tmp_dir=tmp_dir,
            out_dir=out_dir,
            n_jobs=n_jobs,
        )
        self.selected_samples_file = selected_samples_file
        self.out_dir = out_dir
        super().__init__(samples_file, meth_ref_file)

        if self.files_exist() and not disable_loading_from_cache:
            for attr, file_path in self._get_files():
                if file_path.endswith(".tsv"):
                    self.__setattr__(attr, pd.read_csv(file_path, sep="\t"))
                elif file_path.endswith(".feather"):
                    self.__setattr__(attr, pd.read_feather(file_path))

    def run(self, show_plots=False):
        self.load_data()
        self.windowed_methylation_preprocessor.run(show_plots=show_plots)
        self.window_ref = self.windowed_methylation_preprocessor.genome_windows_df
        self.window_ref.columns = ["chr", "start", "end"]
        self.sample_info = self.windowed_methylation_preprocessor.sample_info
        self.calculate_files()
        print("Saving data")
        self.save_data()

    def _get_files(self):
        return [
            ("all_sample_meth_vals", f"{self.out_dir}/all_sample_meth_vals.feather"),
            (
                "all_sample_avg_windows_vals",
                f"{self.out_dir}/all_sample_avg_windows_vals.feather",
            ),
            (
                "all_sample_median_windows_vals",
                f"{self.out_dir}/all_sample_median_windows_vals.feather",
            ),
            (
                "all_sample_intermediate_windows_pct",
                f"{self.out_dir}/all_sample_intermediate_windows_pct.feather",
            ),
            (
                "all_sample_high_windows_pct",
                f"{self.out_dir}/all_sample_high_windows_pct.feather",
            ),
            (
                "all_sample_low_windows_pct",
                f"{self.out_dir}/all_sample_low_windows_pct.feather",
            ),
            (
                "all_sample_windows_counts",
                f"{self.out_dir}/all_sample_windows_counts.feather",
            ),
            ("sample_info", f"{self.out_dir}/sample_info.tsv"),
        ]

    def save_data(self):
        for attr, file_path in self._get_files():
            if not hasattr(self, attr):
                raise ValueError(f"Attribute {attr} not found in object.")
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if file_path.endswith(".tsv"):
                    self.__getattribute__(attr).to_csv(file_path, sep="\t", index=False)
                elif file_path.endswith(".feather"):
                    self.__getattribute__(attr).to_feather(file_path)

    def files_exist(self):
        for attr, file_path in self._get_files():
            if not os.path.exists(file_path):
                return False
        return self.windowed_methylation_preprocessor.files_exist()

    def load_data(self):
        self.windowed_methylation_preprocessor.load_data()
        self.selected_samples = pd.read_csv(self.selected_samples_file, sep="\t")
        super().load_data()

    def calculate_files(self):
        all_sample_windows_vals = self.window_ref.copy()
        all_sample_windows_counts = self.window_ref.copy()
        all_sample_meth_vals = self.meth_ref.copy()[["CpG_chrm", "CpG_beg", "CpG_end"]]
        all_sample_meth_vals.columns = self.window_ref.columns

        average_meth_vals = {}
        median_meth_vals = {}
        intermed_meth_vals = {}
        high_meth_vals = {}
        low_meth_vals = {}
        cpg_counts = {}
        meth_vals = {}
        # TODO wrap in tqdm
        with tqdm(total=len(self.sample_info)) as pbar:
            pbar.set_description("Loading methylation data")
            pbar.set_postfix_str("Loading methylation data")
            pbar.update(0)
            for (
                row_index,
                sample_row,
            ) in self.sample_info.iterrows():

                sample_name = sample_row["sample"]
                pbar.set_postfix_str(f"Loading methylation data for {sample_name}")
                summary_meth_file = sample_row["summary_methylation_file"]
                # Summary file columns
                # "avg_methylation",
                # "median_methylation",
                # "int_pct",
                # "high_pct",
                # "low_pct",
                # "CpG_count",
                meth_file = sample_row["methylation_file"]
                meth = np.load(meth_file).astype(float)
                meth[meth == 255] = np.nan

                meth_vals[sample_name] = meth

                meth_summary = np.load(summary_meth_file, allow_pickle=True)
                meth_summary[meth_summary == "."] = np.nan

                def get_col_index(col_name):
                    col_names = [
                        "avg_methylation",
                        "median_methylation",
                        "int_pct",
                        "high_pct",
                        "low_pct",
                        "CpG_count",
                    ]
                    return col_names.index(col_name)

                average_meth_vals[sample_name] = meth_summary[
                    :, get_col_index("avg_methylation")
                ]
                median_meth_vals[sample_name] = meth_summary[
                    :, get_col_index("median_methylation")
                ]
                intermed_meth_vals[sample_name] = meth_summary[
                    :, get_col_index("int_pct")
                ]
                high_meth_vals[sample_name] = meth_summary[:, get_col_index("high_pct")]
                low_meth_vals[sample_name] = meth_summary[:, get_col_index("low_pct")]
                cpg_counts[sample_name] = meth_summary[:, get_col_index("CpG_count")]
                pbar.update(1)

        average_methylation_values_df = pd.DataFrame(average_meth_vals)
        median_meth_vals_df = pd.DataFrame(median_meth_vals)
        intermed_meth_vals_df = pd.DataFrame(intermed_meth_vals)
        high_meth_vals_df = pd.DataFrame(high_meth_vals)
        low_meth_vals_df = pd.DataFrame(low_meth_vals)
        all_sample_avg_windows_vals = pd.concat(
            [
                all_sample_windows_vals.reset_index(drop=True),
                average_methylation_values_df.reset_index(drop=True),
            ],
            axis=1,
        )

        all_sample_median_windows_vals = pd.concat(
            [
                all_sample_windows_vals.reset_index(drop=True),
                median_meth_vals_df.reset_index(drop=True),
            ],
            axis=1,
        )

        all_sample_intermediate_windows_pct = pd.concat(
            [
                all_sample_windows_vals.reset_index(drop=True),
                intermed_meth_vals_df.reset_index(drop=True),
            ],
            axis=1,
        )

        all_sample_high_windows_pct = pd.concat(
            [
                all_sample_windows_vals.reset_index(drop=True),
                high_meth_vals_df.reset_index(drop=True),
            ],
            axis=1,
        )

        all_sample_low_windows_pct = pd.concat(
            [
                all_sample_windows_vals.reset_index(drop=True),
                low_meth_vals_df.reset_index(drop=True),
            ],
            axis=1,
        )

        cpg_counts_df = pd.DataFrame(cpg_counts)
        all_sample_windows_counts = pd.concat(
            [
                all_sample_windows_counts.reset_index(drop=True),
                cpg_counts_df.reset_index(drop=True),
            ],
            axis=1,
        )
        methylation_values_df = pd.DataFrame(meth_vals)
        all_sample_meth_vals = pd.concat(
            [
                all_sample_meth_vals.reset_index(drop=True),
                methylation_values_df.reset_index(drop=True),
            ],
            axis=1,
        )

        self.all_sample_meth_vals = all_sample_meth_vals
        self.all_sample_avg_windows_vals = all_sample_avg_windows_vals
        self.all_sample_median_windows_vals = all_sample_median_windows_vals
        self.all_sample_intermediate_windows_pct = all_sample_intermediate_windows_pct
        self.all_sample_high_windows_pct = all_sample_high_windows_pct
        self.all_sample_low_windows_pct = all_sample_low_windows_pct
        self.all_sample_windows_counts = all_sample_windows_counts

    def get_methylation_df(
        self, summary_metric="mean", target_region_type="intermediate"
    ):
        if summary_metric == "mean":
            return self.all_sample_avg_windows_vals
        elif summary_metric == "median":
            return self.all_sample_median_windows_vals
        elif summary_metric == "pct":
            if target_region_type == "intermediate":
                return self.all_sample_intermediate_windows_pct
            elif target_region_type == "high":
                return self.all_sample_high_windows_pct
            elif target_region_type == "low":
                return self.all_sample_low_windows_pct
            else:
                raise ValueError(
                    f"Invalid target region type: {target_region_type}. Must be one of 'intermediate', 'high', 'low'."
                )
        else:
            raise ValueError(
                f"Invalid summary metric: {summary_metric}. Must be one of 'mean', 'median', 'pct'."
            )
