import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from data_preprocessor import DataPreprocessor
from preprocess_window_data import WindowPreprocessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class DataPrep:

    def __init__(
        self,
        preprocessor: DataPreprocessor,
        use_window_averaging=False,
        window_summary_metric="pct",
        out_dir=".",
        disable_loading_from_cache=False,
    ):
        self.preprocessor = preprocessor
        self.out_dir = out_dir
        self.use_window_averaging = use_window_averaging
        self.window_summary_metric = window_summary_metric

        if self.files_exist() and not disable_loading_from_cache:
            self.test_data = pd.read_feather(
                f"{self.out_dir}/prepped_methylation_data.feather"
            )
            self.all_methylation_data = pd.read_feather(
                f"{self.out_dir}/all_methylation_data.feather"
            )
            self.all_methylation_count = pd.read_feather(
                f"{self.out_dir}/all_methylation_count.feather"
            )
            self.meth_ref = self.preprocessor.meth_ref

            self.samples_info = self.preprocessor.sample_info

    def run(
        self, filter_chroms=[], filter_regions=[], filter_samples=[], show_plots=False
    ):
        self.load_data()
        self.clean_data(
            filter_chroms=filter_chroms,
            filter_regions=filter_regions,
            filter_samples=filter_samples,
        )
        self.save_data()

    def load_data(self):
        if self.use_window_averaging:
            self.preprocessor: WindowPreprocessor
            all_methylation_data = self.preprocessor.get_methylation_df(
                self.window_summary_metric
            )
            all_methylation_count = self.preprocessor.all_sample_windows_counts
        else:
            raise NotImplementedError(
                "Full data preprocessing is not implemented yet. Please use window averaging."
            )
            # self.preprocessor: FullDataPreprocessor
            # all_methylation_data = self.preprocessor.methyl_df
            # if "chrom" in all_methylation_data.columns:
            #     all_methylation_data["chr"] = all_methylation_data["chrom"]
            #     all_methylation_data.drop(columns="chrom", inplace=True)
            # all_methylation_data = all_methylation_data[
            #     [
            #         "chr",
            #         "start",
            #         "end",
            #         *list(
            #             set(all_methylation_data.columns) - set(["chr", "start", "end"])
            #         ),
            #     ]
            # ]
            # all_methylation_count = pd.concat(
            #     [
            #         all_methylation_data[all_methylation_data.columns[:3]].reset_index(
            #             drop=True
            #         ),
            #         all_methylation_data[all_methylation_data.columns[3:]]
            #         .notnull()
            #         .astype(int)
            #         .reset_index(drop=True),
            #     ],
            #     axis=1,
            # )
        self.all_methylation_data = all_methylation_data
        self.all_methylation_count = all_methylation_count

        self.meth_ref = self.preprocessor.meth_ref

        self.samples_info = self.preprocessor.sample_info

    def clean_data(self, filter_chroms=[], filter_regions=[], filter_samples=[]):

        self.filtered_methylation_data = self.all_methylation_data.dropna(
            axis=0, how="all", subset=self.all_methylation_data.columns[3:]
        )
        self.filtered_methylation_data.reset_index(drop=True, inplace=True)
        logger.info(
            f"Filtered out {self.all_methylation_data.shape[0] - self.filtered_methylation_data.shape[0]} rows"
        )

        test_data = self.filtered_methylation_data.copy()
        if filter_chroms:
            test_data = test_data[test_data["chr"].isin(filter_chroms)].copy()
        if filter_samples:
            test_data = test_data[list(test_data.columns[0:3]) + filter_samples].copy()
        if filter_regions:
            chr, start, end = filter_regions
            test_data = test_data[
                (test_data["chr"] == chr)
                & (test_data["start"] >= start)
                & (test_data["end"] <= end)
            ].copy()

        self.interpolated_data = pd.DataFrame(
            SimpleImputer(strategy="mean")
            .fit_transform(test_data[test_data.columns[3:]].T)
            .T,
            columns=test_data.columns[3:],
        )

        test_data = pd.concat(
            [
                test_data[test_data.columns[:3]].reset_index(drop=True),
                self.interpolated_data.reset_index(drop=True),
            ],
            axis=1,
        )

        data_types = {"chr": str, "start": int, "end": int}

        # Convert specified columns to defined types
        for col, dtype in data_types.items():
            test_data[col] = test_data[col].astype(dtype)

        cols_to_convert = test_data.columns.drop(data_types.keys())
        test_data[cols_to_convert] = test_data[cols_to_convert].astype(float)
        test_data = test_data.copy()
        test_data["median"] = (test_data["end"] + test_data["start"]) / 2
        test_data["median"] = test_data["median"].astype(int)
        test_data = test_data.copy()

        def calculate_distance(df):
            dist = [-1]
            for i in range(1, len(df)):
                if df.loc[i, "chr"] == df.loc[i - 1, "chr"]:
                    dist.append(df.loc[i, "median"] - df.loc[i - 1, "median"])
                else:
                    dist.append(-1)
            return dist

        test_data["distance"] = calculate_distance(test_data)
        test_data = test_data[
            [
                "chr",
                "median",
                "distance",
            ]
            + list(
                set(test_data.columns)
                - {
                    "chr",
                    "median",
                    "distance",
                }
            )
        ].copy()
        test_data.drop(columns=["start", "end"], inplace=True)
        test_data = test_data.dropna(subset=test_data.columns[3:], how="all").copy()
        test_data.reset_index(drop=True, inplace=True)
        self.test_data = test_data.copy()

    def save_data(self):
        self.test_data.to_feather(f"{self.out_dir}/prepped_methylation_data.feather")
        self.all_methylation_data.to_feather(
            f"{self.out_dir}/all_methylation_data.feather"
        )
        self.all_methylation_count.to_feather(
            f"{self.out_dir}/all_methylation_count.feather"
        )

    def files_exist(self):
        return all(
            [
                os.path.exists(f"{self.out_dir}/prepped_methylation_data.feather"),
                os.path.exists(f"{self.out_dir}/all_methylation_data.feather"),
                os.path.exists(f"{self.out_dir}/all_methylation_count.feather"),
            ]
        )
