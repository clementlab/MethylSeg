import gc
import multiprocessing
import os
from matplotlib import gridspec, pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from pybedtools import BedTool
import pybedtools
import pyarrow.feather as feather
from pybedtools import BedTool
import numpy as np
from tqdm import tqdm


def get_methylation_df(samples_file, meth_ref_file, sample):
    sample_info = pd.read_csv(samples_file, sep="\t")
    meth_ref = pd.read_csv(meth_ref_file, sep="\t")

    meth_vals = {}
    meth_file = sample_info[sample_info["sample"] == sample]["methylation_file"].values[
        0
    ]
    meth = np.load(meth_file).astype(float)
    meth[meth == 255] = np.nan

    meth_vals[sample] = meth

    methylation_values_df = pd.DataFrame(meth_vals)

    methylation_df = pd.concat(
        [
            meth_ref.reset_index(drop=True),
            methylation_values_df.reset_index(drop=True),
        ],
        axis=1,
    )

    methylation_df.dropna(subset=["CpG_beg"], inplace=True)

    methylation_df["start"] = methylation_df["CpG_beg"] - 1
    methylation_df["end"] = methylation_df["CpG_beg"]
    methylation_df["chrom"] = methylation_df["CpG_chrm"]

    # Drop unused columns and convert relevant columns to integers with proper error handling.
    columns_to_drop = ["CpG_beg", "CpG_chrm"] + list(
        set(methylation_df.columns)
        - set(meth_vals.keys())
        - {"CpG_ beg", "CpG_chrm", "start", "end", "chrom"}
    )
    methylation_df = methylation_df.drop(columns=columns_to_drop)

    data_types = {"chrom": str, "start": int, "end": int}

    # Convert specified columns to defined types
    for col, dtype in data_types.items():
        methylation_df[col] = methylation_df[col].astype(dtype)

    cols_to_convert = methylation_df.columns.drop(data_types.keys())
    methylation_df[cols_to_convert] = methylation_df[cols_to_convert].astype(float)

    # reorder columns so it is chrom, start, end, sample1, sample2, ...
    methylation_df = methylation_df[["chrom", "start", "end"] + list(meth_vals.keys())]
    methylation_df = methylation_df.dropna(
        subset=methylation_df.iloc[:, 3:].columns, how="all"
    )
    methylation_df.reset_index(drop=True, inplace=True)

    return methylation_df


def create_sample_summary(
    args,
):
    (
        sample,
        genome_window_file,
        genome_file,
        samples_file,
        meth_ref_file,
        out_dir,
        low_cut_off,
        high_cut_off,
        # aggregation_method,
    ) = args
    methylation_df = get_methylation_df(samples_file, meth_ref_file, sample)
    genome_windows_bed = BedTool(genome_window_file)

    methylation_df["count_int"] = np.where(
        (methylation_df[sample] < low_cut_off)
        | (methylation_df[sample] > high_cut_off),
        0,
        1,
    )

    methylation_df["count_low"] = np.where(methylation_df[sample] <= low_cut_off, 1, 0)
    methylation_df["count_high"] = np.where(
        methylation_df[sample] >= high_cut_off, 1, 0
    )
    count_int_index = methylation_df.columns.get_loc("count_int") + 1
    count_low_index = methylation_df.columns.get_loc("count_low") + 1
    count_high_index = methylation_df.columns.get_loc("count_high") + 1

    all_meth_bed = BedTool.sort(BedTool.from_dataframe(methylation_df), g=genome_file)
    average_methylation_bed = genome_windows_bed.map(
        b=all_meth_bed, c=4, o="mean", g=genome_file
    )
    median_methylation_bed = genome_windows_bed.map(
        b=all_meth_bed, c=4, o="median", g=genome_file
    )
    int_count_bed = genome_windows_bed.map(
        b=all_meth_bed, c=count_int_index, o="sum", g=genome_file
    )
    low_count_bed = genome_windows_bed.map(
        b=all_meth_bed, c=count_low_index, o="sum", g=genome_file
    )
    high_count_bed = genome_windows_bed.map(
        b=all_meth_bed, c=count_high_index, o="sum", g=genome_file
    )

    cpg_count_bed = genome_windows_bed.map(
        b=all_meth_bed, c=4, o="count", g=genome_file
    )
    summary_methylation_df = average_methylation_bed.to_dataframe(
        names=["chrom", "start", "end", "mean"]
    )
    median_methylation_df = median_methylation_bed.to_dataframe(
        names=["chrom", "start", "end", "median"]
    )
    int_count_df = int_count_bed.to_dataframe(
        names=["chrom", "start", "end", "count_int"]
    )
    summary_methylation_df["count_int"] = int_count_df["count_int"]
    low_count_df = low_count_bed.to_dataframe(
        names=["chrom", "start", "end", "count_low"]
    )
    summary_methylation_df["count_low"] = low_count_df["count_low"]
    high_count_df = high_count_bed.to_dataframe(
        names=["chrom", "start", "end", "count_high"]
    )
    summary_methylation_df["count_high"] = high_count_df["count_high"]

    count_df = cpg_count_bed.to_dataframe(names=["chrom", "start", "end", "count"])
    summary_methylation_df["CpG_count"] = count_df["count"]
    summary_methylation_df["avg_methylation"] = summary_methylation_df["mean"]
    summary_methylation_df["median_methylation"] = median_methylation_df["median"]

    # Replace "." with None for relevant columns
    for col in [
        "avg_methylation",
        "median_methylation",
        "count_int",
        "count_low",
        "count_high",
    ]:
        summary_methylation_df[col] = summary_methylation_df[col].replace(".", np.nan)

    summary_methylation_df["count_int"] = pd.to_numeric(
        summary_methylation_df["count_int"]
    )
    summary_methylation_df["count_high"] = pd.to_numeric(
        summary_methylation_df["count_high"]
    )
    summary_methylation_df["count_low"] = pd.to_numeric(
        summary_methylation_df["count_low"]
    )
    summary_methylation_df["CpG_count"] = pd.to_numeric(
        summary_methylation_df["CpG_count"]
    )
    # Avoid division by zero
    summary_methylation_df["CpG_count_safe"] = summary_methylation_df[
        "CpG_count"
    ].replace(0, np.nan)

    # Calculate the percentages
    summary_methylation_df["int_pct"] = (
        summary_methylation_df["count_int"] / summary_methylation_df["CpG_count_safe"]
    )
    summary_methylation_df["high_pct"] = (
        summary_methylation_df["count_high"] / summary_methylation_df["CpG_count_safe"]
    )
    summary_methylation_df["low_pct"] = (
        summary_methylation_df["count_low"] / summary_methylation_df["CpG_count_safe"]
    )

    summary_dir = os.path.join(out_dir, "window_data", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    save_file = f"{out_dir}/window_data/summary/{sample}.summary.npy"
    np.save(
        save_file,
        summary_methylation_df[
            [
                "avg_methylation",
                "median_methylation",
                "int_pct",
                "high_pct",
                "low_pct",
                "CpG_count",
            ]
        ].values,
        allow_pickle=True,
    )
    pybedtools.cleanup()
    return summary_methylation_df, save_file


class WindowedMethylationPreProcessor:

    def __init__(
        self,
        meth_ref_file,
        samples_file,
        window_size,
        step_size,
        genome_file,
        low_cut_off=20,
        high_cut_off=70,
        tmp_dir="/tmp",
        out_dir=".",
        n_jobs=1,
    ):
        self.meth_ref_file = meth_ref_file
        self.samples_file = samples_file
        self.window_size = window_size
        self.step_size = step_size
        self.genome_file = genome_file
        self.tmp_dir = tmp_dir
        self.out_dir = out_dir
        self.n_jobs = n_jobs
        self.low_cut_off = low_cut_off
        self.high_cut_off = high_cut_off

        if self.files_exist():
            self.genome_windows_bed = BedTool(f"{self.out_dir}/hg38_windows.bed")
            self.genome_windows_df = self.genome_windows_bed.to_dataframe()
            self.genome_windows_ref = f"{self.out_dir}/hg38_windows.bed"
            self.sample_info = pd.read_csv(
                f"{self.out_dir}/runAll.sh.samples.hm450k.means", sep="\t"
            )
        self.load_data()

    def run(self, show_plots=False):
        self.create_windows_ref()
        self.generate_windowed_data()
        self.make_plots(show_plots=show_plots)

    def files_exist(self):
        return all(
            [
                os.path.exists(f"{self.out_dir}/hg38_windows.bed"),
                os.path.exists(f"{self.out_dir}/window_step_sizes.tsv"),
                os.path.exists(f"{self.out_dir}/runAll.sh.samples.hm450k.means"),
            ]
        )

    def _run_single_sample(self, sample):
        return create_sample_summary(
            [
                sample,
                self.genome_windows_ref,
                self.genome_file,
                self.samples_file,
                self.meth_ref_file,
                self.out_dir,
                20,
                70,
            ]
        )

    def generate_windowed_data(self, aggregation_method="mean"):
        sample_file_name = os.path.basename(self.samples_file)
        pybedtools.set_tempdir(self.tmp_dir)
        self.sample_info = pd.read_csv(self.samples_file, sep="\t")
        num_cpus = self.n_jobs
        column_name = "sample"

        total_to_process = len(self.sample_info[column_name])
        chunksize = 1

        pool = multiprocessing.Pool(processes=num_cpus)
        print(f"Processing {total_to_process} samples with {num_cpus} CPUs")
        args = [
            [
                sample,
                self.genome_windows_ref,
                self.genome_file,
                self.samples_file,
                self.meth_ref_file,
                self.out_dir,
                self.low_cut_off,
                self.high_cut_off,
                # aggregation_method,
            ]
            for sample in self.sample_info[column_name][:total_to_process]
        ]

        if not os.path.exists(f"{self.out_dir}/window_data/summary"):
            os.makedirs(f"{self.out_dir}/window_data/summary")

        for _ in tqdm(
            pool.imap_unordered(
                create_sample_summary,
                args,
                chunksize=chunksize,
            ),
            total=total_to_process,
        ):
            gc.collect()
        self.sample_info["summary_methylation_file"] = self.sample_info["sample"].apply(
            lambda sample: f"{self.out_dir}/window_data/summary/{sample}.summary.npy"
        )
        self.sample_info.to_csv(
            f"{self.out_dir}/{sample_file_name}.hm450k.summary", sep="\t", index=False
        )

    def load_data(self):
        self.sample_info = pd.read_csv(self.samples_file, sep="\t")
        self.meth_ref = pd.read_csv(self.meth_ref_file, sep="\t")

    def plot_single_sample_summary(self, sample, summary_df, region, show_plots=False):
        # Get raw methylation data
        methylation_df = get_methylation_df(
            self.samples_file, self.meth_ref_file, sample
        )

        # Unpack region
        chrom, start, end = region
        adj_start = start - (end - start) // 4
        adj_end = end + (end - start) // 4

        # Filter summary_df to current region
        region_df = summary_df[
            (summary_df["chrom"] == chrom)
            & (summary_df["start"] >= adj_start)
            & (summary_df["end"] <= adj_end)
        ]

        # Filter methylation_df to region
        raw_region_df = methylation_df[
            (methylation_df["chrom"] == chrom)
            & (methylation_df["start"] >= adj_start)
            & (methylation_df["end"] <= adj_end)
        ]

        # Summary metrics and scale types
        summary_metrics = [
            ("avg_methylation", 100),
            ("median_methylation", 100),
            ("int_pct", 1),
            ("high_pct", 1),
            ("low_pct", 1),
        ]

        n_summary = len(summary_metrics)
        n_cols = 3
        n_summary_rows = (n_summary + n_cols - 1) // n_cols

        # Set up figure with GridSpec
        total_rows = 1 + n_summary_rows  # 1 for raw plot
        fig = plt.figure(figsize=(5 * n_cols, 2.8 * total_rows))
        gs = gridspec.GridSpec(total_rows, n_cols, figure=fig)

        # --- Plot raw methylation across the full top row ---
        ax_raw = fig.add_subplot(gs[0, :])  # span all columns
        raw_vals = raw_region_df[sample]
        colors = np.where((raw_vals >= 20) & (raw_vals <= 70), "red", "blue")
        ax_raw.scatter(raw_region_df["start"], raw_vals, color=colors, s=10)
        ax_raw.set_ylabel("Raw Methylation")
        ax_raw.set_title(f"Methylation Profile for {sample} ({chrom}:{start}-{end})")
        ax_raw.set_ylim(0, 100)
        ax_raw.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_raw.grid(True)
        ax_raw.axvline(start, color="black", linestyle="--", linewidth=1)
        ax_raw.axvline(end, color="black", linestyle="--", linewidth=1)

        # --- Plot each summary metric ---
        for i, (metric, max_y) in enumerate(summary_metrics):
            row = (i // n_cols) + 1  # +1 to skip raw row
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            y = pd.to_numeric(region_df[metric], errors="coerce")
            colors = (
                np.where((y >= 20) & (y <= 70), "red", "blue")
                if max_y == 100
                else "blue"
            )
            ax.scatter(region_df["start"], y, color=colors, s=10)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_ylim(0, max_y)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=10))
            ax.grid(True)
            ax.set_xlabel("Genomic Position")
            ax.axvline(start, color="black", linestyle="--", linewidth=1)
            ax.axvline(end, color="black", linestyle="--", linewidth=1)

        # --- Remove unused grid slots (if any) ---
        total_cells = n_summary_rows * n_cols
        for i in range(n_summary, total_cells):
            row = (i // n_cols) + 1
            col = i % n_cols
            fig.add_subplot(gs[row, col]).axis("off")

        plt.tight_layout()

        # Save and optionally show
        outfile = f"{self.out_dir}/{sample}_{chrom}_{start}_{end}_summary_grid.png"
        plt.savefig(outfile)

        if show_plots:
            plt.show()
            plt.close()

    def make_plots(self, show_plots=False):
        # TODO: adjust to plot all summary metrics
        all_meth_values = []
        all_median_values = []
        all_int_pct_values = []
        all_high_pct_values = []
        all_low_pct_values = []

        for file_path in self.sample_info["summary_methylation_file"]:
            # Summary file columns
            # "avg_methylation",
            # "median_methylation",
            # "int_pct",
            # "high_pct",
            # "low_pct",
            # "CpG_count",
            data = np.load(file_path, allow_pickle=True)
            avg_meth = data[:, 0]
            med_meth = data[:, 1]
            int_pct = data[:, 2]
            high_pct = data[:, 3]
            low_pct = data[:, 4]
            avg_meth[avg_meth == "."] = np.nan
            med_meth[med_meth == "."] = np.nan
            int_pct[int_pct == "."] = np.nan
            high_pct[high_pct == "."] = np.nan
            low_pct[low_pct == "."] = np.nan

            # Collect the second column (index 1) values
            all_meth_values.extend(avg_meth.astype(float))
            all_median_values.extend(med_meth.astype(float))
            all_int_pct_values.extend(int_pct.astype(float))
            all_high_pct_values.extend(high_pct.astype(float))
            all_low_pct_values.extend(low_pct.astype(float))

        all_meth_values = np.array(all_meth_values)
        all_median_values = np.array(all_median_values)
        all_int_pct_values = np.array(all_int_pct_values)
        all_high_pct_values = np.array(all_high_pct_values)
        all_low_pct_values = np.array(all_low_pct_values)

        all_data = {
            "Average Methylation": all_meth_values,
            "Median Methylation": all_median_values,
            "Intermediate Percentage": all_int_pct_values,
            "High Percentage": all_high_pct_values,
            "Low Percentage": all_low_pct_values,
        }

        for metric, values in all_data.items():

            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=100)
            plt.title(f"{metric} Distribution")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)

            plt.savefig(f"{self.out_dir}/{metric.lower().replace(' ', '_')}_hist.png")

            if show_plots:
                plt.show()

    def create_windows_ref(
        self,
    ):
        pybedtools.set_tempdir(self.tmp_dir)

        df = pd.DataFrame(
            {"window_size": [self.window_size], "step_size": [self.step_size]}
        )

        df.to_csv(f"{self.out_dir}/window_step_sizes.tsv", sep="\t", index=False)
        self.genome_windows_bed = BedTool().makewindows(
            g=self.genome_file, w=self.window_size, s=self.step_size
        )

        self.genome_windows_df = self.genome_windows_bed.to_dataframe(
            names=["chrom", "start", "end"]
        )

        self.genome_windows_df = self.genome_windows_df[
            self.genome_windows_df.chrom.isin(self.meth_ref.CpG_chrm.unique())
        ]

        self.genome_windows_bed = BedTool.from_dataframe(self.genome_windows_df)
        self.genome_windows_bed = self.genome_windows_bed.sort(g=self.genome_file)

        self.genome_windows_df = self.genome_windows_bed.to_dataframe()
        self.genome_windows_bed.saveas(f"{self.out_dir}/hg38_windows.bed")
        pybedtools.cleanup()
        self.genome_windows_ref = f"{self.out_dir}/hg38_windows.bed"
