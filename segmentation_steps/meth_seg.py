import multiprocessing
from collections import Counter, OrderedDict
from enum import Enum

import cthmm
import numpy as np
import pandas as pd
from hmmlearn import hmm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from tqdm.auto import tqdm
from scipy.special import logit
from scipy.stats import boxcox


def boxcox_transform(x, lmbda=None):
    np_x = np.array(x, dtype=float)
    if lmbda is None:
        lmbda = boxcox(np_x + 1e-10)[1]  # Add small value to avoid log(0)
    transformed_x = boxcox(np_x + 1e-10, lmbda=lmbda)  # Add small value to avoid log(0)
    return transformed_x


def logit_transform(x):
    np_x = np.array(x, dtype=float)
    np_x = np.clip(np_x, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return logit(np_x)


def log_transform(x):
    np_x = np.array(x, dtype=float)
    np_x = np.clip(np_x, 1e-10, None)  # Avoid log(0)
    return np.log(np_x)


def z_score_transform(x):
    np_x = np.array(x, dtype=float)
    scaler = StandardScaler()
    return scaler.fit_transform(np_x.reshape(-1, 1)).flatten()


def logit_z_score_transform(x):
    transformed_x = logit_transform(x)
    return z_score_transform(transformed_x)


class MethSegMethod(Enum):
    POISSON_HMM = 0
    GAUSSIAN_HMM = 1
    CT_HMM = 2
    GMM_HMM = 3
    WINDOW = 4

    def __eq__(self, value: object) -> bool:
        if isinstance(value, MethSegMethod):
            return self.value == value.value
        return super().__eq__(value)


class MethState(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    OFFTARGET = -1

    def __str__(self):
        if self == MethState.LOW:
            return "Unmethylated"
        elif self == MethState.MEDIUM:
            return "Intermediatly Methylated"
        elif self == MethState.HIGH:
            return "Methylated"
        elif self == MethState.OFFTARGET:
            return "Off-Target"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, MethState):
            return self.value == value.value
        return super().__eq__(value)

    def __hash__(self):
        return super().__hash__()

    def __lt__(self, value: object) -> bool:
        if isinstance(value, MethState):
            return self.value < value.value
        return super().__lt__(value)


class MethSeg:
    ## Initialization
    def __init__(
        self,
        segmentation_method: MethSegMethod,
        aggregation_method: MethSegMethod,
        methylation_data,
        cpg_count_data,
        random_state=None,
        model_args: dict | None = None,
        summary_measurement="pct",
    ):
        """
        Args:
            segmentation_method (MethSegMethod): Method to use for segmentation
            aggregation_method (MethSegMethod): Method to use for aggregation
            methylation_data (pd.DataFrame): Methylation data to segment
            cpg_count_data (pd.DataFrame): CpG count data to use for segmentation
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
            model_args (dict, optional): Model arguments. Defaults to None.

        methylation_data should be a DataFrame with the following columns:
            - chr: Chromosome
            - median: Median CpG value for the region
            - distance: Distance between CpG sites
            - sample columns: Methylation values for each

        cpg_count_data should be a DataFrame with the following columns:
            - chr: Chromosome
            - start: Start of the region
            - end: End of the region
            - sample columns: Average CpG count for the region

        """
        self.segmentation_method = segmentation_method
        self.aggregation_method = aggregation_method
        self.model_args = model_args
        self.methylation_data = methylation_data
        self.cpg_count_data = cpg_count_data

        self.fitted_sample = None
        self.random_state = random_state
        self._last_method_run = None
        self.summary_measurement = summary_measurement
        self.transform_method = None

        self.model = self.get_model()

    def get_model(self, aggregation=False):
        method = self.aggregation_method if aggregation else self.segmentation_method
        model_args = self.model_args if not aggregation is not None else None
        # TODO: Add Beta HMM
        n_components = 2 if self.summary_measurement == "pct" else 3
        if method == MethSegMethod.GAUSSIAN_HMM:
            if model_args is None:
                model_args = {
                    "n_components": n_components,
                    "covariance_type": "full",
                }
            if self.random_state is not None:
                model_args["random_state"] = self.random_state
            model = hmm.GaussianHMM(**model_args)
        elif method == MethSegMethod.POISSON_HMM:
            if model_args is None:
                model_args = {
                    "n_components": n_components,
                }
            if self.random_state is not None:
                model_args["random_state"] = self.random_state
            model = hmm.PoissonHMM(**model_args)
        elif method == MethSegMethod.GMM_HMM:
            if model_args is None:
                model_args = {
                    "n_components": n_components,
                    "n_mix": 1,
                }
            if self.random_state is not None:
                model_args["random_state"] = self.random_state
            model = hmm.GMMHMM(**model_args)
        elif method == MethSegMethod.CT_HMM:
            if model_args is None:
                model_args = {
                    "n_states": n_components,
                    "n_emissions": len(self.methylation_data),
                    "holding_time": 1,
                }
            if self.random_state is not None:
                model_args["seed"] = self.random_state
            model = cthmm.MultinomialCTHMM(**model_args)

        elif method == MethSegMethod.WINDOW:
            if model_args is None:
                model_args = {}
            model = WindowSeg(**model_args)

        return model

    ## Public methods
    def fit(self, sample, fit_args: dict | None = None):
        self.fitted_sample = sample
        if self.segmentation_method == MethSegMethod.GAUSSIAN_HMM:
            return self.__fit_dt_hmm(sample, fit_args)
        elif self.segmentation_method == MethSegMethod.GMM_HMM:
            return self.__fit_dt_hmm(sample, fit_args)
        elif self.segmentation_method == MethSegMethod.POISSON_HMM:
            return self.__fit_dt_hmm(sample, fit_args)
        elif self.segmentation_method == MethSegMethod.CT_HMM:
            if fit_args is None:
                fit_args = {"fit_startprob": True, "verbose": False, "max_iter": 10}
            return self.__fit_ct_hmm(sample, fit_args)
        elif self.segmentation_method == MethSegMethod.WINDOW:
            return self.__fit_window(sample, fit_args)

    def predict(self, sample):
        if self.fitted_sample != sample:
            raise ValueError("Sample must be fitted before predicting")
        if self.segmentation_method == MethSegMethod.GAUSSIAN_HMM:
            return self.__predict_dt_hmm(sample)
        elif self.segmentation_method == MethSegMethod.POISSON_HMM:
            return self.__predict_dt_hmm(sample)
        elif self.segmentation_method == MethSegMethod.GMM_HMM:
            return self.__predict_dt_hmm(sample)
        elif self.segmentation_method == MethSegMethod.CT_HMM:
            return self.__predict_ct_hmm(sample)
        elif self.segmentation_method == MethSegMethod.WINDOW:
            return self.__predict_window(sample)

    def fit_predict(self, sample, fit_args: dict | None = None, reset=False):
        # if reset:
        #     self.__init__(
        #         self.segmentation_method, self.methylation_data, self.random_state, self.model_args
        #     )
        self.fit(sample, fit_args)
        return self.predict(sample)

    def generate_genomic_regions(
        self,
        sample,
        predicted_states=None,
        fit=False,
        region_type="intermediate",
    ):
        self._last_method_run = f"Single Sample Regions - {sample}"
        if not fit and self.fitted_sample != sample:
            raise ValueError("Sample must be fitted before generating genomic regions")
        if fit:
            self.fit(sample)
        if predicted_states is None:
            predicted_states = self.predict(sample)
        states_df = self.__prep_genomic_input(predicted_states)
        regions = self.__create_regions(states_df)
        regions_df = self._translate_state(
            self._regions_list_to_df(regions, [sample]), region_type=region_type
        )
        self.regions_df = regions_df
        return regions_df

    def aggregate_genomic_regions(
        self, n_jobs=1, fit_args: dict = {}, region_type="intermediate"
    ):
        if fit_args is None:
            fit_args = {}
        self._last_method_run = f"Aggregate all samples"
        samples = self.methylation_data.columns[3:]

        if self.segmentation_method == MethSegMethod.WINDOW:
            regions = self.model.aggregate_genomic_regions(self.methylation_data)
            regions_df = self._translate_state(
                self._regions_list_to_df(
                    regions, samples, ["count_commonly_methylated"]
                ),
                region_type=region_type,
            )
            self.regions_df = regions_df
            return regions_df
        elif self.segmentation_method == MethSegMethod.CT_HMM:
            states_df = self._aggregate_hmm(n_jobs, fit_args=fit_args)
        elif self.segmentation_method == MethSegMethod.GAUSSIAN_HMM:
            states_df = self._aggregate_hmm(n_jobs, fit_args=fit_args)
        elif self.segmentation_method == MethSegMethod.POISSON_HMM:
            states_df = self._aggregate_hmm(n_jobs, fit_args=fit_args)
        elif self.segmentation_method == MethSegMethod.GMM_HMM:
            states_df = self._aggregate_hmm(n_jobs, fit_args=fit_args)
        else:
            raise ValueError("Method does not support aggregating genomic regions")
        regions = self.__create_regions(states_df)
        regions_df = self._translate_state(self._regions_list_to_df(regions, samples))
        self.regions_df = regions_df
        return regions_df

    def plot_state_distribution(
        self,
        title,
        plot_col="avg_summary_stat",
        bins=30,
        density=True,
        scale="linear",
        base=None,
        save_file=None,
        show_plots=False,
    ):
        if self.regions_df is None:
            raise ValueError("No genomic regions have been generated")
        group_col = (
            "readable_state" if "readable_state" in self.regions_df.columns else "state"
        )
        self._graph_states(
            self.regions_df,
            group_col,
            plot_col,
            title + " - " + self._last_method_run,
            bins,
            density,
            scale,
            base,
            save_file,
            show_plots,
        )

    ## Helper methods
    def _average_CpGs_per_region(self, region, avg_CpG_count_df):
        avg_CpG_count_df
        overlapping_windows = avg_CpG_count_df[
            (avg_CpG_count_df["chr"] == region["chr"])
            & (avg_CpG_count_df["start"] < region["end"])
            & (avg_CpG_count_df["end"] > region["start"])
        ]
        if not overlapping_windows.empty:
            return overlapping_windows["average_cpg_count"].mean()
        else:
            return 0

    def _regions_list_to_df(self, regions, samples_list, extra_cols=[]):
        n_columns = len(regions[0])
        column_names = ["chr", "start", "end", "state"]
        if n_columns > len(column_names):
            for i in range(n_columns - len(column_names)):
                col_name = f"extra_{i}" if i >= len(extra_cols) else extra_cols[i]
                column_names.append(col_name)
        df = pd.DataFrame(regions, columns=column_names)
        df["chr"] = df["chr"].astype(str)
        df["start"] = df["start"].astype(float)
        df["end"] = df["end"].astype(float)
        df["state"] = df["state"].astype(int)

        df["avg_summary_stat"] = df.apply(
            lambda x: self.methylation_data[
                (self.methylation_data["chr"] == x["chr"])
                & (self.methylation_data["median"] >= x["start"])
                & (self.methylation_data["median"] <= x["end"])
            ][samples_list].mean(axis=None),
            axis=1,
        )
        df["length"] = df["end"] - df["start"]

        avg_CpG_count = self.cpg_count_data[samples_list].mean(axis=1)
        avg_CpG_count_df = pd.concat(
            [
                self.cpg_count_data.iloc[:, :3].reset_index(drop=True),
                avg_CpG_count.reset_index(drop=True),
            ],
            axis=1,
        )
        avg_CpG_count_df.columns = ["chr", "start", "end", "average_cpg_count"]

        df["avg_cpg_count"] = df.apply(
            lambda row: self._average_CpGs_per_region(row, avg_CpG_count_df), axis=1
        )
        return df

    # TODO remember how this works??
    def _translate_state(self, regions_df, region_type="intermediate"):
        regions_df_copy = regions_df.copy()
        if self.model.n_components == 3:
            translations = [MethState.LOW, MethState.MEDIUM, MethState.HIGH]
        elif self.model.n_components == 2:
            target = (
                MethState.MEDIUM
                if region_type == "intermediate"
                else MethState.HIGH if region_type == "high" else MethState.LOW
            )
            translations = [MethState.OFFTARGET, target]
        states_median = {}
        for i in range(len(translations)):
            states_median[i] = regions_df_copy[regions_df_copy["state"] == i][
                "avg_summary_stat"  # TODO Change avg_summary_stat to summary_stat and find where this is set
            ].median()
        states_median = OrderedDict(
            sorted(states_median.items(), key=lambda item: item[1])
        )
        translated_states = {}
        for i, k in enumerate(states_median.keys()):
            translated_states[k] = translations[i]
        regions_df_copy["readable_state"] = regions_df_copy["state"].apply(
            lambda x: translated_states[x]
        )
        return regions_df_copy

    def _graph_states(
        self,
        df,
        group_col,
        plot_col,
        title,
        bins=30,
        density=True,
        scale="linear",
        base=None,
        save_file=None,
        show_plots=False,
    ):

        grouped = df.groupby(group_col)

        plt.figure(figsize=(10, 6))

        for name, group in grouped:
            plt.hist(
                group[plot_col],
                bins=bins,
                alpha=0.6,
                label=name,
                density=density,
            )

        plt.xlabel(plot_col)
        plt.ylabel("Density")
        if base is not None:
            plt.xscale(scale, base=base)
        else:
            plt.xscale(scale)
        plt.title(title)
        plt.legend()
        if save_file is not None:
            plt.savefig(save_file)
        if show_plots:
            plt.show()
        plt.close()

    def _prep_fit_data(self, sample=None):
        meth_data = self.methylation_data[sample].astype(int)
        return self.__prep_fit_data(meth_data)

    ## Private methods

    def __prep_fit_data(self, data):
        # self.transform_method = boxcox_transform
        # self.segmentation_method = MethSegMethod.GAUSSIAN_HMM
        # self.model_args = {
        #     "n_components": 2,
        #     # "n_mix": 3,
        #     "n_iter": 100,
        # }
        # self.model = self.get_model()

        # print("here", self.segmentation_method, self.transform_method)

        if self.segmentation_method == MethSegMethod.CT_HMM:
            return [
                (
                    data.values,
                    self.methylation_data["median"].values,
                )
            ]
        elif (
            self.segmentation_method == MethSegMethod.GAUSSIAN_HMM
            or self.segmentation_method == MethSegMethod.GMM_HMM
            or self.segmentation_method == MethSegMethod.POISSON_HMM
        ):
            if self.transform_method is not None:
                data = self.transform_method(data)
                data = pd.Series(data)
            return data.values.reshape(-1, 1)
        elif self.segmentation_method == MethSegMethod.WINDOW:
            return data

    def __prep_predict_data(self, sample=None):
        if self.segmentation_method == MethSegMethod.CT_HMM:
            return [
                self.methylation_data[sample].astype(int).values,
                self.methylation_data["median"].values,
            ]
        elif (
            self.segmentation_method == MethSegMethod.GAUSSIAN_HMM
            or self.segmentation_method == MethSegMethod.GMM_HMM
            or self.segmentation_method == MethSegMethod.POISSON_HMM
        ):
            return self.methylation_data[sample].values.reshape(-1, 1)
        elif self.segmentation_method == MethSegMethod.WINDOW:
            return self.methylation_data[sample]

    def __fit_dt_hmm(self, sample, fit_args: dict | None = None):
        if fit_args is None:
            fit_args = {}
        data = self._prep_fit_data(sample)
        if hasattr(self.model, "transmat_"):
            del self.model.transmat_
        if hasattr(self.model, "lambdas_"):
            del self.model.lambdas_
        if hasattr(self.model, "startprob_"):
            del self.model.startprob_
        if hasattr(self.model, "means_"):
            del self.model.means_
        if hasattr(self.model, "_covars_"):
            del self.model._covars_
        return self.model.fit(data, **fit_args)

    def __fit_ct_hmm(self, sample, fit_args: dict | None = None):
        if fit_args is None:
            fit_args = {}
        data = self._prep_fit_data(sample)
        return self.model.fit_observation_params(data, **fit_args)

    def __fit_window(self, sample, fit_args: dict | None = None):
        if fit_args is None:
            fit_args = {}
        data = self._prep_fit_data(sample)
        return self.model.fit(data, **fit_args)

    def __predict_dt_hmm(self, sample):
        data = self.__prep_predict_data(sample)
        return self.model.predict(data)

    def __predict_ct_hmm(self, sample):
        data = self.__prep_predict_data(sample)
        return self.model.predict(data[0], data[1])

    def __predict_window(self, sample):
        data = self.__prep_predict_data(sample)
        return self.model.predict(data)

    def __prep_genomic_input(self, predicted_states):
        states_df = self.methylation_data[["chr", "median"]][["chr", "median"]].copy()
        states_df["state"] = predicted_states
        return states_df

    def __create_regions(self, df: pd.DataFrame):
        regions = []
        start = None
        end = None
        curr_state = None
        curr_chrom = None

        def close_region():
            nonlocal start, end, curr_state, curr_chrom
            if start is not None and end is not None:
                regions.append((curr_chrom, start, end, curr_state))
                start = None
                end = None

        for i, row in df.iterrows():
            # First loop open first window
            if curr_chrom is None:
                curr_chrom = row["chr"]
                start = row["median"]
                curr_state = row["state"]
            # If we have a new chromosome, close the current region
            elif curr_chrom != row["chr"]:
                close_region()
                curr_chrom = row["chr"]
                curr_state = row["state"]
                start = row["median"]
            elif curr_state != row["state"]:
                close_region()
                curr_state = row["state"]
                start = row["median"]
            else:
                end = row["median"]

        if start is not None and end is None:
            end = row["median"]
        close_region()

        return regions

    def _fit_predict_wrapper(self, sample, fit_args):
        local_model = MethSeg(
            self.segmentation_method,
            self.aggregation_method,
            self.methylation_data,
            self.cpg_count_data,
            self.random_state,
            self.model_args,
        )
        result = local_model.fit_predict(sample, fit_args)

        return result

    def __use_hmm_aggregation(self, predicted_states, fit_args):
        transposed_states = list(zip(*predicted_states))

        state_percentages = []
        total_states = len(predicted_states)

        for states in transposed_states:
            state_counts = Counter(states)
            percentages = []
            for i in range(self.model.n_components):
                percentages.append(state_counts.get(i, 0) / total_states * 100)
            state_percentages.append(percentages)
        aggregation_model = self.get_model(aggregation=True)
        if self.aggregation_method == MethSegMethod.CT_HMM:
            fit_data = self.__prep_fit_data(pd.Series(state_percentages))
            observations = fit_data[0][0]
            times = fit_data[0][1]
            observations = [np.argmax(obs) for obs in observations]
            fit_data = [(observations, times)]
            aggregation_model.fit_observation_params(fit_data, **fit_args)
            predicted_states = aggregation_model.predict(observations, times)
        else:
            fit_data = state_percentages
            aggregation_model.fit(fit_data, **fit_args)
            predicted_states = aggregation_model.predict(state_percentages)

        states_df = self.__prep_genomic_input(predicted_states)
        return states_df

    def __use_simple_aggregation(self, predicted_states, fit_args):
        transposed_states = list(zip(*predicted_states))

        state_percentages = []
        total_states = len(predicted_states)
        for states in transposed_states:
            state_counts = Counter(states)
            percentages = []
            for i in range(self.model.n_components):
                percentages.append(state_counts.get(i, 0) / total_states * 100)
            state_percentages.append(percentages)

        top_states = []
        for percentages in state_percentages:
            top_states.append(np.argmax(percentages))
        return self.__prep_genomic_input(top_states)

    def _aggregate_hmm(self, n_jobs, fit_args: dict = {}):
        use_hmm_aggregation = False
        use_simple_aggregation = True
        samples = self.methylation_data.columns[3:]
        n_jobs = 1  # TODO get multiprocessing to work
        if n_jobs == -1:
            n_jobs = min(multiprocessing.cpu_count(), len(samples))
        predicted_states = []

        if n_jobs == 1:
            with tqdm(
                total=len(samples), desc="Predicting States", disable=False
            ) as pbar:
                for sample in samples:
                    predicted_states.append(self._fit_predict_wrapper(sample, fit_args))
                    pbar.update(1)

        else:
            with Parallel(n_jobs=n_jobs) as parallel:
                predicted_states = list(
                    parallel(
                        delayed(self._fit_predict_wrapper)(sample, fit_args)
                        for sample in tqdm(samples)
                    ),
                )
        if use_hmm_aggregation:
            return self.__use_hmm_aggregation(predicted_states, fit_args)
        elif use_simple_aggregation:
            return self.__use_simple_aggregation(predicted_states, fit_args)


class WindowSeg:
    def __init__(
        self,
        meth_low=20,
        meth_high=70,
        interpercentile_low=20,
        interpercentile_high=70,
        lower_percent_cutoff=0.3,
        higher_percent_cutoff=0.6,
        percent_per_region=0.8,
        use_majority_criteria=False,
    ):
        self.meth_low = meth_low
        self.meth_high = meth_high
        self.interpercentile_low = interpercentile_low
        self.interpercentile_high = interpercentile_high
        self.lower_percent_cutoff = lower_percent_cutoff
        self.higher_percent_cutoff = higher_percent_cutoff
        self.percent_per_region = percent_per_region
        self.use_majority_criteria = use_majority_criteria

    def __fit_definition(self, data):
        def state(c):
            if c < self.meth_low:
                return MethState.LOW.value
            elif c > self.meth_high:
                return MethState.HIGH.value
            else:
                return MethState.MEDIUM.value

        return [state(x) for x in data]

    def fit(self, sample: pd.Series):
        return self.__fit_definition(sample)

    def predict(self, sample: pd.Series):
        return self.__fit_definition(sample)

    def aggregate_genomic_regions(self, data: pd.DataFrame):
        num_samples = len(data.columns[3:-1])

        criteria_df = self.__generate_criteria_df(data)

        criteria_fct = (
            self.__majority_criteria_fct
            if self.use_majority_criteria
            else self.__subset_criteria_fct
        )

        regions = {}
        for region_type in ["intermediate_count", "low_count", "high_count"]:
            regions[region_type] = self.__create_regions(
                criteria_df,
                region_type,
                num_samples,
                criteria_fct,
            )
        full_regions_data = []
        for region_type in regions:
            for region in regions[region_type]:
                state = (
                    MethState.LOW.value
                    if region_type == "low_count"
                    else (
                        MethState.HIGH.value
                        if region_type == "high_count"
                        else MethState.MEDIUM.value
                    )
                )
                full_regions_data.append(
                    [
                        region[0],
                        region[1],
                        region[2],
                        state,
                        region[3],
                    ]
                )
        return full_regions_data

    def __generate_criteria_df(self, data: pd.DataFrame):
        intermediate_count = data[data.columns[3:-1]].apply(
            lambda x: x[(x >= self.meth_low) & (x <= self.meth_high)].count(),
            axis=1,
        )
        low_count = data[data.columns[3:-1]].apply(
            lambda x: x[(x < self.meth_low)].count(), axis=1
        )
        high_count = data[data.columns[3:-1]].apply(
            lambda x: x[(x > self.meth_high)].count(), axis=1
        )

        percentile_5th = data[data.columns[3:-1]].quantile(0.05, axis=1)
        percentile_95th = data[data.columns[3:-1]].quantile(0.95, axis=1)
        interpercentile_range = percentile_95th - percentile_5th

        criteria_df = data[["chr", "median"]].copy()
        criteria_df["intermediate_count"] = intermediate_count
        criteria_df["low_count"] = low_count
        criteria_df["high_count"] = high_count
        criteria_df["interpercentile_range"] = interpercentile_range

        return criteria_df

    def __majority_criteria_fct(self, percent):
        return percent > 0.5

    def __subset_criteria_fct(self, percent):
        return (
            percent >= self.lower_percent_cutoff
            and percent <= self.higher_percent_cutoff
        )

    def __is_commonly_methylated_similar(
        self,
        window_count,
        total_samples,
        interpercentile_range,
        criteria_fct,
    ):
        percent_intermedate = window_count / total_samples
        return criteria_fct(percent_intermedate) and (
            interpercentile_range >= self.interpercentile_low
            and interpercentile_range <= self.interpercentile_high
        )

    def __begin_or_update_region(
        self,
        current_chr,
        current_start,
        current_common_count,
        current_count_in_window,
        row,
    ):
        if current_start is None:
            current_chr = row[1]["chr"]
            current_start = row[1]["median"]
        current_common_count += 1
        current_count_in_window += 1
        current_end = row[1]["median"]
        return (
            current_chr,
            current_start,
            current_end,
            current_common_count,
            current_count_in_window,
        )

    def __close_current_region(
        self,
        regions,
        current_chr,
        current_start,
        current_end,
        current_common_count,
    ):
        regions.append((current_chr, current_start, current_end, current_common_count))
        current_chr = None
        current_start = None
        current_end = None
        current_count_in_window = 0
        current_common_count = 0
        return (
            current_chr,
            current_start,
            current_end,
            current_common_count,
            current_count_in_window,
        )

    def __should_keep_window_open(self, current_common_count, current_count_in_window):
        if (
            current_count_in_window
            and (current_common_count / current_count_in_window)
            >= self.percent_per_region
        ):
            return True
        return False

    def __create_regions(
        self,
        criteria_df: pd.DataFrame,
        region_type,
        num_samples,
        criteria_fct,
    ):
        regions = []
        current_chr = None
        current_start = None
        current_end = None
        current_count_in_window = 0
        current_common_count = 0
        for row in criteria_df.iterrows():
            # Set initial chromosome
            if current_chr is None:
                current_chr = row[1]["chr"]
            # If we change chromosomes, close current region
            if current_chr != row[1]["chr"]:
                if current_end is not None:
                    (
                        current_chr,
                        current_start,
                        current_end,
                        current_common_count,
                        current_count_in_window,
                    ) = self.__close_current_region(
                        regions,
                        current_chr,
                        current_start,
                        current_end,
                        current_common_count,
                    )

            if self.__is_commonly_methylated_similar(
                row[1][region_type],
                num_samples,
                row[1]["interpercentile_range"],
                criteria_fct,
            ):
                (
                    current_chr,
                    current_start,
                    current_end,
                    current_common_count,
                    current_count_in_window,
                ) = self.__begin_or_update_region(
                    current_chr,
                    current_start,
                    current_common_count,
                    current_count_in_window,
                    row,
                )
            else:
                if self.__should_keep_window_open(
                    current_common_count,
                    current_count_in_window,
                ):
                    current_count_in_window += 1
                    continue
                else:
                    if current_end is not None:
                        (
                            current_chr,
                            current_start,
                            current_end,
                            current_common_count,
                            current_count_in_window,
                        ) = self.__close_current_region(
                            regions,
                            current_chr,
                            current_start,
                            current_end,
                            current_common_count,
                        )
        if current_end is not None:
            self.__close_current_region(
                regions,
                current_chr,
                current_start,
                current_end,
                current_common_count,
            )
        return regions
