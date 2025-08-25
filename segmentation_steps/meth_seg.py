import multiprocessing
from collections import Counter, OrderedDict
from enum import Enum

import copy
import multiprocessing as mp
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


def _fit_single(args):
    self, sample, chrom, fit_args = args
    print("Fitting model for sample:", sample, "chromosome:", chrom)
    if chrom not in self.models[sample]:
        self.models[sample][chrom] = self.get_model(aggregation=False)
    model = copy.deepcopy(self.models[sample][chrom])
    self._fit_dt_hmm(model, chrom, sample, fit_args)
    return chrom, model


def _predict_single(args):
    self, sample, chrom = args
    model = self.models[sample][chrom]
    preds = self._predict_dt_hmm(model, chrom, sample)
    return preds


class MethSegMethod(Enum):
    POISSON_HMM = 0
    GAUSSIAN_HMM = 1
    GMM_HMM = 2

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
        n_jobs=1,
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
        self.n_jobs = 1

        self.models = {}
        for sample in self.methylation_data.columns[3:]:
            self.models[sample] = {}

        n_components = 2 if self.summary_measurement == "pct" else 3
        self.n_components = n_components

    def get_model(self, aggregation=False):
        method = self.aggregation_method if aggregation else self.segmentation_method
        model_args = self.model_args if not aggregation is not None else None
        # TODO: Add Beta HMM

        if method == MethSegMethod.GAUSSIAN_HMM:
            if model_args is None:
                model_args = {
                    "n_components": self.n_components,
                    "covariance_type": "full",
                }
            if self.random_state is not None:
                model_args["random_state"] = self.random_state
            model = hmm.GaussianHMM(**model_args)
        elif method == MethSegMethod.POISSON_HMM:
            if model_args is None:
                model_args = {
                    "n_components": self.n_components,
                }
            if self.random_state is not None:
                model_args["random_state"] = self.random_state
            model = hmm.PoissonHMM(**model_args)
        elif method == MethSegMethod.GMM_HMM:
            if model_args is None:
                model_args = {
                    "n_components": self.n_components,
                    "n_mix": 1,
                }
            if self.random_state is not None:
                model_args["random_state"] = self.random_state
            model = hmm.GMMHMM(**model_args)

        return model

    def fit(self, sample, fit_args: dict | None = None):
        if fit_args is None:
            fit_args = {}

        chroms = self.methylation_data["chr"].unique()
        args_list = [(self, sample, chrom, fit_args) for chrom in chroms]

        if self.n_jobs == 1:
            results = [_fit_single(args) for args in args_list]
        else:
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                results = pool.map(_fit_single, args_list)

        # collect fitted models back
        for chrom, model in results:
            self.models[sample][chrom] = model

    def predict(self, sample):
        if not self.models[sample]:
            raise ValueError("Sample must be fitted before predicting")

        chroms = self.methylation_data["chr"].unique()
        args_list = [(self, sample, chrom) for chrom in chroms]

        if self.n_jobs == 1:
            results = [_predict_single(args) for args in args_list]
        else:
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                results = pool.map(_predict_single, args_list)

        predicted_states = np.concatenate(results)
        return predicted_states

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

        if self.segmentation_method == MethSegMethod.GAUSSIAN_HMM:
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
        if self.n_components == 3:
            translations = [MethState.LOW, MethState.MEDIUM, MethState.HIGH]
        elif self.n_components == 2:
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

    def _prep_fit_data(self, sample=None, chrom=None):
        chrom_data = self.methylation_data[self.methylation_data["chr"] == chrom]
        pos = chrom_data["median"].values
        meth_data = chrom_data[sample].astype(int)
        return self.__prep_fit_data(meth_data), pos

    ## Private methods

    def __prep_fit_data(self, data):

        if (
            self.segmentation_method == MethSegMethod.GAUSSIAN_HMM
            or self.segmentation_method == MethSegMethod.GMM_HMM
            or self.segmentation_method == MethSegMethod.POISSON_HMM
        ):
            if self.transform_method is not None:
                data = self.transform_method(data)
                data = pd.Series(data)
            return data.values.reshape(-1, 1)

    def __prep_predict_data(self, sample=None, chrom=None):
        chrom_data = self.methylation_data[self.methylation_data["chr"] == chrom]
        pos = chrom_data["median"].values
        if (
            self.segmentation_method == MethSegMethod.GAUSSIAN_HMM
            or self.segmentation_method == MethSegMethod.GMM_HMM
            or self.segmentation_method == MethSegMethod.POISSON_HMM
        ):
            return chrom_data[sample].values.reshape(-1, 1), pos

    def _distance_scaled_transmat(self, A_base, delta, tau):
        f = 1 - np.exp(-delta / tau)
        A = (1 - f) * np.eye(A_base.shape[0]) + f * A_base
        return A

    def _viterbi_with_distance(self, model, X, positions, tau=1000):
        n_samples, n_states = len(X), model.n_components
        logprob = np.full((n_samples, n_states), -np.inf)
        backpointer = np.zeros((n_samples, n_states), dtype=int)

        framelogprob = model._compute_log_likelihood(X)

        # init
        logprob[0] = np.log(model.startprob_ + 0.001) + framelogprob[0]

        # recursion
        for t in range(1, n_samples):
            delta = positions[t] - positions[t - 1]
            A = self._distance_scaled_transmat(model.transmat_, delta, tau)
            for j in range(n_states):
                trans_scores = logprob[t - 1] + np.log(A[:, j])
                backpointer[t, j] = np.argmax(trans_scores)
                logprob[t, j] = np.max(trans_scores) + framelogprob[t, j]

        # traceback
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(logprob[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def _fit_dt_hmm(
        self, model, chrom, sample, fit_args: dict | None = None, tau: int = 1000
    ):
        if fit_args is None:
            fit_args = {}

        # Get both data (meth values) and positions
        data, positions = self._prep_fit_data(sample, chrom)

        # Stage 1: vanilla hmmlearn fit (learn emissions + base transitions)
        model.fit(data, **fit_args)

        # Stage 2: adjust transitions based on genomic distances
        A_base = model.transmat_
        dist = np.diff(positions)
        transmats = [self._distance_scaled_transmat(A_base, d, tau) for d in dist]

        # Replace transition matrix with an averaged distance-aware version
        model.transmat_ = np.mean(transmats, axis=0)

        return model

    def _predict_dt_hmm(self, model, chrom, sample, tau: int = 1000):
        # Get both data and positions
        data, positions = self.__prep_predict_data(sample, chrom)

        # Use custom Viterbi with distance-scaled transitions
        return self._viterbi_with_distance(model, data, positions, tau)

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
            for i in range(self.n_components):
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
