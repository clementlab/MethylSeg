import warnings

from typing import List, Optional

import cthmm

import numpy as np

from hmmlearn import hmm

warnings.filterwarnings(
    "ignore", message="divide by zero encountered in log", module="cthmm"
)


class MethylSegHMM:
    def __init__(self, n_states: int):
        raise NotImplementedError("MethylSegHMM is an abstract class")

    def fit(self, emissions, sample_info=None, chrom=None):
        raise NotImplementedError("fit method not implemented")

    def create_model(self):
        raise NotImplementedError("fit method not implemented")

    def predict(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_fit(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_predict(self, emissions):
        raise NotImplementedError("fit method not implemented")


# TODO implement and test custom DAHMM to test speed and performance against CTHMM
class DAMethylSegHMM(MethylSegHMM):
    def __init__(self, n_states: int):
        raise NotImplementedError("DAMethylSegHMM is not implemented")

    def fit(self, emissions, sample_info=None, chrom=None):
        raise NotImplementedError("fit method not implemented")

    def create_model(self):
        raise NotImplementedError("fit method not implemented")

    def predict(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_fit(self, emissions):
        raise NotImplementedError("fit method not implemented")

    def format_predict(self, emissions):
        raise NotImplementedError("fit method not implemented")


class MultinomialSegHMM(MethylSegHMM):

    def __init__(
        self,
        n_states: int,
        random_state: int = 42,
        n_iter: int = 30,
        alpha: float = 0.7,
    ):
        self.random_state = random_state
        self.n_states = n_states
        self.n_iter = n_iter
        self.alpha = alpha
        self.lengths = None

    def create_model(self):
        self.hmm_model = hmm.MultinomialHMM(
            n_components=self.n_states,
            n_iter=self.n_iter,  # EM iterations for transitions
            random_state=self.random_state,
        )

    def format_fit(self, emissions):
        return np.eye(self.n_states, dtype=int)[emissions]

    def format_predict(self, emissions):
        return self.format_fit(emissions)

    def fit(self, emissions, sample_info=None, chrom=None):
        fit_emissions = self.format_fit(emissions)
        if self.lengths is None:
            self.hmm_model.fit(fit_emissions)
        else:
            self.hmm_model.fit(fit_emissions, lengths=self.lengths)

    def predict(self, emissions):
        predict_emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(predict_emissions)
        return self.hmm_model.predict(predict_emissions, lengths=self.lengths)


class StickyCategoricalMethylSegHMM(MethylSegHMM):
    DEFAULT_STAY_PROB = 0.99995
    EMISSION_MISMATCH_PROB = 0.45
    TRANSITION_PRIOR_STRENGTH = 50.0

    def __init__(
        self,
        n_states: int,
        random_state: int = 42,
        n_iter: int = 30,
        stay_prob: float = DEFAULT_STAY_PROB,
        emission_mismatch_prob: float = EMISSION_MISMATCH_PROB,
        transition_prior_strength: float = TRANSITION_PRIOR_STRENGTH,
        fit_transitions: bool = False,
    ):
        if not np.isfinite(stay_prob) or not 0 <= stay_prob <= 1:
            raise ValueError(
                f"stay_prob must be a finite value strictly between 0 and 1. "
                f"Received: {stay_prob!r}"
            )
        if (
            not np.isfinite(emission_mismatch_prob)
            or emission_mismatch_prob <= 0
            or emission_mismatch_prob >= 1
        ):
            raise ValueError(
                "emission_mismatch_prob must be a finite value in the "
                f"range [0, 1). Received: {emission_mismatch_prob!r}"
            )
        if not np.isfinite(transition_prior_strength) or transition_prior_strength < 0:
            raise ValueError(
                "transition_prior_strength must be a finite non-negative "
                f"value. Received: {transition_prior_strength!r}"
            )
        self.random_state = random_state
        self.n_states = n_states
        self.n_iter = n_iter
        self.stay_prob = float(stay_prob)
        self.emission_mismatch_prob = float(emission_mismatch_prob)
        self.transition_prior_strength = float(transition_prior_strength)
        self.fit_transitions = fit_transitions
        self.lengths = None

    def format_fit(self, emissions):
        emissions = np.asarray(emissions, dtype=int)
        return emissions.reshape(-1, 1)

    def format_predict(self, emissions):
        return self.format_fit(emissions)

    def make_sticky_transmat(
        self,
        n_states: Optional[int] = None,
        stay_prob: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build a 'sticky' transition matrix.

        stay_prob controls the diagonal self-transition probability for every
        state. Remaining mass is shared uniformly across off-diagonal entries.
        """
        if n_states is None:
            n_states = self.n_states
        if stay_prob is None:
            stay_prob = self.stay_prob
        if n_states < 1:
            raise ValueError("n_states must be positive.")
        if n_states == 1:
            return np.ones((1, 1), dtype=float)

        switch_prob = (1.0 - stay_prob) / (n_states - 1)
        trans = np.full((n_states, n_states), switch_prob, dtype=float)
        np.fill_diagonal(trans, stay_prob)
        return trans

    def make_emissionprob(self) -> np.ndarray:
        if self.n_states == 1:
            return np.ones((1, 1), dtype=float)

        eps = self.emission_mismatch_prob
        emission = np.full(
            (self.n_states, self.n_states),
            eps / (self.n_states - 1),
            dtype=float,
        )
        np.fill_diagonal(emission, 1.0 - eps)
        return emission

    def create_model(self):
        self.prior_trans = self.make_sticky_transmat()
        self.hmm_model = hmm.CategoricalHMM(
            n_components=self.n_states,
            n_features=self.n_states,
            n_iter=self.n_iter,
            algorithm="viterbi",
            init_params="",
            params="t" if self.fit_transitions else "",
            random_state=self.random_state,
            transmat_prior=(1.0 + self.transition_prior_strength * self.prior_trans),
        )

        self.hmm_model.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.hmm_model.emissionprob_ = self.make_emissionprob()
        self.hmm_model.transmat_ = self.prior_trans.copy()

    def fit(self, emissions, sample_info=None, chrom=None):
        if self.fit_transitions:
            fit_emissions = self.format_fit(emissions)
            if self.lengths is None:
                self.hmm_model.fit(fit_emissions)
            else:
                self.hmm_model.fit(fit_emissions, lengths=self.lengths)

    def predict(self, emissions):
        predict_emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(predict_emissions)
        return self.hmm_model.predict(predict_emissions, lengths=self.lengths)


class GaussianMethylSegHMM(MethylSegHMM):
    TRANSITION_FLOOR = 1e-6
    COVARIANCE_FLOOR = 1e-3

    def __init__(
        self,
        n_states: int,
        random_state: int = 42,
        n_iter: int = 100,
        tol: float = 1e-3,
        covariance_type: str = "diag",
        init_params: str = "",
        params: str = "stmc",
    ):
        if covariance_type != "diag":
            raise ValueError(
                "GaussianMethylSegHMM currently supports only covariance_type='diag'."
            )
        self.random_state = random_state
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.params = params
        self.lengths = None

    def create_model(self):
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params=self.init_params,
            params=self.params,
        )

    def format_fit(self, emissions):
        emissions = np.asarray(emissions, dtype=np.float64)
        if emissions.ndim != 2:
            raise ValueError(
                "GaussianMethylSegHMM expects a 2D emission matrix shaped "
                "(n_observations, n_features)."
            )
        return emissions

    def format_predict(self, emissions):
        return self.format_fit(emissions)

    def _get_sequence_start_indices(
        self,
        n_observations: int,
        lengths: Optional[List[int]] = None,
    ) -> np.ndarray:
        if lengths is None:
            return np.array([0], dtype=int)

        if sum(lengths) != n_observations:
            raise ValueError(
                "The provided sequence lengths do not sum to the number of "
                "emission rows."
            )

        starts = [0]
        offset = 0
        for seq_len in lengths[:-1]:
            offset += int(seq_len)
            starts.append(offset)
        return np.asarray(starts, dtype=int)

    def _get_valid_transition_mask(
        self,
        n_observations: int,
        lengths: Optional[List[int]] = None,
    ) -> np.ndarray:
        if n_observations < 2:
            return np.zeros(0, dtype=bool)

        valid_mask = np.ones(n_observations - 1, dtype=bool)
        if lengths is None:
            return valid_mask

        boundary = 0
        for seq_len in lengths[:-1]:
            boundary += int(seq_len)
            valid_mask[boundary - 1] = False
        return valid_mask

    def initialize_from_kmeans(
        self,
        X_scaled: np.ndarray,
        km_labels: np.ndarray,
        lengths: Optional[List[int]] = None,
    ):
        X_scaled = self.format_fit(X_scaled)
        km_labels = np.asarray(km_labels, dtype=int)
        if len(X_scaled) != len(km_labels):
            raise ValueError("X_scaled and km_labels must have the same length.")
        if len(X_scaled) == 0:
            raise ValueError("Cannot initialize a Gaussian HMM on empty emissions.")
        if km_labels.min() < 0 or km_labels.max() >= self.n_states:
            raise ValueError(
                "KMeans initialization labels must be in the range "
                f"[0, {self.n_states - 1}] for the configured Gaussian HMM."
            )

        self.lengths = None if lengths is None else [int(length) for length in lengths]

        start_indices = self._get_sequence_start_indices(
            n_observations=len(km_labels),
            lengths=self.lengths,
        )
        startprob = np.bincount(
            km_labels[start_indices],
            minlength=self.n_states,
        ).astype(np.float64)
        startprob_sum = startprob.sum()
        if startprob_sum == 0:
            startprob = np.full(self.n_states, 1.0 / self.n_states, dtype=np.float64)
        else:
            startprob /= startprob_sum

        transmat = np.full(
            (self.n_states, self.n_states),
            self.TRANSITION_FLOOR,
            dtype=np.float64,
        )
        valid_transitions = self._get_valid_transition_mask(
            n_observations=len(km_labels),
            lengths=self.lengths,
        )
        for start_state, end_state in zip(
            km_labels[:-1][valid_transitions],
            km_labels[1:][valid_transitions],
        ):
            transmat[int(start_state), int(end_state)] += 1.0
        transmat /= transmat.sum(axis=1, keepdims=True)

        global_mean = np.mean(X_scaled, axis=0)
        global_var = np.var(X_scaled, axis=0) + self.COVARIANCE_FLOOR

        means = np.zeros((self.n_states, X_scaled.shape[1]), dtype=np.float64)
        covars = np.zeros((self.n_states, X_scaled.shape[1]), dtype=np.float64)
        for state_idx in range(self.n_states):
            members = X_scaled[km_labels == state_idx]
            if len(members) == 0:
                means[state_idx] = global_mean
                covars[state_idx] = global_var
                continue
            means[state_idx] = np.mean(members, axis=0)
            covars[state_idx] = np.var(members, axis=0) + self.COVARIANCE_FLOOR

        self.hmm_model.startprob_ = startprob
        self.hmm_model.transmat_ = transmat
        self.hmm_model.means_ = means
        self.hmm_model.covars_ = covars

    def fit(self, emissions, sample_info=None, chrom=None):
        emissions = self.format_fit(emissions)
        if self.lengths is None:
            self.hmm_model.fit(emissions)
        else:
            self.hmm_model.fit(emissions, lengths=self.lengths)

    def predict(self, emissions):
        emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(emissions)
        return self.hmm_model.predict(emissions, lengths=self.lengths)


class CTMethylSegHMM(MethylSegHMM):

    def __init__(
        self,
        n_states,
        n_emissions: int = 4,
        holding_time_guess: int = 1_500_000,
        time_scale: float = 1,
        max_iter: int = 25,
        tol: float = 1e-2,
        random_state: int = 42,
        algorithm="forward-backward",
    ):
        self.random_state = random_state
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.holding_time_guess = holding_time_guess
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.time_scale = time_scale

    def create_model(self):
        eps = 0.02  # mislabel rate

        emission_probs = np.full(
            (self.n_states, self.n_emissions), eps / (self.n_emissions - 1)
        )
        np.fill_diagonal(emission_probs, 1.0 - eps)
        self.hmm_model = cthmm.MultinomialCTHMM(
            n_states=self.n_states,
            n_emissions=self.n_emissions,
            emission_probs=emission_probs,  # our near-identity matrix
            holding_time=self.holding_time_guess,  # library builds a default Q from this
            seed=self.random_state,
        )

    def format_fit(self, emissions):
        obs_states = emissions
        times = self.times
        # print(len(obs_states), len(times))
        return [(obs_states, times)]

    def format_predict(self, emissions):
        obs_states = emissions
        times = self.times
        return (obs_states, times)

    def fit(self, emissions, sample_info, chrom):
        self.times = (
            sample_info.meth_data[sample_info.meth_data["CpG_chrm"] == chrom][
                "CpG_beg"
            ].values
            / self.time_scale
        )
        return self.hmm_model.fit(
            self.format_fit(emissions),
            verbose=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def predict(self, emissions):
        return self.hmm_model.predict(
            *self.format_predict(emissions), algorithm=self.algorithm
        )
