"""Hidden Markov Model backends used by the methylseg segmentor."""

import warnings

from typing import List, Optional

import cthmm

import numpy as np

from hmmlearn import hmm

warnings.filterwarnings(
    "ignore", message="divide by zero encountered in log", module="cthmm"
)


class MethylSegHMM:
    """Abstract base class for methylseg HMM backends."""

    def __init__(self, n_states: int):
        """
        Initialize an abstract HMM backend.

        Parameters
        ----------
        n_states
            Number of hidden methylation states the backend will model.
        """
        raise NotImplementedError("MethylSegHMM is an abstract class")

    def fit(self, emissions, sample_info=None, chrom=None):
        """
        Fit backend-specific HMM parameters on the provided emissions.

        Parameters
        ----------
        emissions
            Observation sequence or feature matrix already prepared for the
            backend.
        sample_info
            Optional sample metadata used by backends that require genomic
            coordinates or other sample-level context.
        chrom
            Chromosome label for single-chromosome fitting when relevant.
        """
        raise NotImplementedError("fit method not implemented")

    def create_model(self):
        """Instantiate and initialize the backend-specific HMM object."""
        raise NotImplementedError("fit method not implemented")

    def predict(self, emissions):
        """
        Decode hidden states for the supplied emissions.

        Parameters
        ----------
        emissions
            Observation sequence or feature matrix prepared for prediction.

        Returns
        -------
        numpy.ndarray
            Hidden-state assignments in backend-specific numeric form.
        """
        raise NotImplementedError("fit method not implemented")

    def format_fit(self, emissions):
        """
        Convert raw emissions into the representation expected by ``fit``.

        Parameters
        ----------
        emissions
            Raw observation labels or features.
        """
        raise NotImplementedError("fit method not implemented")

    def format_predict(self, emissions):
        """
        Convert raw emissions into the representation expected by ``predict``.

        Parameters
        ----------
        emissions
            Raw observation labels or features.
        """
        raise NotImplementedError("fit method not implemented")


# TODO implement and test custom DAHMM to test speed and performance against CTHMM
class DAMethylSegHMM(MethylSegHMM):
    """Placeholder for an alternative HMM implementation that is not yet built."""

    def __init__(self, n_states: int):
        """
        Initialize the unimplemented DA HMM placeholder.

        Parameters
        ----------
        n_states
            Number of hidden methylation states the backend would model.
        """
        raise NotImplementedError("DAMethylSegHMM is not implemented")

    def fit(self, emissions, sample_info=None, chrom=None):
        """
        Placeholder method for the not-yet-implemented DA backend.

        Parameters
        ----------
        emissions
            Observation sequence or feature matrix prepared for the DA backend.
        sample_info
            Optional sample metadata for API compatibility.
        chrom
            Optional chromosome label for API compatibility.
        """
        raise NotImplementedError("fit method not implemented")

    def create_model(self):
        """Placeholder method for the not-yet-implemented DA backend."""
        raise NotImplementedError("fit method not implemented")

    def predict(self, emissions):
        """
        Placeholder method for the not-yet-implemented DA backend.

        Parameters
        ----------
        emissions
            Observation sequence or feature matrix prepared for prediction.

        Returns
        -------
        numpy.ndarray
            Placeholder hidden-state assignments once this backend is
            implemented.
        """
        raise NotImplementedError("fit method not implemented")

    def format_fit(self, emissions):
        """
        Placeholder method for the not-yet-implemented DA backend.

        Parameters
        ----------
        emissions
            Raw observation labels or features.
        """
        raise NotImplementedError("fit method not implemented")

    def format_predict(self, emissions):
        """
        Placeholder method for the not-yet-implemented DA backend.

        Parameters
        ----------
        emissions
            Raw observation labels or features.
        """
        raise NotImplementedError("fit method not implemented")


class MultinomialSegHMM(MethylSegHMM):
    """Discrete multinomial HMM over categorical methylation state labels."""

    def __init__(
        self,
        n_states: int,
        random_state: int = 42,
        n_iter: int = 30,
        alpha: float = 0.7,
    ):
        """
        Initialize a multinomial HMM for discrete state observations.

        Parameters
        ----------
        n_states
            Number of hidden and observed categorical states.
        random_state
            Random seed passed to the underlying HMM implementation.
        n_iter
            Maximum number of expectation-maximization iterations.
        alpha
            Compatibility smoothing parameter retained for this backend.
        """
        self.random_state = random_state
        self.n_states = n_states
        self.n_iter = n_iter
        self.alpha = alpha
        self.lengths = None

    def create_model(self):
        """Create the underlying ``hmmlearn.MultinomialHMM`` instance."""
        self.hmm_model = hmm.MultinomialHMM(
            n_components=self.n_states,
            n_iter=self.n_iter,  # EM iterations for transitions
            random_state=self.random_state,
        )

    def format_fit(self, emissions):
        """
        One-hot encode discrete state labels for multinomial fitting.

        Parameters
        ----------
        emissions
            Integer state labels in ``[0, n_states)``.

        Returns
        -------
        numpy.ndarray
            One-hot encoded matrix with shape ``(n_observations, n_states)``.
        """
        return np.eye(self.n_states, dtype=int)[emissions]

    def format_predict(self, emissions):
        """
        Format discrete labels for prediction using the fit representation.

        Parameters
        ----------
        emissions
            Integer state labels in ``[0, n_states)``.

        Returns
        -------
        numpy.ndarray
            One-hot encoded matrix with shape ``(n_observations, n_states)``.
        """
        return self.format_fit(emissions)

    def fit(self, emissions, sample_info=None, chrom=None):
        """
        Fit the multinomial HMM on discrete state observations.

        Parameters
        ----------
        emissions
            Integer state labels to smooth with the HMM.
        sample_info
            Unused placeholder for API compatibility.
        chrom
            Unused placeholder for API compatibility.
        """
        fit_emissions = self.format_fit(emissions)
        if self.lengths is None:
            self.hmm_model.fit(fit_emissions)
        else:
            self.hmm_model.fit(fit_emissions, lengths=self.lengths)

    def predict(self, emissions):
        """
        Predict hidden states for discrete observations.

        Parameters
        ----------
        emissions
            Integer state labels to decode.

        Returns
        -------
        numpy.ndarray
            Predicted HMM state sequence.
        """
        predict_emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(predict_emissions)
        return self.hmm_model.predict(predict_emissions, lengths=self.lengths)


class StickyCategoricalMethylSegHMM(MethylSegHMM):
    """Categorical HMM with strong self-transition priors for smoother segments."""

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
        """
        Initialize a sticky categorical HMM.

        Parameters
        ----------
        n_states
            Number of hidden and observed categorical states.
        random_state
            Random seed passed to the underlying HMM implementation.
        n_iter
            Maximum number of expectation-maximization iterations.
        stay_prob
            Initial self-transition probability for each hidden state.
        emission_mismatch_prob
            Probability mass assigned to nonmatching observed states.
        transition_prior_strength
            Weight of the sticky transition prior when transitions are fitted.
        fit_transitions
            If ``True``, estimate transitions during fitting; otherwise retain
            the initialized sticky transition matrix.
        """
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
        """
        Reshape integer categorical observations for ``CategoricalHMM``.

        Parameters
        ----------
        emissions
            Integer observation labels in ``[0, n_states)``.

        Returns
        -------
        numpy.ndarray
            Column vector with one categorical code per observation.
        """
        emissions = np.asarray(emissions, dtype=int)
        return emissions.reshape(-1, 1)

    def format_predict(self, emissions):
        """
        Format categorical observations for prediction.

        Parameters
        ----------
        emissions
            Integer observation labels in ``[0, n_states)``.

        Returns
        -------
        numpy.ndarray
            Column vector with one categorical code per observation.
        """
        return self.format_fit(emissions)

    def make_sticky_transmat(
        self,
        n_states: Optional[int] = None,
        stay_prob: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build a 'sticky' transition matrix.

        Parameters
        ----------
        n_states
            Optional number of hidden states. When omitted, uses
            ``self.n_states``.
        stay_prob
            Optional diagonal self-transition probability. When omitted, uses
            ``self.stay_prob``.

        Returns
        -------
        numpy.ndarray
            Square transition matrix whose rows sum to one.

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
        """
        Build the near-identity emission matrix used by the sticky HMM.

        Returns
        -------
        numpy.ndarray
            Square emission-probability matrix whose diagonal retains most of
            the probability mass.
        """
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
        """Create and initialize the sticky categorical HMM backend."""
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
        """
        Fit transition probabilities when ``fit_transitions`` is enabled.

        Parameters
        ----------
        emissions
            Integer categorical observations.
        sample_info
            Unused placeholder for API compatibility.
        chrom
            Unused placeholder for API compatibility.
        """
        if self.fit_transitions:
            fit_emissions = self.format_fit(emissions)
            if self.lengths is None:
                self.hmm_model.fit(fit_emissions)
            else:
                self.hmm_model.fit(fit_emissions, lengths=self.lengths)

    def predict(self, emissions):
        """
        Predict smoothed hidden states from categorical observations.

        Parameters
        ----------
        emissions
            Integer categorical observations.

        Returns
        -------
        numpy.ndarray
            Decoded HMM state sequence.
        """
        predict_emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(predict_emissions)
        return self.hmm_model.predict(predict_emissions, lengths=self.lengths)


class GaussianMethylSegHMM(MethylSegHMM):
    """Gaussian HMM for continuous emission vectors such as PCA features."""

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
        """
        Initialize a Gaussian HMM for continuous emission features.

        Parameters
        ----------
        n_states
            Number of hidden methylation states.
        random_state
            Random seed passed to the underlying HMM implementation.
        n_iter
            Maximum number of expectation-maximization iterations.
        tol
            Convergence tolerance for expectation-maximization.
        covariance_type
            Covariance parameterization; currently only ``"diag"`` is
            supported.
        init_params
            ``hmmlearn`` parameter codes to initialize automatically.
        params
            ``hmmlearn`` parameter codes to update during fitting.
        """
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
        """Create the underlying ``hmmlearn.GaussianHMM`` instance."""
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
        """
        Validate and coerce continuous emissions into a 2-D float matrix.

        Parameters
        ----------
        emissions
            Continuous emission features with shape
            ``(n_observations, n_features)``.

        Returns
        -------
        numpy.ndarray
            Float64 feature matrix ready for Gaussian HMM fitting.
        """
        emissions = np.asarray(emissions, dtype=np.float64)
        if emissions.ndim != 2:
            raise ValueError(
                "GaussianMethylSegHMM expects a 2D emission matrix shaped "
                "(n_observations, n_features)."
            )
        return emissions

    def format_predict(self, emissions):
        """
        Format continuous emission features for prediction.

        Parameters
        ----------
        emissions
            Continuous emission features with shape
            ``(n_observations, n_features)``.

        Returns
        -------
        numpy.ndarray
            Float64 feature matrix ready for Gaussian HMM decoding.
        """
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
        """
        Seed Gaussian HMM parameters from clustered feature vectors.

        Parameters
        ----------
        X_scaled
            Scaled continuous emission matrix.
        km_labels
            Integer cluster assignments used to initialize means, variances, and
            transition structure.
        lengths
            Optional per-sequence lengths for multi-chromosome fitting.
        """
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
        """
        Fit the Gaussian HMM on continuous emission features.

        Parameters
        ----------
        emissions
            Continuous emission matrix.
        sample_info
            Unused placeholder for API compatibility.
        chrom
            Unused placeholder for API compatibility.
        """
        emissions = self.format_fit(emissions)
        if self.lengths is None:
            self.hmm_model.fit(emissions)
        else:
            self.hmm_model.fit(emissions, lengths=self.lengths)

    def predict(self, emissions):
        """
        Predict hidden states from continuous emission features.

        Parameters
        ----------
        emissions
            Continuous emission matrix.

        Returns
        -------
        numpy.ndarray
            Predicted HMM state sequence.
        """
        emissions = self.format_predict(emissions)
        if self.lengths is None:
            return self.hmm_model.predict(emissions)
        return self.hmm_model.predict(emissions, lengths=self.lengths)


class CTMethylSegHMM(MethylSegHMM):
    """Continuous-time HMM backend for sparsely spaced CpGs along a chromosome."""


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
        """
        Initialize a continuous-time HMM for unevenly spaced CpGs.

        Parameters
        ----------
        n_states
            Number of hidden methylation states.
        n_emissions
            Number of discrete observed emission categories.
        holding_time_guess
            Initial genomic holding-time scale in base pairs.
        time_scale
            Multiplier applied to genomic time intervals.
        max_iter
            Maximum fitting iterations for the continuous-time backend.
        tol
            Convergence tolerance for continuous-time fitting.
        random_state
            Random seed passed to the continuous-time backend.
        algorithm
            Fitting algorithm supported by the continuous-time backend.
        """
        self.random_state = random_state
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.holding_time_guess = holding_time_guess
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.time_scale = time_scale

    def create_model(self):
        """Create the continuous-time HMM with a default near-identity emission model."""
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
        """
        Pair observed states with genomic coordinates for CT-HMM fitting.

        Parameters
        ----------
        emissions
            Integer observed state sequence.

        Returns
        -------
        list
            Single-sequence list containing ``(observed_states, times)``.
        """
        obs_states = emissions
        times = self.times
        # print(len(obs_states), len(times))
        return [(obs_states, times)]

    def format_predict(self, emissions):
        """
        Pair observed states with genomic coordinates for CT-HMM decoding.

        Parameters
        ----------
        emissions
            Integer observed state sequence.

        Returns
        -------
        tuple
            ``(observed_states, times)`` for the decoder.
        """
        obs_states = emissions
        times = self.times
        return (obs_states, times)

    def fit(self, emissions, sample_info, chrom):
        """
        Fit the continuous-time HMM using CpG coordinates as observation times.

        Parameters
        ----------
        emissions
            Integer observed state sequence for one chromosome.
        sample_info
            Sample metadata whose methylation table supplies CpG genomic
            positions.
        chrom
            Chromosome whose CpGs should be used to derive observation times.

        Returns
        -------
        object
            Fitted CT-HMM object returned by ``cthmm``.
        """
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
        """
        Decode hidden states with the configured continuous-time algorithm.

        Parameters
        ----------
        emissions
            Integer observed state sequence.

        Returns
        -------
        numpy.ndarray
            Decoded hidden-state assignments.
        """
        return self.hmm_model.predict(
            *self.format_predict(emissions), algorithm=self.algorithm
        )
