# Metrics.py
# ------------------------------------------------------------------------------
# This module contains evaluation metrics for the Gaussian style-topic model.
# It is intentionally separated from Train.py and Model.py so that:
#   - Training code (Train.py) focuses only on data loading, preprocessing,
#     training, and hyperparameter tuning.
#   - Model code (Model.py) defines only the generative model.
#   - Metrics code (this file) provides a clean interface for evaluating
#     reconstruction error, ELBO-based criteria, and—importantly—stability
#     measures used in unsupervised learning (e.g., STAT 5244).
#
# The goal is to make it straightforward to:
#   * add new metrics (stability, bootstrap, PPP, held-out loglik, etc.)
#   * keep CV results clean and extensible
#   * maintain a modular and professional codebase
# ------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np


# ------------------------------------------------------------------------------
# Basic reconstruction metric
# ------------------------------------------------------------------------------

def compute_reconstruction_mse(
    X: np.ndarray,
    theta: np.ndarray,
    beta: np.ndarray,
) -> float:
    """
    Compute reconstruction Mean Squared Error (MSE) between
    the observed standardized feature matrix X and its
    model-implied reconstruction mu = theta @ beta.

    Parameters
    ----------
    X : np.ndarray, shape (N, P)
        Standardized feature matrix.
    theta : np.ndarray, shape (N, K)
        Player-style membership weights.
    beta : np.ndarray, shape (K, P)
        Style loading matrix.

    Returns
    -------
    mse : float
        The mean squared reconstruction error across all (n, p).
    """
    mu = theta @ beta
    diff = X - mu
    mse = float((diff ** 2).mean())
    return mse


# ------------------------------------------------------------------------------
# Placeholder metrics for future work (ELBO, stability, etc.)
# ------------------------------------------------------------------------------

def compute_elbo_placeholder(
    history: Optional[List[float]] = None,
) -> Optional[float]:
    """
    Placeholder for ELBO-based metrics.

    In principle, one can:
      - Track the ELBO during training.
      - Use the final ELBO or an average over the last N iterations
        as a model selection criterion.

    Currently this function returns:
      - None if no history is provided.
      - The final ELBO value if history is provided.

    Parameters
    ----------
    history : list of float, optional
        Sequence of ELBO (negative loss) values during training.

    Returns
    -------
    elbo_value : float or None
        Summary statistic of ELBO. Currently only returns the last value.
    """
    if history is None or len(history) == 0:
        return None
    return float(history[-1])


def compute_stability_beta_placeholder(
    betas: List[np.ndarray],
) -> Optional[float]:
    """
    Placeholder for style-loading (beta) stability estimates.

    Motivation:
      In unsupervised learning (e.g., STAT 5244),
      model stability is a key criterion. A robust model should yield
      similar 'beta' (style vectors) across resampled subsets of data.

    Future implementation may:
      - Fit the model multiple times on bootstrap subsamples.
      - Align style components across runs (Hungarian matching based on
        cosine similarity or correlation).
      - Compute an average similarity score across aligned beta matrices.

    Parameters
    ----------
    betas : list of np.ndarray
        A list of beta matrices (each of shape (K, P)) from different runs.

    Returns
    -------
    stability_score : float or None
        A scalar stability score. Placeholder returns None.
    """
    if len(betas) == 0:
        return None
    # TODO: implement alignment + similarity computation (cosine, correlation, etc.)
    return None


def compute_stability_theta_placeholder(
    thetas: List[np.ndarray],
) -> Optional[float]:
    """
    Placeholder for theta-based stability metrics.

    Motivation:
      - Each run of the model yields an N×K matrix of player-style weights.
      - One can cluster players based on theta and evaluate stability across runs.
      - Typical metrics: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).

    Future implementation may:
      - Cluster each theta matrix (e.g., k-means on rows).
      - Use NMI/ARI to compare cluster assignments across runs.
      - Aggregate these scores to form a stability metric.

    Parameters
    ----------
    thetas : list of np.ndarray
        A list of theta matrices (shape (N, K)) from different runs or subsamples.

    Returns
    -------
    stability_score : float or None
        A placeholder value. Returns None for now.
    """
    if len(thetas) == 0:
        return None
    # TODO: implement clustering + ARI/NMI stability in future work.
    return None


# ------------------------------------------------------------------------------
# Dataclass for cross-validation results
# ------------------------------------------------------------------------------

@dataclass
class CVResult:
    """
    Container for summarizing cross-validation metrics for one
    hyperparameter configuration.

    Attributes
    ----------
    config : Any
        The hyperparameter configuration (e.g., StyleModelConfig).
    mean_mse : float
        Mean reconstruction MSE across CV folds.
    std_mse : float
        Standard deviation of reconstruction MSE across CV folds.
    extra_metrics : dict
        A flexible dictionary for additional metrics, such as:
            {
               "elbo": <float or None>,
               "beta_stability": <float or None>,
               "theta_stability": <float or None>,
               ...
            }
        This allows clean extension of the CV pipeline as more
        unsupervised criteria are added.
    """
    config: Any
    mean_mse: float
    std_mse: float
    extra_metrics: Dict[str, float] = None

    def __post_init__(self):
        # Initialize dictionary if not provided
        if self.extra_metrics is None:
            self.extra_metrics = {}


# ------------------------------------------------------------------------------
# Posterior Predictive Checking (PPC) + Posterior Predictive Nulls (PPN)
# Pure NumPy implementation (safe to import anywhere)
# ------------------------------------------------------------------------------

from typing import Tuple

def _as_np(x):
    return np.asarray(x)

def predict_mu(theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """mu = theta @ beta, shapes: (N,K)@(K,P)->(N,P)"""
    theta = _as_np(theta)
    beta = _as_np(beta)
    return theta @ beta

def replicate_gaussian(mu: np.ndarray, sigma: np.ndarray, n_rep: int = 200, seed: int = 0) -> np.ndarray:
    """
    X_rep ~ Normal(mu, sigma), with featurewise sigma (P,)
    Returns: (n_rep, N, P)
    """
    mu = _as_np(mu)
    sigma = _as_np(sigma)
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma[None, :], size=(n_rep, mu.shape[0], mu.shape[1]))
    return mu[None, :, :] + eps

# --------------------------
# Discrepancy statistics T(X)
# --------------------------

def T_feature_moments(X: np.ndarray) -> dict:
    X = _as_np(X)
    return {
        "mean": X.mean(axis=0),
        "std":  X.std(axis=0),
        "q05":  np.quantile(X, 0.05, axis=0),
        "q50":  np.quantile(X, 0.50, axis=0),
        "q95":  np.quantile(X, 0.95, axis=0),
    }

def T_feature_mse(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    X = _as_np(X); mu = _as_np(mu)
    return ((X - mu) ** 2).mean(axis=0)

def T_corr_upper(X: np.ndarray) -> np.ndarray:
    """
    Flattened upper-triangular of correlation matrix (excluding diagonal).
    Useful for checking cross-feature structure.
    """
    X = _as_np(X)
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]

def T_unique_ratio(X: np.ndarray, decimals: int = 2) -> np.ndarray:
    """
    Fraction of unique values (after rounding) per feature.
    Detects discretization/banding effects.
    """
    Xr = np.round(_as_np(X), decimals=decimals)
    return np.array([len(np.unique(Xr[:, p])) / Xr.shape[0] for p in range(Xr.shape[1])])

def ppc_tail_prob(T_obs: np.ndarray, T_rep: np.ndarray) -> np.ndarray:
    """
    Elementwise tail prob: P(T_rep >= T_obs). Shapes:
    - T_obs: (...)
    - T_rep: (n_rep, ...)
    """
    T_obs = _as_np(T_obs)
    T_rep = _as_np(T_rep)
    return (T_rep >= T_obs[None, ...]).mean(axis=0)

# --------------------------
# Null generators (PPN)
# --------------------------

def null_independent_standard_normal(N: int, P: int, n_rep: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n_rep, N, P))

def null_column_permutation(X: np.ndarray, n_rep: int = 200, seed: int = 0) -> np.ndarray:
    """
    Preserve each feature's marginal distribution, destroy joint structure.
    """
    rng = np.random.default_rng(seed)
    X = _as_np(X)
    reps = np.empty((n_rep, X.shape[0], X.shape[1]), dtype=X.dtype)
    for r in range(n_rep):
        Xp = X.copy()
        for p in range(X.shape[1]):
            rng.shuffle(Xp[:, p])
        reps[r] = Xp
    return reps

# --------------------------
# Main suite runner
# --------------------------

def run_ppc_suite(
    X_obs: np.ndarray,
    theta: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    n_rep: int = 200,
    seed: int = 0,
    compute_unique: bool = True,
    unique_decimals: int = 2,
) -> Dict[str, Any]:
    """
    PPC using plug-in posterior means:
      mu = theta @ beta
      X_rep = mu + N(0, sigma)
    Returns a structured report with:
      - observed discrepancies
      - replicated discrepancies (arrays)
      - posterior predictive tail probabilities (p-values)
    """
    X_obs = _as_np(X_obs)
    mu = predict_mu(theta, beta)
    X_rep = replicate_gaussian(mu, sigma, n_rep=n_rep, seed=seed)

    # moments
    obs_mom = T_feature_moments(X_obs)
    rep_mom = {k: np.stack([T_feature_moments(X_rep[r])[k] for r in range(n_rep)], axis=0)
               for k in obs_mom.keys()}
    p_mom = {k: ppc_tail_prob(obs_mom[k], rep_mom[k]) for k in obs_mom.keys()}

    # correlation upper triangle
    obs_corr = T_corr_upper(X_obs)
    rep_corr = np.stack([T_corr_upper(X_rep[r]) for r in range(n_rep)], axis=0)
    p_corr = ppc_tail_prob(obs_corr, rep_corr)

    # mse (feature-wise)
    obs_mse = T_feature_mse(X_obs, mu)
    rep_mse = np.stack([T_feature_mse(X_rep[r], mu) for r in range(n_rep)], axis=0)
    p_mse = ppc_tail_prob(obs_mse, rep_mse)

    out = {
        "mu": mu,
        "obs": {
            "moments": obs_mom,
            "corr_upper": obs_corr,
            "mse_feat": obs_mse,
        },
        "rep": {
            "moments": rep_mom,
            "corr_upper": rep_corr,
            "mse_feat": rep_mse,
        },
        "p": {
            "moments": p_mom,
            "corr_upper": p_corr,
            "mse_feat": p_mse,
        },
        "meta": {
            "n_rep": n_rep,
            "seed": seed,
            "N": int(X_obs.shape[0]),
            "P": int(X_obs.shape[1]),
            "K": int(_as_np(beta).shape[0]),
        },
    }

    if compute_unique:
        obs_u = T_unique_ratio(X_obs, decimals=unique_decimals)
        rep_u = np.stack([T_unique_ratio(X_rep[r], decimals=unique_decimals) for r in range(n_rep)], axis=0)
        out["obs"]["unique_ratio"] = obs_u
        out["rep"]["unique_ratio"] = rep_u
        out["p"]["unique_ratio"] = ppc_tail_prob(obs_u, rep_u)
        out["meta"]["unique_decimals"] = unique_decimals

    return out

def run_null_suite(
    X_obs: np.ndarray,
    null_reps: np.ndarray,
    unique_decimals: int = 2,
) -> Dict[str, Any]:
    """
    Evaluate discrepancy statistics of observed X under a null replication set.
    null_reps: (n_rep, N, P)
    Returns p-values for moments/corr/unique (mse not defined without mu).
    """
    X_obs = _as_np(X_obs)
    null_reps = _as_np(null_reps)
    n_rep = null_reps.shape[0]

    obs_mom = T_feature_moments(X_obs)
    rep_mom = {k: np.stack([T_feature_moments(null_reps[r])[k] for r in range(n_rep)], axis=0)
               for k in obs_mom.keys()}
    p_mom = {k: ppc_tail_prob(obs_mom[k], rep_mom[k]) for k in obs_mom.keys()}

    obs_corr = T_corr_upper(X_obs)
    rep_corr = np.stack([T_corr_upper(null_reps[r]) for r in range(n_rep)], axis=0)
    p_corr = ppc_tail_prob(obs_corr, rep_corr)

    obs_u = T_unique_ratio(X_obs, decimals=unique_decimals)
    rep_u = np.stack([T_unique_ratio(null_reps[r], decimals=unique_decimals) for r in range(n_rep)], axis=0)
    p_u = ppc_tail_prob(obs_u, rep_u)

    return {
        "obs": {"moments": obs_mom, "corr_upper": obs_corr, "unique_ratio": obs_u},
        "rep": {"moments": rep_mom, "corr_upper": rep_corr, "unique_ratio": rep_u},
        "p":   {"moments": p_mom, "corr_upper": p_corr, "unique_ratio": p_u},
        "meta": {"n_rep": int(n_rep), "N": int(X_obs.shape[0]), "P": int(X_obs.shape[1]), "unique_decimals": unique_decimals},
    }
