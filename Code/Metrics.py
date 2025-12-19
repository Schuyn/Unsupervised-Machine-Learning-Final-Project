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
