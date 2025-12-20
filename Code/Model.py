# # Model.py
# # Author: Kangyu Zhao
# # Gaussian style-topic model (Dirichlet latent styles + Gaussian emissions)
# # for NBA draft rate + efficiency features.
# #
# # Requires: numpyro, jax, jaxlib

# from dataclasses import dataclass
# from typing import Optional

# import jax.numpy as jnp
# import numpyro
# import numpyro.distributions as dist


# @dataclass
# class StyleModelConfig:
#     """
#     Configuration for the Gaussian style-topic model.

#     Attributes
#     ----------
#     K : int
#         Number of latent styles (topics).
#     alpha : float
#         Dirichlet concentration for theta (player-style proportions).
#         Typically a small positive value, e.g. 0.5 or 1.0.
#     tau_beta : float
#         Scale parameter for the Normal prior on beta (style loadings).
#         beta_{k,p} ~ Normal(0, tau_beta).
#     tau_sigma : float
#         Scale parameter for the Half-Cauchy prior on sigma (feature noise).
#         sigma_p ~ HalfCauchy(0, tau_sigma).
#     """
#     K: int
#     alpha: float = 3.0
#     tau_beta: float = 1.0      # global prior scale for beta
#     tau_sigma: float = 1.0

#     # NEW: group-specific prior scale multipliers (optional)
#     tau_beta_count: float = 1.0
#     tau_beta_pct: float = 1.0
#     tau_beta_other: float = 1.0



# def style_topic_model(
#     X: jnp.ndarray,
#     K: Optional[int] = None,
#     config: Optional[StyleModelConfig] = None,
# ) -> None:
#     """
#     Gaussian style-topic model for standardized rate + efficiency features.

#     Parameters
#     ----------
#     X : jnp.ndarray, shape (N, P)
#         Standardized feature matrix (rate + efficiency).
#         Each column should be roughly zero-mean, unit-variance over the TRAIN set.
#     K : int, optional
#         Number of latent styles. If None, will be taken from `config.K`.
#     config : StyleModelConfig, optional
#         Model hyperparameters. If None, a default config with given K is used.

#     Model
#     -----
#     For each player n = 1..N and feature p = 1..P:

#       theta_n ~ Dirichlet(alpha * 1_K)            # player-style proportions
#       beta_{k,p} ~ Normal(0, tau_beta)            # style loadings for feature p
#       sigma_p ~ HalfCauchy(tau_sigma)             # feature-specific noise scale

#       X_{n,p} ~ Normal( (theta_n @ beta[:, p]), sigma_p )

#     Shapes
#     ------
#     - theta : (N, K)
#     - beta  : (K, P)
#     - sigma : (P,)
#     - X     : (N, P)
#     """
#     # Infer shapes
#     N, P = X.shape

#     if config is None and K is None:
#         raise ValueError("Either `K` or `config` with `K` must be provided.")

#     if config is None:
#         config = StyleModelConfig(K=K)
#     if K is None:
#         K = config.K

#     # ----------------------------------------------------------------------
#     # Priors
#     # ----------------------------------------------------------------------

#     # Player-style proportions theta_n on the K-dimensional simplex
#     # theta has shape (N, K)
#     alpha_vec = jnp.ones(K) * config.alpha
#     theta = numpyro.sample(
#         "theta",
#         dist.Dirichlet(alpha_vec).expand([N]).to_event(1),
#     )

#     # Style loadings beta_{k,p}, shape (K, P)
#     # beta = numpyro.sample(
#     #     "beta",
#     #     dist.Normal(0.0, config.tau_beta).expand([K, P]).to_event(2),
#     # )
#     # Group-level scales (HalfCauchy is standard for scale parameters)
#     tau_beta_count = numpyro.sample("tau_beta_count", dist.HalfCauchy(config.tau_beta * config.tau_beta_count))
#     tau_beta_pct   = numpyro.sample("tau_beta_pct",   dist.HalfCauchy(config.tau_beta * config.tau_beta_pct))
#     tau_beta_other = numpyro.sample("tau_beta_other", dist.HalfCauchy(config.tau_beta * config.tau_beta_other))

#     # IMPORTANT: feature order MUST match Train.build_design_matrix
#     # [PTS36, REB36, AST36, FG_logit, 3P_logit, FT_logit, WS/48, BPM, VORP, AvgMP]
#     tau_vec = jnp.array([
#         tau_beta_count,  # PTS per 36
#         tau_beta_count,  # REB per 36
#         tau_beta_count,  # AST per 36
#         tau_beta_pct,    # FG% (logit)
#         tau_beta_pct,    # 3P% (logit)
#         tau_beta_pct,    # FT% (logit)
#         tau_beta_other,  # WS/48
#         tau_beta_other,  # BPM
#         tau_beta_other,  # VORP
#         tau_beta_count,  # Avg minutes played
#     ])  # shape (10,)

#     # If you ever change P != 10, guard it explicitly
#     # (recommended to avoid silent mismatch)
#     if P != tau_vec.shape[0]:
#         raise ValueError(f"Expected P={tau_vec.shape[0]} features, got P={P}. Check feature construction order.")

#     beta = numpyro.sample(
#         "beta",
#         dist.Normal(0.0, tau_vec).expand([K, P]).to_event(2),
#     )


#     # Feature-specific noise scales sigma_p, shape (P,)
#     sigma = numpyro.sample(
#         "sigma",
#         dist.HalfCauchy(config.tau_sigma).expand([P]).to_event(1),
#     )

#     # ----------------------------------------------------------------------
#     # Likelihood
#     # ----------------------------------------------------------------------

#     # Mean matrix mu_{n,p} = theta_n^T beta[:, p]
#     # theta: (N, K), beta: (K, P) -> mu: (N, P)
#     mu = jnp.matmul(theta, beta)

#     # Observation model: X_{n,p} ~ Normal(mu_{n,p}, sigma_p)
#     # We treat (N, P) as independent observations under the hood via broadcasting.
#     with numpyro.plate("players", N, dim=-2):
#         with numpyro.plate("features", P, dim=-1):
#             numpyro.sample(
#                 "X",
#                 dist.Normal(mu, sigma),
#                 obs=X,
#             )


# Model.py
# Author: Kangyu Zhao
# Gaussian style-topic model (Dirichlet latent styles + Gaussian emissions)
# for NBA draft rate + efficiency features.
#
# Requires: numpyro, jax, jaxlib

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


@dataclass
class StyleModelConfig:
    """
    Configuration for the Gaussian style-topic model.

    Attributes
    ----------
    K : int
        Number of latent styles (topics).
    alpha : float
        Dirichlet concentration for theta (player-style proportions).
        Typically a small positive value, e.g. 0.5 or 1.0.
    tau_beta : float
        Scale parameter for the Normal prior on beta (style loadings).
        beta_{k,p} ~ Normal(0, tau_beta).
    tau_sigma : float
        Scale parameter for the Half-Cauchy prior on sigma (feature noise).
        sigma_p ~ HalfCauchy(0, tau_sigma).
    """
    K: int
    alpha: float = 3.0
    alpha_w: Optional[jnp.ndarray] = None  # shape (K,), breaks topic symmetry
    tau_beta: float = 1.0      # global prior scale for beta
    tau_sigma: float = 1.0

    # NEW: group-specific prior scale multipliers (optional)
    tau_beta_count: float = 1.0
    tau_beta_pct: float = 1.0
    tau_beta_other: float = 1.0


def style_topic_model(
    X: jnp.ndarray,
    K: Optional[int] = None,
    config: Optional[StyleModelConfig] = None,
    beta_fixed: Optional[jnp.ndarray] = None,  # shape (K, P)
) -> None:
    """
    Gaussian style-topic model for standardized rate + efficiency features.

    Parameters
    ----------
    X : jnp.ndarray, shape (N, P)
        Standardized feature matrix (rate + efficiency).
        Each column should be roughly zero-mean, unit-variance over the TRAIN set.
    K : int, optional
        Number of latent styles. If None, will be taken from `config.K`.
    config : StyleModelConfig, optional
        Model hyperparameters. If None, a default config with given K is used.
    beta_fixed : jnp.ndarray, optional
        If provided (shape (K, P)), the style loading matrix is treated as fixed/known and
        is not sampled. This is useful for two-stage workflows where you fit beta on the
        training set and then infer only theta on validation/test sets.

    Model
    -----
    For each player n = 1..N and feature p = 1..P:

      theta_n ~ Dirichlet(alpha * 1_K)            # player-style proportions
      beta_{k,p} ~ Normal(0, tau_beta)            # style loadings for feature p
      sigma_p ~ HalfCauchy(0, tau_sigma)          # noise scales per feature
      X_{n,p} ~ Normal( (theta_n @ beta[:, p]), sigma_p )

    Notes
    -----
    - X is assumed standardized; do NOT standardize inside this model.
    - Shapes:
    - theta : (N, K)
    - beta  : (K, P)
    - sigma : (P,)
    - mu    : (N, P)
    """

    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape (N, P), got {X.shape}.")

    N, P = X.shape

    if config is None and K is None:
        raise ValueError("Either `K` or `config` with `K` must be provided.")

    if config is None:
        config = StyleModelConfig(K=K)
    if K is None:
        K = config.K

    # ----------------------------------------------------------------------
    # Priors
    # ----------------------------------------------------------------------

    # Player-style proportions theta_n on the K-dimensional simplex
    # theta has shape (N, K)
    if config.alpha_w is None:
        alpha_vec = jnp.ones(K) * config.alpha
    else:
        # Break topic symmetry: scale each topic's concentration by positive weights.
        # We normalize weights to keep the overall concentration comparable.
        w = config.alpha_w / jnp.mean(config.alpha_w)
        alpha_vec = config.alpha * w

    theta = numpyro.sample(
        "theta",
        dist.Dirichlet(alpha_vec).expand([N]).to_event(1),
    )

    # ----------------------------------------------------------------------
    # Group-specific priors for beta (feature-dependent scales)
    # ----------------------------------------------------------------------
    # Group-level scales (HalfCauchy is standard for scale parameters)
    tau_beta_count = numpyro.sample("tau_beta_count", dist.HalfCauchy(config.tau_beta * config.tau_beta_count))
    tau_beta_pct   = numpyro.sample("tau_beta_pct",   dist.HalfCauchy(config.tau_beta * config.tau_beta_pct))
    tau_beta_other = numpyro.sample("tau_beta_other", dist.HalfCauchy(config.tau_beta * config.tau_beta_other))

    # IMPORTANT: feature order MUST match Train.build_design_matrix
    # [PTS36, REB36, AST36, FG_logit, 3P_logit, FT_logit, WS/48, BPM, VORP, AvgMP]
    tau_vec = jnp.array([
        tau_beta_count,  # PTS per 36
        tau_beta_count,  # REB per 36
        tau_beta_count,  # AST per 36
        tau_beta_pct,    # FG% (logit)
        tau_beta_pct,    # 3P% (logit)
        tau_beta_pct,    # FT% (logit)
        tau_beta_other,  # WS/48
        tau_beta_other,  # BPM
        tau_beta_other,  # VORP
        tau_beta_count,  # Avg minutes played
    ])  # shape (10,)

    # If you ever change P != 10, guard it explicitly
    # (recommended to avoid silent mismatch)
    if P != tau_vec.shape[0]:
        raise ValueError(f"Expected P={tau_vec.shape[0]} features, got P={P}. Check feature construction order.")

    if beta_fixed is None:
        beta = numpyro.sample(
            "beta",
            dist.Normal(0.0, tau_vec).expand([K, P]).to_event(2),
        )
    else:
        # Treat beta as fixed/known (two-stage inference).
        if beta_fixed.shape != (K, P):
            raise ValueError(f"beta_fixed must have shape {(K, P)}, got {beta_fixed.shape}.")
        beta = numpyro.deterministic("beta", beta_fixed)

    # Feature-specific noise scales sigma_p, shape (P,)
    sigma = numpyro.sample(
        "sigma",
        dist.HalfCauchy(config.tau_sigma).expand([P]).to_event(1),
    )

    # ----------------------------------------------------------------------
    # Likelihood
    # ----------------------------------------------------------------------

    # Mean matrix mu_{n,p} = theta_n^T beta[:, p]
    # theta: (N, K), beta: (K, P) -> mu: (N, P)
    mu = jnp.matmul(theta, beta)

    # Observation model: X_{n,p} ~ Normal(mu_{n,p}, sigma_p)
    # We treat (N, P) as independent observations under the hood via broadcasting.
    with numpyro.plate("players", N, dim=-2):
        with numpyro.plate("features", P, dim=-1):
            numpyro.sample(
                "X",
                dist.Normal(mu, sigma),
                obs=X,
            )
