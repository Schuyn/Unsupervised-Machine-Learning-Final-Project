from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Callable

import numpy as np


# ============================================================
# Utilities
# ============================================================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _theta_entropy(theta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # theta: (N, K), row-stochastic
    return -np.sum(theta * _safe_log(theta, eps=eps), axis=1)


def _keff(theta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.exp(_theta_entropy(theta, eps=eps))


def _cosine_similarity_matrix(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # M: (K, D)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + eps
    A = M / norms
    return A @ A.T


def _upper_tri_values(A: np.ndarray, k: int = 1) -> np.ndarray:
    iu = np.triu_indices(A.shape[0], k=k)
    return A[iu]


def _corr_upper(X: np.ndarray) -> np.ndarray:
    # X: (N, P)
    C = np.corrcoef(X, rowvar=False)
    return _upper_tri_values(C, k=1)


def _moments_per_feature(X: np.ndarray) -> Dict[str, np.ndarray]:
    # returns arrays length P
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    z = (X - mu) / std
    skew = np.mean(z**3, axis=0)
    kurt = np.mean(z**4, axis=0)  # not excess
    return {"mean": mu, "var": var, "skew": skew, "kurt": kurt}


def _quantiles_per_feature(X: np.ndarray, qs=(0.05, 0.5, 0.95)) -> Dict[str, np.ndarray]:
    Q = np.quantile(X, qs, axis=0)  # shape (len(qs), P)
    return {f"q{int(q*100):02d}": Q[i] for i, q in enumerate(qs)}


def _tail_prob_per_feature(X: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    # thresh shape (P,)
    return np.mean(X > thresh[None, :], axis=0)


def _p_value_mc(T_rep: np.ndarray, T_obs: np.ndarray, side: str = "two_sided") -> np.ndarray:
    """
    Monte Carlo p-values elementwise.
    T_rep: (R, ...) ; T_obs: (...)
    Returns p in [0,1] with same shape as T_obs.
    """
    # flatten all but first dim
    R = T_rep.shape[0]
    rep = T_rep.reshape(R, -1)
    obs = np.ravel(T_obs)

    if side == "upper":
        p = np.mean(rep >= obs[None, :], axis=0)
    elif side == "lower":
        p = np.mean(rep <= obs[None, :], axis=0)
    else:
        # two-sided around median: 2 * min(P(rep>=obs), P(rep<=obs))
        pu = np.mean(rep >= obs[None, :], axis=0)
        pl = np.mean(rep <= obs[None, :], axis=0)
        p = 2.0 * np.minimum(pu, pl)
        p = np.clip(p, 0.0, 1.0)

    return p.reshape(T_obs.shape)


def _unique_ratio_per_feature(X: np.ndarray, decimals: int = 2) -> np.ndarray:
    # unique ratio after rounding, per feature
    Xr = np.round(X, decimals=decimals)
    N, P = Xr.shape
    out = np.zeros(P, dtype=float)
    for j in range(P):
        out[j] = len(np.unique(Xr[:, j])) / max(N, 1)
    return out


# ============================================================
# Model sampling / generation
# ============================================================

def sample_theta_dirichlet(alpha: np.ndarray, N: int, seed: int) -> np.ndarray:
    """
    alpha: (K,)
    returns theta: (N,K)
    """
    g = _rng(seed)
    # numpy supports dirichlet for 1d alpha
    return g.dirichlet(alpha, size=N)


def generate_X_from_model(
    *,
    theta: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Gaussian emission:
      X ~ N(theta @ beta, diag(sigma^2))  OR diag(sigma) depending on your convention.

    Assumption here:
      - beta: (K, P)
      - theta: (N, K)
      - sigma: (P,) representing std per feature
    """
    g = _rng(seed)
    mean = theta @ beta  # (N,P)
    sigma = np.asarray(sigma).reshape(1, -1)  # (1,P)
    eps = g.normal(size=mean.shape)
    return mean + eps * sigma


# ============================================================
# Core PPC / Null suites
# ============================================================

def run_ppc_suite(
    *,
    X_obs: np.ndarray,
    theta_for_generation: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    n_rep: int = 200,
    seed: int = 0,
    compute_unique: bool = True,
    unique_decimals: int = 2,
    compute_mse_recon: bool = False,
    theta_for_recon: Optional[np.ndarray] = None,
    style_tau: float = 0.7,
    max_style_examples: int = 2000,
) -> Dict[str, Any]:
    """
    TRUE PPC suite:
      - Uses theta_for_generation to generate replicate datasets from the model.
      - Compares observed statistics to replicate distribution.

    Optional:
      - compute_mse_recon: reconstruction metric; if enabled, requires theta_for_recon
        and is treated as an inference/reconstruction diagnostic, NOT a generative PPC.

    Returns:
      dict with:
        stats_obs, stats_rep (some), p-values, and diagnostics.
    """
    X_obs = np.asarray(X_obs)
    N, P = X_obs.shape
    beta = np.asarray(beta)
    sigma = np.asarray(sigma)

    # -------------------------
    # Observed statistics
    # -------------------------
    obs_mom = _moments_per_feature(X_obs)
    obs_corr = _corr_upper(X_obs)
    obs_q = _quantiles_per_feature(X_obs, qs=(0.05, 0.5, 0.95))
    obs_tail_thr = obs_q["q95"]  # feature-wise 95% threshold
    obs_tail = _tail_prob_per_feature(X_obs, obs_tail_thr)

    obs_unique = _unique_ratio_per_feature(X_obs, decimals=unique_decimals) if compute_unique else None

    # θ diagnostics (observed θ is not "data statistic" per se, but critical for degeneracy)
    theta_diag = {
        "theta_entropy": _theta_entropy(theta_for_generation),
        "theta_keff": _keff(theta_for_generation),
        "theta_max": np.max(theta_for_generation, axis=1),
    }

    # prototype diagnostics
    proto_cos = _cosine_similarity_matrix(beta)
    proto_cos_upper = _upper_tri_values(proto_cos, k=1)

    # -------------------------
    # Replicate statistics
    # -------------------------
    g = _rng(seed)
    X_rep_stats = {
        "moments": {k: np.zeros((n_rep, P)) for k in obs_mom.keys()},
        "corr_upper": np.zeros((n_rep, obs_corr.shape[0])),
        "quantiles": {k: np.zeros((n_rep, P)) for k in obs_q.keys()},
        "tail_prob": np.zeros((n_rep, P)),
        "unique_ratio": np.zeros((n_rep, P)) if compute_unique else None,
    }

    # For p-values on θ diagnostics under generation (optional, but useful)
    theta_rep_entropy = np.zeros((n_rep, N))
    theta_rep_keff = np.zeros((n_rep, N))
    theta_rep_max = np.zeros((n_rep, N))

    # Replicate loop
    for r in range(n_rep):
        # If you want θ to be stochastic per replicate, you can pass a different theta_for_generation each time.
        # Here we keep it fixed (conditional PPC). This is standard in many PPC workflows.
        Xr = generate_X_from_model(theta=theta_for_generation, beta=beta, sigma=sigma, seed=int(g.integers(1e9)))

        mr = _moments_per_feature(Xr)
        for k in mr:
            X_rep_stats["moments"][k][r] = mr[k]

        X_rep_stats["corr_upper"][r] = _corr_upper(Xr)

        qr = _quantiles_per_feature(Xr, qs=(0.05, 0.5, 0.95))
        for k in qr:
            X_rep_stats["quantiles"][k][r] = qr[k]

        # tail prob relative to OBS threshold (critical!)
        X_rep_stats["tail_prob"][r] = _tail_prob_per_feature(Xr, obs_tail_thr)

        if compute_unique:
            X_rep_stats["unique_ratio"][r] = _unique_ratio_per_feature(Xr, decimals=unique_decimals)

        # θ diagnostics replicate (same theta if fixed; kept for API symmetry)
        theta_rep_entropy[r] = theta_diag["theta_entropy"]
        theta_rep_keff[r] = theta_diag["theta_keff"]
        theta_rep_max[r] = theta_diag["theta_max"]

    # -------------------------
    # P-values for stats
    # -------------------------
    p = {"moments": {}, "quantiles": {}}
    for k in obs_mom:
        p["moments"][k] = _p_value_mc(X_rep_stats["moments"][k], obs_mom[k], side="two_sided")

    p["corr_upper"] = _p_value_mc(X_rep_stats["corr_upper"], obs_corr, side="two_sided")

    for k in obs_q:
        p["quantiles"][k] = _p_value_mc(X_rep_stats["quantiles"][k], obs_q[k], side="two_sided")

    p["tail_prob"] = _p_value_mc(X_rep_stats["tail_prob"], obs_tail, side="two_sided")

    if compute_unique:
        p["unique_ratio"] = _p_value_mc(X_rep_stats["unique_ratio"], obs_unique, side="two_sided")

    # θ diag p-values (not strictly PPC on X, but helpful)
    p["theta_entropy"] = _p_value_mc(theta_rep_entropy, theta_diag["theta_entropy"], side="two_sided")
    p["theta_keff"] = _p_value_mc(theta_rep_keff, theta_diag["theta_keff"], side="two_sided")
    p["theta_max"] = _p_value_mc(theta_rep_max, theta_diag["theta_max"], side="two_sided")

    # -------------------------
    # Optional: reconstruction MSE diagnostic (NOT PPC)
    # -------------------------
    recon = None
    if compute_mse_recon:
        if theta_for_recon is None:
            raise ValueError("compute_mse_recon=True requires theta_for_recon.")
        Xhat = np.asarray(theta_for_recon) @ beta
        mse_feat = np.mean((X_obs - Xhat) ** 2, axis=0)
        recon = {"mse_feat": mse_feat}

    # -------------------------
    # Style-conditional stats (degeneracy-sensitive)
    # -------------------------
    style_cond = compute_style_conditional_stats(
        X=X_obs,
        theta=theta_for_generation,
        beta=beta,
        sigma=sigma,
        tau=style_tau,
        max_examples=max_style_examples,
        seed=seed + 999,
    )

    return {
        "obs": {
            "moments": obs_mom,
            "corr_upper": obs_corr,
            "quantiles": obs_q,
            "tail_prob": obs_tail,
            "unique_ratio": obs_unique,
        },
        "rep": {
            # store only what you need; can be large, so keep summary by default
            "moments_mean": {k: np.mean(v, axis=0) for k, v in X_rep_stats["moments"].items()},
            "moments_sd": {k: np.std(v, axis=0) for k, v in X_rep_stats["moments"].items()},
            "quantiles_mean": {k: np.mean(v, axis=0) for k, v in X_rep_stats["quantiles"].items()},
            "tail_prob_mean": np.mean(X_rep_stats["tail_prob"], axis=0),
            "unique_ratio_mean": (np.mean(X_rep_stats["unique_ratio"], axis=0) if compute_unique else None),
        },
        "p": p,
        "diagnostics": {
            "theta": {
                "entropy": theta_diag["theta_entropy"],
                "keff": theta_diag["theta_keff"],
                "theta_max": theta_diag["theta_max"],
                "entropy_mean": float(np.mean(theta_diag["theta_entropy"])),
                "keff_mean": float(np.mean(theta_diag["theta_keff"])),
                "theta_max_mean": float(np.mean(theta_diag["theta_max"])),
            },
            "prototypes": {
                "cosine_upper": proto_cos_upper,
                "cosine_upper_mean": float(np.mean(proto_cos_upper)),
                "cosine_upper_max": float(np.max(proto_cos_upper)),
            },
        },
        "reconstruction": recon,
        "style_conditional": style_cond,
    }


def compute_style_conditional_stats(
    *,
    X: np.ndarray,
    theta: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    tau: float = 0.7,
    max_examples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Degeneracy-sensitive check:
      - For each style k, select "high-purity" subset I_k = {i: theta_ik > tau}
      - Compare observed subset moments to model-implied conditional mean beta[k]
      - Optionally generate conditional samples with one-hot theta to compare tails.

    Returns compact per-style summaries.
    """
    g = _rng(seed)
    X = np.asarray(X)
    theta = np.asarray(theta)
    beta = np.asarray(beta)
    sigma = np.asarray(sigma)

    N, P = X.shape
    K = theta.shape[1]

    out = {
        "tau": tau,
        "per_style": [],
        "global": {
            "avg_purity": float(np.mean(np.max(theta, axis=1))),
            "frac_assigned": None,
        },
    }

    total_assigned = 0

    for k in range(K):
        idx = np.where(theta[:, k] > tau)[0]
        if idx.size > max_examples:
            idx = g.choice(idx, size=max_examples, replace=False)
        total_assigned += idx.size

        if idx.size == 0:
            out["per_style"].append({
                "k": k,
                "n": 0,
                "obs_mean": None,
                "obs_var": None,
                "model_mean": beta[k].copy(),
                "mean_l2": None,
                "tail_obs": None,
                "tail_rep": None,
            })
            continue

        Xk = X[idx]
        obs_mean = np.mean(Xk, axis=0)
        obs_var = np.var(Xk, axis=0, ddof=0)

        model_mean = beta[k]
        mean_l2 = float(np.linalg.norm(obs_mean - model_mean))

        # tail comparison: threshold based on style subset
        thr = np.quantile(Xk, 0.95, axis=0)
        tail_obs = _tail_prob_per_feature(Xk, thr)

        # conditional replicate: one-hot theta on style k
        thetak = np.zeros((idx.size, K))
        thetak[:, k] = 1.0
        Xrep = generate_X_from_model(theta=thetak, beta=beta, sigma=sigma, seed=int(g.integers(1e9)))
        tail_rep = _tail_prob_per_feature(Xrep, thr)

        out["per_style"].append({
            "k": k,
            "n": int(idx.size),
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "model_mean": model_mean.copy(),
            "mean_l2": mean_l2,
            "tail_obs": tail_obs,
            "tail_rep": tail_rep,
            "tail_gap_l1": float(np.mean(np.abs(tail_obs - tail_rep))),
        })

    out["global"]["frac_assigned"] = float(total_assigned / max(N, 1))
    return out


def run_null_suite(
    *,
    X_obs: np.ndarray,
    null_reps: np.ndarray,
    unique_decimals: int = 2,
) -> Dict[str, Any]:
    """
    Null suite:
      - Compare observed stats against stats computed on null replicate datasets.
    null_reps: (R, N, P)
    """
    X_obs = np.asarray(X_obs)
    R, N, P = null_reps.shape

    obs_mom = _moments_per_feature(X_obs)
    obs_corr = _corr_upper(X_obs)
    obs_q = _quantiles_per_feature(X_obs, qs=(0.05, 0.5, 0.95))
    obs_tail_thr = obs_q["q95"]
    obs_tail = _tail_prob_per_feature(X_obs, obs_tail_thr)
    obs_unique = _unique_ratio_per_feature(X_obs, decimals=unique_decimals)

    rep_mom = {k: np.zeros((R, P)) for k in obs_mom.keys()}
    rep_corr = np.zeros((R, obs_corr.shape[0]))
    rep_q = {k: np.zeros((R, P)) for k in obs_q.keys()}
    rep_tail = np.zeros((R, P))
    rep_unique = np.zeros((R, P))

    for r in range(R):
        Xr = null_reps[r]
        mr = _moments_per_feature(Xr)
        for k in mr:
            rep_mom[k][r] = mr[k]
        rep_corr[r] = _corr_upper(Xr)
        qr = _quantiles_per_feature(Xr, qs=(0.05, 0.5, 0.95))
        for k in qr:
            rep_q[k][r] = qr[k]
        rep_tail[r] = _tail_prob_per_feature(Xr, obs_tail_thr)
        rep_unique[r] = _unique_ratio_per_feature(Xr, decimals=unique_decimals)

    p = {"moments": {}, "quantiles": {}}
    for k in obs_mom:
        p["moments"][k] = _p_value_mc(rep_mom[k], obs_mom[k], side="two_sided")
    p["corr_upper"] = _p_value_mc(rep_corr, obs_corr, side="two_sided")
    for k in obs_q:
        p["quantiles"][k] = _p_value_mc(rep_q[k], obs_q[k], side="two_sided")
    p["tail_prob"] = _p_value_mc(rep_tail, obs_tail, side="two_sided")
    p["unique_ratio"] = _p_value_mc(rep_unique, obs_unique, side="two_sided")

    return {
        "obs": {
            "moments": obs_mom,
            "corr_upper": obs_corr,
            "quantiles": obs_q,
            "tail_prob": obs_tail,
            "unique_ratio": obs_unique,
        },
        "p": p,
    }


# ============================================================
# Null generators (you already have; included for completeness)
# ============================================================

def null_independent_standard_normal(*, N: int, P: int, n_rep: int, seed: int) -> np.ndarray:
    g = _rng(seed)
    return g.normal(size=(n_rep, N, P))


def null_column_permutation(X: np.ndarray, n_rep: int, seed: int) -> np.ndarray:
    g = _rng(seed)
    X = np.asarray(X)
    N, P = X.shape
    out = np.zeros((n_rep, N, P), dtype=float)
    for r in range(n_rep):
        Xr = X.copy()
        for j in range(P):
            g.shuffle(Xr[:, j])
        out[r] = Xr
    return out


# ============================================================
# Top-level Orchestrator (UPDATED)
# ============================================================

def run_full_checks(
    *,
    X_train_std: np.ndarray,
    X_val_std: Optional[np.ndarray],
    posterior_all: Dict[str, Any],
    infer_theta_given_beta_fn: Optional[Callable[..., np.ndarray]] = None,
    stats_train=None,
    n_rep: int = 200,
    seed: int = 0,
    unique_decimals: int = 2,
    # NEW knobs
    style_tau: float = 0.7,
    compute_train_reconstruction: bool = True,
    dirichlet_alpha_fallback: float = 0.5,
) -> Dict[str, Any]:
    """
    Orchestrates:
      - Train GENERATIVE PPC:
          uses posterior theta if aligned; otherwise uses Dirichlet-sampled theta
      - Train inference diagnostics (optional):
          uses infer_theta_given_beta_fn to compute reconstruction MSE and θ collapse metrics
      - Val PPC:
          uses inferred theta (beta fixed) as CONDITIONAL generation anchor; plus reconstruction metrics
      - Null checks on train:
          N(0,1) and column-permutation null, using structure-aware stats

    Returns a dict report suitable for saving.
    """
    X_train_std = np.asarray(X_train_std)
    Ntr, P = X_train_std.shape

    beta = np.asarray(posterior_all["beta"])
    sigma = np.asarray(posterior_all["sigma"])

    theta_post = posterior_all.get("theta", None)
    alpha_post = posterior_all.get("alpha", None)  # optional, if you stored Dirichlet params

    report: Dict[str, Any] = {
        "meta": {
            "n_rep": int(n_rep),
            "seed": int(seed),
            "unique_decimals": int(unique_decimals),
            "style_tau": float(style_tau),
            "feature_names": getattr(stats_train, "feature_names", None) if stats_train is not None else None,
        },
        "train": {},
        "val": {},
        "nulls": {},
    }

    # ------------------------------------------------------------
    # TRAIN: choose theta for GENERATIVE PPC (do NOT infer from X!)
    # ------------------------------------------------------------
    theta_train_gen = None

    if theta_post is not None:
        theta_post = np.asarray(theta_post)
        if theta_post.shape[0] == Ntr:
            theta_train_gen = theta_post
        else:
            # If misaligned, we refuse to "infer" for PPC; instead sample from Dirichlet.
            theta_train_gen = None

    if theta_train_gen is None:
        # Dirichlet alpha choice: use stored alpha if available; else symmetric fallback
        K = beta.shape[0]
        if alpha_post is not None:
            alpha = np.asarray(alpha_post).reshape(-1)
            if alpha.shape[0] != K:
                raise ValueError(f"alpha has shape {alpha.shape}, expected (K,) with K={K}.")
        else:
            alpha = np.full(K, dirichlet_alpha_fallback, dtype=float)
        theta_train_gen = sample_theta_dirichlet(alpha=alpha, N=Ntr, seed=seed + 12345)

    report["train"]["theta_gen_shape"] = list(theta_train_gen.shape)

    # -------------------------
    # TRAIN: true generative PPC
    # -------------------------
    report["train"]["ppc"] = run_ppc_suite(
        X_obs=X_train_std,
        theta_for_generation=theta_train_gen,
        beta=beta,
        sigma=sigma,
        n_rep=n_rep,
        seed=seed,
        compute_unique=True,
        unique_decimals=unique_decimals,
        compute_mse_recon=False,  # critical: MSE is NOT PPC
        style_tau=style_tau,
    )

    # ------------------------------------------------------------
    # TRAIN: inference/reconstruction diagnostics (optional)
    # ------------------------------------------------------------
    if compute_train_reconstruction and infer_theta_given_beta_fn is not None:
        theta_train_inf = infer_theta_given_beta_fn(X_train_std, beta_fixed=beta, l2_theta=1e-3)
        report["train"]["theta_inf_shape"] = list(np.asarray(theta_train_inf).shape)

        # Store collapse diagnostics on inferred θ
        ent = _theta_entropy(theta_train_inf)
        report["train"]["inference"] = {
            "theta_entropy_mean": float(np.mean(ent)),
            "theta_keff_mean": float(np.mean(np.exp(ent))),
            "theta_max_mean": float(np.mean(np.max(theta_train_inf, axis=1))),
            "mse_feat": np.mean((X_train_std - (theta_train_inf @ beta)) ** 2, axis=0),
        }

    # ------------------------------------------------------------
    # VAL: conditional PPC with inferred θ (beta fixed)
    # ------------------------------------------------------------
    if X_val_std is not None and infer_theta_given_beta_fn is not None:
        X_val_std = np.asarray(X_val_std)
        theta_val = infer_theta_given_beta_fn(X_val_std, beta_fixed=beta, l2_theta=1e-3)
        report["val"]["theta_val_shape"] = list(np.asarray(theta_val).shape)

        # Generative PPC conditioned on inferred θ_val:
        report["val"]["ppc"] = run_ppc_suite(
            X_obs=X_val_std,
            theta_for_generation=theta_val,  # conditional PPC anchor
            beta=beta,
            sigma=sigma,
            n_rep=n_rep,
            seed=seed + 1,
            compute_unique=True,
            unique_decimals=unique_decimals,
            compute_mse_recon=True,
            theta_for_recon=theta_val,  # reconstruction diagnostic only
            style_tau=style_tau,
        )

    # ------------------------------------------------------------
    # NULL checks on TRAIN
    # ------------------------------------------------------------
    Xnull0 = null_independent_standard_normal(N=Ntr, P=P, n_rep=n_rep, seed=seed + 10)
    report["nulls"]["indep_standard_normal"] = run_null_suite(
        X_obs=X_train_std,
        null_reps=Xnull0,
        unique_decimals=unique_decimals,
    )

    Xnull1 = null_column_permutation(X_train_std, n_rep=n_rep, seed=seed + 11)
    report["nulls"]["column_permutation"] = run_null_suite(
        X_obs=X_train_std,
        null_reps=Xnull1,
        unique_decimals=unique_decimals,
    )

    # Summaries
    report["summary"] = summarize_checks(report)
    return report


# ============================================================
# Summaries (UPDATED)
# ============================================================

def summarize_checks(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact summaries designed to diagnose degeneracy, not to hide it.

    For each block, we report:
      - median p-value (closer to 0.5 is better)
      - frac_extreme: fraction of p-values outside [0.05, 0.95]
      - worst_features: indices with largest |p-0.5| for feature-wise stats
    """
    def _summarize_p_array(p_arr: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if p_arr is None:
            return None
        p_flat = np.ravel(p_arr)
        med = float(np.median(p_flat))
        frac_ext = float(np.mean((p_flat < 0.05) | (p_flat > 0.95)))
        return {"median_p": med, "frac_extreme": frac_ext}

    def _worst_feature_list(p_feat: Optional[np.ndarray], topk: int = 3) -> Optional[List[int]]:
        if p_feat is None:
            return None
        score = np.abs(p_feat - 0.5)
        idx = np.argsort(-score)[:topk]
        return [int(i) for i in idx]

    def _block_summary(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not block:
            return None
        p = block.get("p", {})

        out = {}
        # moments
        if "moments" in p:
            out["moments"] = {k: _summarize_p_array(v) for k, v in p["moments"].items()}

        # corr upper
        if "corr_upper" in p:
            out["corr_upper"] = _summarize_p_array(p["corr_upper"])

        # quantiles
        if "quantiles" in p:
            out["quantiles"] = {k: _summarize_p_array(v) for k, v in p["quantiles"].items()}

        # tail prob
        if "tail_prob" in p:
            out["tail_prob"] = _summarize_p_array(p["tail_prob"])
            out["tail_prob_worst_features"] = _worst_feature_list(p["tail_prob"], topk=5)

        # unique ratio
        if "unique_ratio" in p:
            out["unique_ratio"] = _summarize_p_array(p["unique_ratio"])
            out["unique_ratio_worst_features"] = _worst_feature_list(p["unique_ratio"], topk=5)

        # θ diags
        if "theta_entropy" in p:
            out["theta_entropy"] = _summarize_p_array(p["theta_entropy"])
        if "theta_keff" in p:
            out["theta_keff"] = _summarize_p_array(p["theta_keff"])
        if "theta_max" in p:
            out["theta_max"] = _summarize_p_array(p["theta_max"])

        # prototype diagnostics
        diag = block.get("diagnostics", {})
        if "prototypes" in diag:
            out["prototype_cosine_mean"] = diag["prototypes"].get("cosine_upper_mean")
            out["prototype_cosine_max"] = diag["prototypes"].get("cosine_upper_max")

        # reconstruction diagnostics if present
        if "reconstruction" in block and block["reconstruction"] is not None:
            mse = block["reconstruction"].get("mse_feat", None)
            if mse is not None:
                out["recon_mse_feat_mean"] = float(np.mean(mse))
                out["recon_mse_feat_worst_features"] = _worst_feature_list(
                    _p_value_mc(np.expand_dims(mse, 0), mse, side="two_sided"),  # dummy; keep API
                    topk=5,
                )

        # style-conditional compact signal
        sc = block.get("style_conditional", None)
        if sc is not None:
            # summarize mean_l2 over styles with n>0
            l2s = []
            gaps = []
            ns = []
            for item in sc.get("per_style", []):
                if item.get("n", 0) > 0:
                    l2s.append(item.get("mean_l2", np.nan))
                    gaps.append(item.get("tail_gap_l1", np.nan))
                    ns.append(item.get("n", 0))
            if len(l2s) > 0:
                out["style_cond_mean_l2_mean"] = float(np.nanmean(l2s))
                out["style_cond_tail_gap_l1_mean"] = float(np.nanmean(gaps))
                out["style_cond_avg_n"] = float(np.mean(ns))
                out["style_cond_frac_assigned"] = sc.get("global", {}).get("frac_assigned")

        return out

    out = {}
    out["train_ppc"] = _block_summary(report.get("train", {}).get("ppc", {}))
    out["val_ppc"] = _block_summary(report.get("val", {}).get("ppc", {}))
    out["null_indep"] = _block_summary(report.get("nulls", {}).get("indep_standard_normal", {}))
    out["null_perm"] = _block_summary(report.get("nulls", {}).get("column_permutation", {}))

    # Include train inference block if present
    if "inference" in report.get("train", {}):
        out["train_inference"] = report["train"]["inference"]

    return out
