# Train.py
# End-to-end pipeline for the Gaussian style-topic model:
#  - load train/validation CSVs
#  - build rate + efficiency feature matrix
#  - standardize features (fit on train only)
#  - cross-validate on train_data to choose hyperparameters
#  - train final model on full train_data
#  - evaluate on validation_data (treated as held-out test)
#
# Assumes Model.py defines:
#   - StyleModelConfig
#   - style_topic_model

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from scipy.stats import rankdata, norm
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal

from Model import StyleModelConfig, style_topic_model
from Metrics import compute_reconstruction_mse, CVResult
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------
# Small container to hold feature scaling statistics
# ---------------------------------------------------------------------

@dataclass
class FeatureStats:
    """Stores per-feature mean/std and train-fitted Gaussianization state."""
    feature_names: List[str]
    mean: np.ndarray  # shape (P,)
    std: np.ndarray   # shape (P,)
    extra_metrics: Dict[str, float] = None 


# ---------------------------------------------------------------------
# Feature engineering utilities
# ---------------------------------------------------------------------
def hard_assign_from_theta(theta: np.ndarray) -> np.ndarray:
    """Convert soft assignments theta (N, K) to hard assignments (N,) via argmax."""
    return np.argmax(theta, axis=1)


def ari_stability(assignments: list[np.ndarray]) -> float:
    """Compute average pairwise Adjusted Rand Index (ARI) over a list of hard assignments."""
    R = len(assignments)
    if R < 2:
        return float("nan")
    scores = []
    for i in range(R):
        for j in range(i + 1, R):
            scores.append(adjusted_rand_score(assignments[i], assignments[j]))
    return float(np.mean(scores))


def extract_theta_mean(posterior_samples: dict) -> np.ndarray:
    """Extract mean theta from posterior samples dict."""
    theta = posterior_samples["theta"]
    return np.mean(theta, axis=0)  # (N, K)


def make_alpha_w(K):
    # simple, safe asymmetry: linearly increasing weights
    w = np.linspace(0.7, 1.3, K)
    return w / w.mean()
    

def safe_per36(per_game: np.ndarray, avg_minutes: np.ndarray) -> np.ndarray:
    """Compute per-36 rate safely."""
    per_game = np.asarray(per_game, dtype=float)
    avg_minutes = np.asarray(avg_minutes, dtype=float)
    out = np.zeros_like(per_game, dtype=float)
    mask = avg_minutes > 0
    out[mask] = per_game[mask] / avg_minutes[mask] * 36.0
    return out


def logit_transform(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Logit transform for percentages in [0,1].

    Clips into [eps, 1-eps] before applying logit.
    """
    p = np.asarray(p, dtype=float)
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def build_design_matrix(
    df: pd.DataFrame,
    stats: Optional[FeatureStats] = None,
) -> Tuple[np.ndarray, FeatureStats]:
    """
    Build standardized feature matrix X_std from raw NBA draft dataframe.

    Steps:
      1. Construct rate features (per-36 for PTS/REB/AST).
      2. Use efficiency / advanced metrics as-is (with optional transforms).
      3. Standardize each feature (fit mean/std if stats is None).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with columns like:
        ['points_per_game', 'average_total_rebounds', 'average_assists',
         'average_minutes_played', 'field_goal_percentage',
         '3_point_percentage', 'free_throw_percentage',
         'win_shares_per_48_minutes', 'box_plus_minus',
         'value_over_replacement', ...]
    stats : FeatureStats, optional
        If provided, use these mean/std to standardize (no refit).
        If None, fit mean/std on this df and return them.

    Returns
    -------
    X_std : np.ndarray, shape (N, P)
        Standardized feature matrix.
    stats_out : FeatureStats
        Fitted statistics (either new or same as input).
    """
    # ------------- 1. Construct raw feature matrix (unstandardized) -------------
    # Rate inputs
    pts_pg = df["points_per_game"].to_numpy()
    reb_pg = df["average_total_rebounds"].to_numpy()
    ast_pg = df["average_assists"].to_numpy()
    mp_pg = df["average_minutes_played"].to_numpy()

    pts_per36 = safe_per36(pts_pg, mp_pg)
    reb_per36 = safe_per36(reb_pg, mp_pg)
    ast_per36 = safe_per36(ast_pg, mp_pg)

    # Percentages (0-1). We'll logit-transform them to R-valued.
    fg_pct = df["field_goal_percentage"].to_numpy()
    tp_pct = df["3_point_percentage"].to_numpy()
    ft_pct = df["free_throw_percentage"].to_numpy()

    fg_logit = logit_transform(fg_pct)
    tp_logit = logit_transform(tp_pct)
    ft_logit = logit_transform(ft_pct)

    # Advanced metrics (already roughly real-valued and unbounded)
    ws48 = df["win_shares_per_48_minutes"].to_numpy()
    bpm = df["box_plus_minus"].to_numpy()
    vorp = df["value_over_replacement"].to_numpy()

    # Optionally include average_minutes_played itself as a role indicator
    # (per-game usage / role).
    avg_mp = mp_pg

    # Concatenate features into X_raw: shape (N, P)
    feature_names = [
        "pts_per36",
        "reb_per36",
        "ast_per36",
        "fg_logit",
        "tp_logit",
        "ft_logit",
        "win_shares_per_48",
        "box_plus_minus",
        "value_over_replacement",
        "avg_minutes_played",
    ]

    X_raw = np.column_stack(
        [
            pts_per36,
            reb_per36,
            ast_per36,
            fg_logit,
            tp_logit,
            ft_logit,
            ws48,
            bpm,
            vorp,
            avg_mp,
        ]
    )

    # # ----- Gaussianization: rank -> normal score -----
    # X_gauss = np.zeros_like(X_raw)
    # for j in range(X_raw.shape[1]):
    #     r = rankdata(X_raw[:, j], method="average")
    #     u = r / (len(r) + 1.0)
    #     X_gauss[:, j] = norm.ppf(u)

    # X_raw = X_gauss

    # ------------- 1.5 Gaussianization (train-fitted CDF) -------------
    def gaussianize_with_train_cdf(X: np.ndarray, sorted_vals: List[np.ndarray]) -> np.ndarray:
        """
        Map each feature to approx N(0,1) using the empirical CDF fitted on TRAIN.
        For each x, u = rank_train(x)/(n_train+1), then z = Phi^{-1}(u).
        """
        N, P = X.shape
        Z = np.zeros_like(X, dtype=float)
        for j in range(P):
            sv = sorted_vals[j]
            # rank in train distribution (0..n_train)
            r = np.searchsorted(sv, X[:, j], side="right")
            u = r / (len(sv) + 1.0)
            # avoid infinities
            u = np.clip(u, 1e-6, 1.0 - 1e-6)
            Z[:, j] = norm.ppf(u)
        return Z
    

    # ------------- 2. Gaussianize + Standardize features -------------
    if stats is None:
        # Fit Gaussianization on THIS df (train or fold-train)
        sorted_vals = [np.sort(X_raw[:, j].astype(float)) for j in range(X_raw.shape[1])]
        X_g = gaussianize_with_train_cdf(X_raw.astype(float), sorted_vals)

        mean = X_g.mean(axis=0)
        std = X_g.std(axis=0, ddof=1)
        std[std == 0.0] = 1.0

        stats_out = FeatureStats(
            feature_names=feature_names,
            mean=mean,
            std=std,
            sorted_vals=sorted_vals,
        )
    else:
        # Use TRAIN-fitted Gaussianization + TRAIN mean/std
        stats_out = stats
        X_g = gaussianize_with_train_cdf(X_raw.astype(float), stats.sorted_vals)

        mean = stats.mean
        std = stats.std

    X_std = (X_g - mean) / std
    return X_std.astype(np.float32), stats_out


# ---------------------------------------------------------------------
# Model training / evaluation utilities
# ---------------------------------------------------------------------

def fit_style_model(
    X: np.ndarray,
    config: StyleModelConfig,
    num_steps: int = 3000,
    rng_seed: int = 2025,
    log_every: int = 500,
) -> Dict:
    """
    Fit the Gaussian style-topic model via SVI on given standardized X.

    Returns a dict with all necessary artifacts for downstream analysis.
    """
    X_jax = jnp.asarray(X, dtype=jnp.float32)

    guide = AutoNormal(style_topic_model)
    optimizer = optim.Adam(step_size=1e-3)
    svi = SVI(style_topic_model, guide, optimizer, loss=Trace_ELBO())

    rng_key = random.PRNGKey(rng_seed)
    svi_state = svi.init(rng_key, X_jax, config=config)

    loss_trace = []

    for step in range(num_steps):
        rng_key, subkey = random.split(rng_key)
        svi_state, loss = svi.update(svi_state, X_jax, config=config)
        loss_val = float(loss)
        loss_trace.append(loss_val)

        if (step + 1) % log_every == 0:
            print(f"[TRAIN] step={step+1}, loss={loss_val:.3f}")

    params = svi.get_params(svi_state)

    # Posterior sampling
    predictive = Predictive(
        model=style_topic_model,
        guide=guide,
        params=params,
        num_samples=500,
        return_sites=["theta", "beta", "sigma"],
    )
    rng_key, subkey = random.split(rng_key)
    post = predictive(subkey, X_jax, config=config)

    theta_mean = post["theta"].mean(axis=0)   # (N, K)
    beta_mean  = post["beta"].mean(axis=0)    # (K, P)
    sigma_mean = post["sigma"].mean(axis=0)   # (P,)

    return {
        "svi": svi,
        "params": params,
        "theta": theta_mean,
        "beta": beta_mean,
        "sigma": sigma_mean,
        "loss_trace": np.asarray(loss_trace),
        "config": config,
    }


def fit_style_model_tuple(
    X: np.ndarray,
    config: StyleModelConfig,
    num_steps: int = 3000,
    rng_seed: int = 2025,
    log_every: int = 500,
):
    """
    Backward-compatible tuple API:
    returns (svi, params, theta_mean, beta_mean, sigma_mean)
    """
    out = fit_style_model(
        X=X,
        config=config,
        num_steps=num_steps,
        rng_seed=rng_seed,
        log_every=log_every,
    )
    return out["svi"], out["params"], out["theta"], out["beta"], out["sigma"]


def infer_theta_given_beta(
    X,                 # (N, P) standardized
    beta_fixed,        # (K, P)
    *,
    theta_init=None,   # optional (N, K)
    l2_theta=1e-3,     # ridge on theta for stability
):
    """
    Solve theta = argmin ||X - theta @ beta_fixed||_F^2 + l2_theta * ||theta||_F^2

    Closed-form ridge regression per row:
      theta = X B^T (B B^T + l2 I)^{-1}
    """
    B = np.asarray(beta_fixed)              # (K, P)
    X = np.asarray(X)                       # (N, P)
    K = B.shape[0]

    BBt = B @ B.T                           # (K, K)
    A = BBt + l2_theta * np.eye(K)          # (K, K)
    A_inv = np.linalg.inv(A)                # (K, K)

    theta = (X @ B.T) @ A_inv               # (N, K)
    theta = np.exp(theta - theta.max(axis=1, keepdims=True))
    theta = theta / theta.sum(axis=1, keepdims=True)

    return theta


# ---------------------------------------------------------------------
# Cross-validation on train_data
# ---------------------------------------------------------------------

def kfold_indices(
    n_samples: int,
    n_splits: int,
    rng: np.random.Generator,
):
    """Generate K-fold train/val indices."""
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, val_idx


def cross_validate_on_train(
    train_df: pd.DataFrame,
    candidate_configs: List[StyleModelConfig],
    n_splits: int = 5,
    num_steps: int = 2000,
    base_seed: int = 42,
    n_repeats: int = 3,
) -> List[CVResult]:
    """
    Cross-validate hyperparameter configs on train_data only.

    Metrics:
      - Reconstruction MSE on fold-val (lower is better)
      - ARI stability on fold-train across repeated fits (higher is better)
        (measures structural stability of learned clustering)
    """
    rng = np.random.default_rng(base_seed)
    N = len(train_df)

    folds = list(kfold_indices(N, n_splits, rng))
    results: List[CVResult] = []

    for cfg_idx, config in enumerate(candidate_configs):
        print(f"\n[CV] Evaluating config {cfg_idx+1}/{len(candidate_configs)}: {config}")

        fold_mean_mse = []   # one scalar per fold
        fold_ari = []        # one scalar per fold (stability within fold)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"[CV]  Fold {fold_idx+1}/{n_splits}")

            df_train_fold = train_df.iloc[train_idx].reset_index(drop=True)
            df_val_fold = train_df.iloc[val_idx].reset_index(drop=True)

            # Fit feature scaling on fold_train only
            X_train_fold_std, stats_fold = build_design_matrix(df_train_fold, stats=None)
            X_val_fold_std, _ = build_design_matrix(df_val_fold, stats=stats_fold)

            # Repeat training n_repeats times (different seeds) to measure stability
            mse_runs = []
            z_train_runs = []

            for r in range(n_repeats):
                seed = base_seed + 1000 * fold_idx + r  # stable & non-overlapping

                # Fit model on fold-train
                svi_out = fit_style_model(
                    X_train_fold_std,
                    config=config,
                    num_steps=num_steps,
                    rng_seed=seed,
                    log_every=max(1, num_steps // 4),
                )

                beta = np.array(svi_out["beta"])    # (K, P)
                theta_train = np.array(svi_out["theta"])  # (N_train, K)

                # --- stability uses TRAIN assignments ---
                z_train = hard_assign_from_theta(theta_train)  # (N_train,)
                z_train_runs.append(z_train)

                # --- validation MSE uses VAL theta inferred with beta fixed ---
                theta_val = infer_theta_given_beta(
                    X_val_fold_std,
                    beta_fixed=beta,
                    l2_theta=1e-3,
                )
                theta_val = np.array(theta_val)

                mse = compute_reconstruction_mse(X_val_fold_std, theta_val, beta)
                mse_runs.append(mse)

            fold_mse = float(np.mean(mse_runs))
            fold_mean_mse.append(fold_mse)

            fold_stab = ari_stability(z_train_runs)  # average pairwise ARI within this fold
            fold_ari.append(float(fold_stab))

            print(f"[CV]    Fold {fold_idx+1} mean MSE = {fold_mse:.4f}, ARI-stability = {fold_stab:.4f}")

        mean_mse = float(np.mean(fold_mean_mse))
        std_mse = float(np.std(fold_mean_mse))

        mean_ari = float(np.mean(fold_ari))
        std_ari = float(np.std(fold_ari))

        print(f"[CV] Config {config} -> MSE = {mean_mse:.4f} ± {std_mse:.4f} | "
              f"ARI-stability = {mean_ari:.4f} ± {std_ari:.4f}")

        results.append(
            CVResult(
                config=config,
                mean_mse=mean_mse,
                std_mse=std_mse,
                extra_metrics={
                    "ari_stability": mean_ari,
                    "ari_stability_std": std_ari,
                    "n_repeats": float(n_repeats),
                    "n_splits": float(n_splits),
                },
            )
        )

    return results


# ---------------------------------------------------------------------
# Final training on full train_data + evaluation on validation_data
# ---------------------------------------------------------------------

def train_final_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    best_config: StyleModelConfig,
    num_steps: int = 4000,
    rng_seed: int = 2025,
    out_dir: str = "./outputs_style_model",
) -> None:
    """
    Train final model on full train_data using best_config, then evaluate on validation_data.

    - Fit feature scaling on full train_df
    - Train model to get final theta_train, beta, sigma
    - Transform validation_data with same stats
    - Infer theta_val (retraining using same config for simplicity)
    - Compute reconstruction MSE on validation_data
    - Save learned parameters and stats to disk
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Fit feature scaling on full train_data
    X_train_std, stats_train = build_design_matrix(train_df, stats=None)

    # 2) Train final model on full train_data
    print("\n[FINAL] Training final model on full train_data...")
    out = fit_style_model(
        X_train_std,
        config=best_config,
        num_steps=num_steps,
        rng_seed=rng_seed,
        log_every=max(1, num_steps // 10),
    )
    params = out["params"]
    theta_train = np.array(out["theta"])
    beta = np.array(out["beta"])
    sigma = np.array(out["sigma"])
    loss_trace = np.array(out["loss_trace"])


    # 3) Transform validation_data using train stats
    X_val_std, _ = build_design_matrix(val_df, stats=stats_train)

    # 4) Infer theta on validation_data (simple full refit using same config)
    print("\n[FINAL] Inferring theta on validation_data with beta fixed...")
    theta_val = infer_theta_given_beta(
        X_val_std,
        beta_fixed=beta,
        l2_theta=1e-3,
    )
    theta_val = np.array(theta_val)

    mse_val = compute_reconstruction_mse(X_val_std, theta_val, beta)
    print(f"\n[FINAL] Validation (test) reconstruction MSE = {mse_val:.4f}")

    # 5) Save parameters and stats
    np.savez(
        os.path.join(out_dir, "style_model_params.npz"),
        theta_train=np.array(theta_train),
        beta=np.array(beta),
        sigma=np.array(sigma),
        feature_names=np.array(stats_train.feature_names),
        mean=stats_train.mean,
        std=stats_train.std,
        config_K=best_config.K,
        config_alpha=best_config.alpha,
        config_tau_beta=best_config.tau_beta,
        config_tau_sigma=best_config.tau_sigma,
    )
    print(f"[FINAL] Saved model parameters to {os.path.join(out_dir, 'style_model_params.npz')}")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # You can adjust paths as needed
    train_path = "./Data/processed/train_data.csv"
    val_path = "./Data/processed/validation_data.csv"

    if not os.path.exists(train_path):
        # Fallback for environment where data is at /mnt/data
        if os.path.exists("/mnt/data/processed/train_data.csv"):
            train_path = "/mnt/data/processed/train_data.csv"
            val_path = "/mnt/data/processed/validation_data.csv"

    print(f"[INFO] Loading train_data from: {train_path}")
    print(f"[INFO] Loading validation_data from: {val_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # -----------------------------------------------------------------
    # Define candidate hyperparameter configurations
    # You can expand this grid as needed.
    # -----------------------------------------------------------------

    candidate_configs = [
        StyleModelConfig(
            K=3,
            alpha=0.5,
            alpha_w=jnp.asarray(make_alpha_w(3)),
            tau_beta=1.0,
            tau_sigma=1.0,
        ),
        StyleModelConfig(
            K=5,
            alpha=0.5,
            alpha_w=jnp.asarray(make_alpha_w(5)),
            tau_beta=1.0,
            tau_sigma=1.0,
        ),
        StyleModelConfig(
            K=7,
            alpha=0.5,
            alpha_w=jnp.asarray(make_alpha_w(7)),
            tau_beta=1.0,
            tau_sigma=1.0,
        ),
    ]


    # -----------------------------------------------------------------
    # Cross-validation on train_data (validation_data is untouched here)
    # -----------------------------------------------------------------
    cv_results = cross_validate_on_train(
        train_df,
        candidate_configs=candidate_configs,
        n_splits=3,      # small N in this dataset; you can change to 5 if larger
        num_steps=1500,  # fewer steps for CV runs
        base_seed=42,
    )

    # Pick best config by lowest mean MSE
    best = min(cv_results, key=lambda r: r.mean_mse)
    print("\n[CV] Summary of configs:")
    for r in cv_results:
        print(
            f"  Config K={r.config.K}, alpha={r.config.alpha}: "
            f"mean MSE={r.mean_mse:.4f} ± {r.std_mse:.4f}"
        )

    print(
        f"\n[CV] Best config: K={best.config.K}, "
        f"alpha={best.config.alpha}, MSE={best.mean_mse:.4f}"
    )

    # -----------------------------------------------------------------
    # Train final model on full train_data, evaluate on validation_data
    # -----------------------------------------------------------------
    train_final_and_evaluate(
        train_df=train_df,
        val_df=val_df,
        best_config=best.config,
        num_steps=3000,
        rng_seed=2025,
        out_dir="./outputs_style_model",
    )
