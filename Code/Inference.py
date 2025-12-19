# Inference.py
'''
Author: Kangyu Zhao
Generic inference wrapper for style-topic model using NumPyro SVI.
Last Updated Date: 2025-12-02
'''

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive


class StyleTopicInference:
    """
    Generic inference wrapper for the style-topic model.

    This class assumes you already:
    - Learned global parameters (topics, biases, etc.) on the training set.
    - Have a NumPyro model and guide that:
        * take X_new as an argument,
        * also accept any fixed global parameters (K, beta_fixed, ...).

    The goal is to run conditional SVI to infer player-level latent variables
    (e.g. eta, theta) for new players, keeping global parameters fixed.
    """

    def __init__(
        self,
        model,
        guide,
        static_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: Optional[numpyro.optim.Optimizer] = None,
        loss: Optional[Any] = None,
        num_steps: int = 2000,
        rng_seed: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        model : callable
            NumPyro model function. Signature should be roughly:
                model(X_new, **static_kwargs)
            where static_kwargs contains K, beta_fixed, gamma_fixed, etc.
        guide : callable
            NumPyro guide function with the same signature as `model`.
        static_kwargs : dict, optional
            Dictionary of fixed global parameters passed to model/guide,
            e.g. {
                "K": K,
                "beta_fixed": beta_fixed,
                "gamma_fixed": gamma_fixed,
                "log_rate_bias_fixed": log_rate_bias_fixed,
                "gate_bias_fixed": gate_bias_fixed,
            }
        optimizer : numpyro.optim.Optimizer, optional
            Optimizer for SVI. If None, use Adam with learning rate 1e-2.
        loss : numpyro.infer.ELBO, optional
            ELBO loss object. If None, use Trace_ELBO().
        num_steps : int
            Number of SVI optimization steps.
        rng_seed : int
            Random seed for PRNGKey.
        """
        self.model = model
        self.guide = guide
        self.static_kwargs = static_kwargs or {}
        self.num_steps = num_steps

        if optimizer is None:
            optimizer = numpyro.optim.Adam(step_size=1e-2)
        self.optimizer = optimizer

        if loss is None:
            loss = Trace_ELBO()
        self.loss = loss

        self.svi = SVI(self.model, self.guide, self.optimizer, self.loss)

        # RNG state
        self.rng_key = jax.random.PRNGKey(rng_seed)

        # Store last SVI state / params for later inspection
        self._last_svi_state = None
        self._last_params = None

    # ------------------------------------------------------------------
    # Core SVI fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X_new,
        num_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run SVI to infer variational parameters for the new data X_new.

        Parameters
        ----------
        X_new : array-like
            New observations, shape (N_new, P).
        num_steps : int, optional
            Override default number of SVI steps for this call.
        verbose : bool
            If True, print loss every ~10% of the total iterations.

        Returns
        -------
        params : dict
            Dictionary of learned variational parameters.
        """
        X_new = jnp.asarray(X_new)
        n_steps = num_steps if num_steps is not None else self.num_steps

        # Split RNG for init
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Initialize SVI state
        svi_state = self.svi.init(
            subkey,
            X_new=X_new,
            **self.static_kwargs,
        )

        # Define one SVI update step (JIT-compiled)
        @jax.jit
        def svi_step(i, state):
            state, loss = self.svi.update(
                state,
                X_new=X_new,
                **self.static_kwargs,
            )
            return state, loss

        losses = []

        # Run optimization loop in Python so we can optionally log losses
        for i in range(n_steps):
            svi_state, loss = svi_step(i, svi_state)
            losses.append(loss)

            if verbose and ((i + 1) % max(1, n_steps // 10) == 0):
                # Convert to float for printing
                print(f"[SVI] step {i + 1}/{n_steps}, loss = {float(loss):.4f}")

        params = self.svi.get_params(svi_state)
        self._last_svi_state = svi_state
        self._last_params = params

        return params

    # ------------------------------------------------------------------
    # Posterior over latent variables (e.g. eta)
    # ------------------------------------------------------------------
    def sample_latent_posterior(
        self,
        X_new,
        num_samples: int = 1000,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample from the approximate posterior of latent variables using the guide.

        This is typically used to get posterior samples of player-level latents
        such as "eta" and "theta".

        Parameters
        ----------
        X_new : array-like
            New observations, shape (N_new, P).
        num_samples : int
            Number of posterior samples to draw from the guide.
        params : dict, optional
            Variational parameters. If None, use the last fitted params.

        Returns
        -------
        samples : dict
            Dictionary of latent samples. Keys are latent names (e.g. "eta"),
            values are arrays of shape (num_samples, ...) .
        """
        X_new = jnp.asarray(X_new)
        if params is None:
            if self._last_params is None:
                raise RuntimeError(
                    "No cached params. Call `fit` first or pass `params` explicitly."
                )
            params = self._last_params

        self.rng_key, subkey = jax.random.split(self.rng_key)

        guide_predictive = Predictive(
            self.guide,
            params=params,
            num_samples=num_samples,
        )
        samples = guide_predictive(
            subkey,
            X_new=X_new,
            **self.static_kwargs,
        )
        return samples

    def get_eta_posterior_mean(
        self,
        X_new,
        num_samples: int = 1000,
        params: Optional[Dict[str, Any]] = None,
        eta_name: str = "eta",
    ) -> jnp.ndarray:
        """
        Compute posterior mean of eta for the new players.

        Parameters
        ----------
        X_new : array-like
            New observations, shape (N_new, P).
        num_samples : int
            Number of posterior samples to draw from the guide.
        params : dict, optional
            Variational parameters to use. If None, use last fitted params.
        eta_name : str
            Name of the latent variable in the model/guide corresponding to eta.

        Returns
        -------
        eta_mean : array
            Posterior mean of eta, shape (N_new, K) (assuming eta has that shape).
        """
        samples = self.sample_latent_posterior(
            X_new=X_new,
            num_samples=num_samples,
            params=params,
        )
        if eta_name not in samples:
            raise KeyError(
                f"Latent '{eta_name}' not found in guide samples. "
                "Check your model/guide or eta_name."
            )

        eta_samples = samples[eta_name]  # shape: (num_samples, N_new, K)
        eta_mean = jnp.mean(eta_samples, axis=0)
        return eta_mean

    # ------------------------------------------------------------------
    # Posterior predictive for counts / observations
    # ------------------------------------------------------------------
    def sample_posterior_predictive(
        self,
        X_new,
        num_samples: int = 1000,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Draw samples from the posterior predictive distribution.

        This uses the model and the learned variational parameters to
        generate new observations (e.g. predicted counts or rates).

        Parameters
        ----------
        X_new : array-like
            New observations, shape (N_new, P). For purely generative
            use-cases you may pass a placeholder or zeros if your model
            does not condition on X_new directly.
        num_samples : int
            Number of posterior predictive samples.
        params : dict, optional
            Variational parameters. If None, use last fitted params.

        Returns
        -------
        pred_samples : dict
            Dictionary of posterior predictive samples (e.g. "X_new", "lambda").
        """
        X_new = jnp.asarray(X_new)
        if params is None:
            if self._last_params is None:
                raise RuntimeError(
                    "No cached params. Call `fit` first or pass `params` explicitly."
                )
            params = self._last_params

        self.rng_key, subkey = jax.random.split(self.rng_key)

        predictive = Predictive(
            self.model,
            params=params,
            num_samples=num_samples,
        )
        pred_samples = predictive(
            subkey,
            X_new=X_new,
            **self.static_kwargs,
        )
        return pred_samples
