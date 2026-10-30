"""Bayesian calibration of a discrete-time OU-style mean-reversion model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def make_spread(prices: pd.DataFrame) -> pd.Series:
    """Create the notebook's log-price spread: log(P1) - log(P2)."""
    if prices.shape[1] != 2:
        raise ValueError("Expected exactly two price columns.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive before taking logs.")
    return (np.log(prices.iloc[:, 0]) - np.log(prices.iloc[:, 1])).rename("spread")


def prepare_ou_observations(spread: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return first differences and the corresponding lagged spread values."""
    values = spread.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError("At least three spread observations are required.")
    return np.diff(values), values[:-1]


def fit_ou_mcmc(
    spread: pd.Series,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 42,
    target_accept: float = 0.95,
):
    """Fit ΔS_t = β(μ − S_(t−1)) + ε_t with Gaussian innovations.

    Returns the InferenceData trace and an ArviZ posterior summary.
    """
    changes, lagged_spread = prepare_ou_observations(spread)
    with pm.Model() as model:
        beta = pm.Beta("beta", alpha=2, beta=5)
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        pm.Deterministic("half_life", np.log(2) / (beta + 1e-6))
        expected_change = beta * (mu - lagged_spread)
        pm.Normal("obs", mu=expected_change, sigma=sigma, observed=changes)
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            random_seed=random_seed, return_inferencedata=True,
            target_accept=target_accept, progressbar=True,
        )
    summary = az.summary(trace, var_names=["beta", "mu", "sigma", "half_life"])
    return model, trace, summary
