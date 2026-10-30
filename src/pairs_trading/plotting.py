"""Plotting helpers for the runnable pipeline."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pandas as pd


def plot_pair_prices(prices: pd.DataFrame, output: str | Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    prices.plot(ax=axes[0], linewidth=1.4)
    axes[0].set_title("Pair price history")
    axes[0].set_ylabel("Adjusted price")
    axes[0].grid(alpha=0.25)
    spread = np.log(prices.iloc[:, 0]) - np.log(prices.iloc[:, 1])
    axes[1].plot(spread.index, spread, linewidth=1.2)
    axes[1].set_title("Log-price spread")
    axes[1].set_ylabel("log(P1) - log(P2)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ou_diagnostics(trace, spread: pd.Series, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    fig = az.plot_trace(trace, var_names=["beta", "mu", "sigma", "half_life"], figsize=(13, 8))
    fig.ravel()[0].figure.tight_layout()
    fig.ravel()[0].figure.savefig(output_dir / "mcmc_trace.png", dpi=160, bbox_inches="tight")
    plt.close(fig.ravel()[0].figure)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name, label in zip(
        axes.flat, ["beta", "mu", "sigma", "half_life"],
        ["Mean-reversion speed β", "Long-run mean μ", "Innovation scale σ", "Half-life (days)"],
    ):
        values = trace.posterior[name].values.reshape(-1)
        ax.hist(values, bins=45, alpha=0.8, edgecolor="white")
        ax.axvline(values.mean(), linestyle="--", linewidth=1.5, label=f"Mean: {values.mean():.4g}")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(spread: pd.Series, results: dict, output: str | Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    z = results["z_score"]
    axes[0].plot(spread.index, spread, label="Spread")
    axes[0].plot(results["rolling_mean"].index, results["rolling_mean"], linestyle="--", label="Rolling mean")
    axes[0].set_title("Test-period spread")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(z.index, z, label="Rolling z-score")
    axes[1].axhline(2, linestyle="--", linewidth=1)
    axes[1].axhline(-2, linestyle="--", linewidth=1)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_title("Entry/exit signal")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(results["cumulative_pnl"].index, results["cumulative_pnl"], label="Cumulative P&L")
    axes[2].set_title("Cumulative P&L (USD)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
