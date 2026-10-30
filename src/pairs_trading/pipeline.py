"""End-to-end experiment runner."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

from .config import (
    PAIRS, START_DATE, END_DATE, SIGNIFICANCE_LEVEL, TRAIN_FRACTION,
    ROLLING_WINDOW, ENTRY_THRESHOLD, EXIT_THRESHOLD, POSITION_SIZE,
    MCMC_DRAWS, MCMC_TUNE, MCMC_CHAINS, RANDOM_SEED, INITIAL_CAPITAL,
)
from .data import download_pair_data
from .screening import screen_pairs, choose_pair
from .bayesian_ou import make_spread, fit_ou_mcmc
from .backtest import backtest_strategy
from .performance import performance_metrics
from .plotting import plot_pair_prices, plot_ou_diagnostics, plot_backtest


def run_pipeline(
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: str | Path = "results/generated",
    draws: int = MCMC_DRAWS,
    tune: int = MCMC_TUNE,
    chains: int = MCMC_CHAINS,
    cores: int = 1,
) -> dict:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(PAIRS)} pairs from {start} to {end} ...")
    pair_data = download_pair_data(PAIRS, start, end)
    screen = screen_pairs(pair_data, alpha=SIGNIFICANCE_LEVEL)
    screen.to_csv(output_dir / "pair_screening.csv", index=False)
    selected_name, selected_prices = choose_pair(pair_data, screen)
    ticker1, ticker2 = selected_name.split("_")
    print(f"Selected pair: {selected_name}")

    spread = make_spread(selected_prices)
    split = int(len(spread) * TRAIN_FRACTION)
    train_spread = spread.iloc[:split]
    test_spread = spread.iloc[split:]
    test_prices = selected_prices.iloc[split:]

    # For compatibility with the source notebook, the original experiment
    # estimated the OU model on the complete selected spread before splitting.
    # This is retained for reproducibility, but it introduces look-ahead in
    # an out-of-sample evaluation and should be changed for research use.
    model, trace, posterior_summary = fit_ou_mcmc(
        spread, draws=draws, tune=tune, chains=chains, cores=cores,
        random_seed=RANDOM_SEED,
    )
    posterior_summary.to_csv(output_dir / "posterior_summary.csv")
    ou_beta = float(trace.posterior["beta"].mean().values)
    posterior_mu = float(trace.posterior["mu"].mean().values)
    posterior_sigma = float(trace.posterior["sigma"].mean().values)
    half_life = float(trace.posterior["half_life"].mean().values)

    backtest = backtest_strategy(
        spread_data=test_spread, prices_data=test_prices,
        stock1=ticker1, stock2=ticker2, beta=ou_beta,
        entry_threshold=ENTRY_THRESHOLD, exit_threshold=EXIT_THRESHOLD,
        position_size=POSITION_SIZE, window=ROLLING_WINDOW,
    )
    metrics = performance_metrics(backtest, initial_capital=INITIAL_CAPITAL)
    metrics.update({
        "selected_pair": selected_name,
        "start_date": str(selected_prices.index[0].date()),
        "end_date": str(selected_prices.index[-1].date()),
        "observations": int(len(selected_prices)),
        "train_observations": int(len(train_spread)),
        "test_observations": int(len(test_spread)),
        "ou_beta_posterior_mean": ou_beta,
        "ou_mu_posterior_mean": posterior_mu,
        "ou_sigma_posterior_mean": posterior_sigma,
        "half_life_posterior_mean_days": half_life,
    })
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True))
    backtest["trades_df"].to_csv(output_dir / "trades.csv", index=False)
    pd.concat(
        [backtest["daily_pnl"], backtest["cumulative_pnl"], backtest["z_score"]], axis=1
    ).to_csv(output_dir / "daily_backtest.csv")
    selected_prices.to_csv(output_dir / f"{selected_name}_prices.csv")

    plot_pair_prices(selected_prices, figures_dir / "pair_price_and_spread.png")
    plot_ou_diagnostics(trace, spread, figures_dir)
    plot_backtest(test_spread, backtest, figures_dir / "backtest_summary.png")

    print("\nBacktest summary")
    for key in ["trade_count", "win_rate_pct", "total_pnl", "profit_factor", "sharpe_ratio", "max_drawdown"]:
        print(f"{key}: {metrics[key]}")
    return {
        "selected_pair": selected_name, "screening": screen,
        "prices": selected_prices, "spread": spread,
        "train_spread": train_spread, "test_spread": test_spread,
        "model": model, "trace": trace, "posterior_summary": posterior_summary,
        "backtest": backtest, "metrics": metrics,
    }
