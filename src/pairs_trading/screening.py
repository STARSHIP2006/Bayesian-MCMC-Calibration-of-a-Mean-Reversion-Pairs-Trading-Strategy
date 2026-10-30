"""ADF stationarity and Engle-Granger cointegration screening."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def screen_pairs(
    pair_data: dict[str, pd.DataFrame], alpha: float = 0.05
) -> pd.DataFrame:
    """Screen pairs using ADF on the simple log-price difference and EG test."""
    rows = []
    for pair_name, prices in pair_data.items():
        ticker1, ticker2 = pair_name.split("_", maxsplit=1)
        log_prices = np.log(prices[[ticker1, ticker2]])
        spread = log_prices[ticker1] - log_prices[ticker2]
        adf_stat, adf_pvalue, *_ = adfuller(spread.dropna(), autolag="AIC")
        coint_stat, coint_pvalue, _ = coint(
            log_prices[ticker1].values, log_prices[ticker2].values
        )
        rows.append({
            "pair": pair_name,
            "observations": int(len(prices)),
            "adf_statistic": float(adf_stat),
            "adf_pvalue": float(adf_pvalue),
            "coint_statistic": float(coint_stat),
            "coint_pvalue": float(coint_pvalue),
            "spread_stationary": bool(adf_pvalue < alpha),
            "cointegrated": bool(coint_pvalue < alpha),
            "selected": bool(adf_pvalue < alpha and coint_pvalue < alpha),
        })
    return pd.DataFrame(rows).sort_values(
        ["selected", "coint_pvalue", "adf_pvalue"], ascending=[False, True, True]
    ).reset_index(drop=True)


def choose_pair(pair_data: dict[str, pd.DataFrame], screen: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Choose the first valid pair in the original candidate-list order."""
    valid = set(screen.loc[screen["selected"], "pair"])
    for name, prices in pair_data.items():
        if name in valid:
            return name, prices.copy()
    raise RuntimeError("No pair passed both the ADF and cointegration screens.")
