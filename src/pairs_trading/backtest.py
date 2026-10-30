"""Rolling z-score pairs-trading backtest."""
from __future__ import annotations

import pandas as pd


def backtest_strategy(
    spread_data: pd.Series,
    prices_data: pd.DataFrame,
    stock1: str,
    stock2: str,
    beta: float,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    position_size: float = 100_000.0,
    window: int = 20,
) -> dict:
    """Run the notebook strategy and return trades and daily P&L series.

    Historical replication note: the source notebook passes the OU
    mean-reversion coefficient (posterior beta) as the second-leg share
    multiplier. This is not a separately estimated cointegration hedge
    ratio; it is retained here to reproduce the archived experiment.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if position_size <= 0:
        raise ValueError("position_size must be positive")
    if stock1 not in prices_data or stock2 not in prices_data:
        raise KeyError("Both ticker columns must be present in prices_data")

    prices = prices_data.loc[spread_data.index, [stock1, stock2]].copy()
    valid = spread_data.notna() & prices[stock1].notna() & prices[stock2].notna()
    spread = spread_data.loc[valid]
    prices = prices.loc[valid]
    rolling_mean = spread.shift(1).rolling(window=window).mean()
    rolling_std = spread.shift(1).rolling(window=window).std()
    z_score = (spread - rolling_mean) / rolling_std

    trades = []
    position = None
    daily_pnl = []
    previous_trade_pnl = 0.0
    entry_date = entry_z = entry_price_1 = entry_price_2 = None
    shares_1 = shares_2 = None

    for i, current_date in enumerate(spread.index):
        price1 = float(prices.iloc[i][stock1])
        price2 = float(prices.iloc[i][stock2])
        z = z_score.iloc[i]
        if pd.isna(z):
            daily_pnl.append(0.0)
            continue

        if position is None:
            if z < -entry_threshold:
                position = "long"
            elif z > entry_threshold:
                position = "short"
            else:
                daily_pnl.append(0.0)
                continue

            entry_date = current_date
            entry_z = float(z)
            entry_price_1, entry_price_2 = price1, price2
            shares_1 = position_size / (price1 + abs(beta) * price2)
            shares_2 = abs(beta) * shares_1
            previous_trade_pnl = 0.0
            daily_pnl.append(0.0)
            continue

        if position == "long":
            current_trade_pnl = (
                shares_1 * (price1 - entry_price_1)
                + shares_2 * (entry_price_2 - price2)
            )
            daily_pnl.append(current_trade_pnl - previous_trade_pnl)
            previous_trade_pnl = current_trade_pnl
            should_exit = z >= exit_threshold
        else:
            current_trade_pnl = (
                shares_1 * (entry_price_1 - price1)
                + shares_2 * (price2 - entry_price_2)
            )
            daily_pnl.append(current_trade_pnl - previous_trade_pnl)
            previous_trade_pnl = current_trade_pnl
            should_exit = z <= -exit_threshold

        if should_exit:
            trades.append({
                "entry_date": entry_date, "exit_date": current_date,
                "position": position.upper(), "entry_z": entry_z,
                "exit_z": float(z), "entry_price_1": entry_price_1,
                "exit_price_1": price1, "entry_price_2": entry_price_2,
                "exit_price_2": price2, "shares_1": shares_1,
                "shares_2": shares_2, "pnl": current_trade_pnl,
                "duration_days": (current_date - entry_date).days,
            })
            position = None
            previous_trade_pnl = 0.0

    # Liquidate a position still open at the end of the sample.
    if position is not None:
        final_date = spread.index[-1]
        final_price1 = float(prices.iloc[-1][stock1])
        final_price2 = float(prices.iloc[-1][stock2])
        final_z = float(z_score.iloc[-1])
        if position == "long":
            final_pnl = (
                shares_1 * (final_price1 - entry_price_1)
                + shares_2 * (entry_price_2 - final_price2)
            )
        else:
            final_pnl = (
                shares_1 * (entry_price_1 - final_price1)
                + shares_2 * (final_price2 - entry_price_2)
            )
        daily_pnl[-1] += final_pnl - previous_trade_pnl
        trades.append({
            "entry_date": entry_date, "exit_date": final_date,
            "position": position.upper(), "entry_z": entry_z,
            "exit_z": final_z, "entry_price_1": entry_price_1,
            "exit_price_1": final_price1, "entry_price_2": entry_price_2,
            "exit_price_2": final_price2, "shares_1": shares_1,
            "shares_2": shares_2, "pnl": final_pnl,
            "duration_days": (final_date - entry_date).days,
        })

    daily = pd.Series(daily_pnl, index=spread.index, name="daily_pnl")
    cumulative = daily.cumsum().rename("cumulative_pnl")
    trade_columns = [
        "entry_date", "exit_date", "position", "entry_z", "exit_z",
        "entry_price_1", "exit_price_1", "entry_price_2", "exit_price_2",
        "shares_1", "shares_2", "pnl", "duration_days",
    ]
    trades_df = pd.DataFrame(trades, columns=trade_columns)
    return {
        "trades": trades, "trades_df": trades_df,
        "daily_pnl": daily, "cumulative_pnl": cumulative,
        "z_score": z_score, "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
    }
