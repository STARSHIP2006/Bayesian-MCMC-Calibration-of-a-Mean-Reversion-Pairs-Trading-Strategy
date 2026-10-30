"""Backtest performance and risk statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(
    backtest_results: dict, initial_capital: float = 100_000.0
) -> dict[str, float | int]:
    trades = backtest_results["trades_df"]
    daily_pnl = backtest_results["daily_pnl"]
    gross_profit = float(trades.loc[trades.pnl > 0, "pnl"].sum()) if len(trades) else 0.0
    gross_loss = float(trades.loc[trades.pnl < 0, "pnl"].sum()) if len(trades) else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss else float("inf")
    daily_returns = daily_pnl / initial_capital
    volatility = float(daily_returns.std())
    sharpe = (
        float(daily_returns.mean() / volatility * np.sqrt(252))
        if volatility > 0 else float("nan")
    )
    equity = initial_capital + backtest_results["cumulative_pnl"]
    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    peak = float(equity.cummax().max()) if len(equity) else initial_capital
    max_drawdown_pct = max_drawdown / peak * 100 if peak else float("nan")
    trade_count = int(len(trades))
    wins = int((trades.pnl > 0).sum()) if trade_count else 0
    return {
        "trade_count": trade_count,
        "winning_trades": wins,
        "losing_trades": trade_count - wins,
        "win_rate_pct": wins / trade_count * 100 if trade_count else 0.0,
        "total_pnl": float(trades.pnl.sum()) if trade_count else 0.0,
        "average_pnl": float(trades.pnl.mean()) if trade_count else 0.0,
        "best_trade": float(trades.pnl.max()) if trade_count else 0.0,
        "worst_trade": float(trades.pnl.min()) if trade_count else 0.0,
        "average_duration_days": float(trades.duration_days.mean()) if trade_count else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": float(max_drawdown_pct),
        "pnl_reconciliation_difference": float(
            (trades.pnl.sum() if trade_count else 0.0)
            - (backtest_results["cumulative_pnl"].iloc[-1] if len(daily_pnl) else 0.0)
        ),
    }
