import numpy as np
import pandas as pd

from pairs_trading.backtest import backtest_strategy


def test_daily_pnl_reconciles_with_trade_pnl():
    index = pd.bdate_range("2024-01-01", periods=60)
    # Deterministic spread with a low excursion followed by reversion.
    spread = pd.Series(np.r_[np.zeros(25), np.linspace(-0.08, 0.08, 15), np.zeros(20)], index=index)
    p1 = pd.Series(np.linspace(100, 105, len(index)), index=index)
    p2 = pd.Series(np.linspace(100, 101, len(index)), index=index)
    prices = pd.DataFrame({"AAA": p1, "BBB": p2})
    result = backtest_strategy(
        spread, prices, "AAA", "BBB", beta=0.8,
        entry_threshold=1.5, exit_threshold=0.0, window=10,
    )
    trades_total = result["trades_df"]["pnl"].sum() if len(result["trades_df"]) else 0.0
    curve_total = result["cumulative_pnl"].iloc[-1]
    assert np.isclose(trades_total, curve_total, atol=1e-8)
