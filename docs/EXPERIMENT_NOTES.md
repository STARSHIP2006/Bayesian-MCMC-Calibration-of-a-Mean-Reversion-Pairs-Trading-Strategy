# Experiment notes

- Data source: Yahoo Finance via `yfinance`.
- Instrument universe: 13 predefined equity / ETF pairs in `src/pairs_trading/config.py`.
- Source notebook data interval: 15 March 2023 to 11 September 2026 (last observed session).
- Pair eligibility: ADF p-value for the simple log-price spread < 0.05 and Engle–Granger p-value < 0.05.
- Bayesian model: discrete-time OU-style regression, four NUTS chains, 1,000 tuning iterations and 2,000 retained draws per chain.
- Backtest: 75/25 chronological split, 20-session rolling z-score, ±2 entry thresholds, zero exit threshold, USD 100,000 notional.
- Results in README refer to the saved notebook execution only.

See the README's research-limitations section before interpreting reported backtest performance.
