"""Project configuration and experiment defaults."""

PAIRS = [
    ("AAPL", "MSFT"), ("JPM", "BAC"), ("GLD", "SLV"),
    ("KO", "PEP"), ("GS", "MS"), ("XOM", "CVX"),
    ("JNJ", "PFE"), ("BA", "RTX"), ("MCD", "CMG"),
    ("XLU", "XLE"), ("XLV", "XLY"), ("IVV", "VOO"),
    ("EEM", "VWO"),
]

# Dates from the archived notebook run. Yahoo Finance's end date is exclusive.
START_DATE = "2023-03-15"
END_DATE = "2026-09-12"
SIGNIFICANCE_LEVEL = 0.05
TRAIN_FRACTION = 0.75
ROLLING_WINDOW = 20
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.0
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 100_000.0
MCMC_DRAWS = 2_000
MCMC_TUNE = 1_000
MCMC_CHAINS = 4
RANDOM_SEED = 42
