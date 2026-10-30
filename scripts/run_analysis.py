#!/usr/bin/env python3
"""Run the end-to-end pairs-trading experiment."""
import argparse
from pairs_trading.pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian MCMC pairs trading analysis")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Exclusive end date (YYYY-MM-DD)")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--output", default="results/generated")
    args = parser.parse_args()
    kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains,
              "cores": args.cores, "output_dir": args.output}
    if args.start is not None: kwargs["start"] = args.start
    if args.end is not None: kwargs["end"] = args.end
    run_pipeline(**kwargs)

if __name__ == "__main__":
    main()
