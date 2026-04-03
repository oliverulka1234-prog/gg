from __future__ import annotations

import argparse
import json

from .backtest import run_backtest
from .data import load_matches


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI football betting machine")
    p.add_argument("--csv", help="Path to local CSV (football-data schema)")
    p.add_argument("--country-code", default="E0", help="football-data league code, e.g. E0")
    p.add_argument("--season", default="2324", help="football-data season folder, e.g. 2324")
    p.add_argument("--estimator", default="rf", choices=["rf", "xgb"])
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--initial-bankroll", type=float, default=1000.0)
    p.add_argument("--max-kelly", type=float, default=0.1)
    p.add_argument("--min-edge", type=float, default=0.02)
    return p


def main() -> None:
    args = build_parser().parse_args()

    matches = load_matches(args.csv, country_code=args.country_code, season=args.season)
    result = run_backtest(
        matches,
        lookback=args.lookback,
        estimator=args.estimator,
        train_fraction=args.train_fraction,
        initial_bankroll=args.initial_bankroll,
        max_kelly_fraction=args.max_kelly,
        min_edge=args.min_edge,
    )

    print(
        json.dumps(
            {
                "train": result.train_metrics,
                "test": result.test_metrics,
                "bets_placed": result.bets_placed,
                "roi": result.roi,
                "final_bankroll": result.final_bankroll,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
