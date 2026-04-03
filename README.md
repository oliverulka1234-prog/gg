# AI Football Betting Machine

This repository implements an end-to-end **AI football betting machine** inspired by:

- https://github.com/clemsage/SportsBet
- https://github.com/qwyt/FootballBettingModel
- Ideas commonly seen in the https://github.com/topics/sports-prediction ecosystem

## What was implemented

The project combines the strongest practical elements from those references:

1. **SportsBet-style workflow**
   - Simple command-line pipeline.
   - Football-data style CSV ingestion (`Date`, `HomeTeam`, `AwayTeam`, `FTR`, bookmaker odds).
   - End-to-end: load data → train model → generate probabilities → simulate betting ROI.

2. **FootballBettingModel-style modeling ideas**
   - Rolling team-form features (time-aware history lookback).
   - Emphasis on probability quality (`log_loss`, confidence), not only hard-label accuracy.
   - Value-based bet filtering using expected edge.

3. **Sports-prediction topic style betting logic**
   - Kelly criterion position sizing with configurable cap.
   - Value betting trigger (`p * odds - 1 > threshold`).
   - Backtest outputs for bankroll, ROI, and bet count.

## Install

```bash
pip install -e .
```

Optional XGBoost support:

```bash
pip install -e .[xgboost]
```

## Usage

### 1) Use remote football-data CSV

```bash
football-bet --country-code E0 --season 2324 --estimator rf
```

### 2) Use your own local CSV

```bash
football-bet --csv /path/to/matches.csv --estimator xgb --lookback 8 --min-edge 0.03
```

## Output

The CLI prints JSON with:

- train/test metrics (`accuracy`, `log_loss`, `avg_confidence`)
- `bets_placed`
- `roi`
- `final_bankroll`

## Notes

- This is for **research/education**, not financial advice.
- Betting is risky; historical backtests do not guarantee future returns.
