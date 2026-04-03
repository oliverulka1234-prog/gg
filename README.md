# AI Football Betting Machine

This repository implements an end-to-end **AI football betting machine** inspired by:

- https://github.com/clemsage/SportsBet
- https://github.com/qwyt/FootballBettingModel
- Ideas commonly seen in the https://github.com/topics/sports-prediction ecosystem

## What was implemented

The project combines practical elements from those references:

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

## How to use this on a phone

You have 3 realistic options:

### Option A (Recommended): Replit / GitHub Codespaces in mobile browser

1. Fork this repo on GitHub.
2. Open the repo in **Codespaces** (or import to Replit).
3. In terminal, run:

```bash
pip install -e .
football-bet --country-code E0 --season 2324
```

4. Copy the JSON output from terminal.

Why this is easiest: no local setup, works on iPhone/Android browser.

### Option B: Android with Termux

1. Install **Termux** from F-Droid.
2. In Termux:

```bash
pkg update && pkg upgrade -y
pkg install git python -y
git clone https://github.com/<your-username>/<your-fork>.git
cd <your-fork>
pip install -e .
football-bet --country-code E0 --season 2324
```

### Option C: iPhone/iPad with a remote Linux box

1. Use an SSH app (Blink/Shelly/Termius).
2. SSH into a VPS or home Linux machine.
3. Run the same install + command steps there.

> iOS does not provide a full native local Python shell like Termux, so remote execution is usually the practical path.

## Output

The CLI prints JSON with:

- train/test metrics (`accuracy`, `log_loss`, `avg_confidence`)
- `bets_placed`
- `roi`
- `final_bankroll`

## Notes

- This is for **research/education**, not financial advice.
- Betting is risky; historical backtests do not guarantee future returns.

## Deploy to Netlify

A mobile-friendly UI is included in `web/` so you can upload CSV files and run a browser backtest.

1. Push this repo to your GitHub account.
2. Go to Netlify and click **Add new site → Import an existing project**.
3. Pick your repo.
4. Build settings:
   - Build command: *(leave empty)*
   - Publish directory: `web`
5. Deploy.

This repo includes `netlify.toml`, so Netlify should auto-detect publish settings.

### Local preview

```bash
python -m http.server 8080 -d web
```

Then open `http://localhost:8080`.
