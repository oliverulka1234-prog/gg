from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_features
from .model import OutcomeModel


@dataclass
class BacktestResult:
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    roi: float
    final_bankroll: float
    bets_placed: int
    trajectory: list[float]


def _kelly_fraction(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    edge = prob * (b + 1.0) - 1.0
    return max(0.0, edge / b)


def run_backtest(
    matches: pd.DataFrame,
    *,
    lookback: int = 5,
    estimator: str = "rf",
    train_fraction: float = 0.8,
    initial_bankroll: float = 1000.0,
    max_kelly_fraction: float = 0.1,
    min_edge: float = 0.02,
) -> BacktestResult:
    X, y = build_features(matches, lookback=lookback)

    split_idx = max(1, int(len(X) * train_fraction))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = OutcomeModel(estimator=estimator).fit(X_train, y_train)
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test) if len(X_test) > 0 else {"accuracy": 0.0, "log_loss": 0.0, "avg_confidence": 0.0}

    classes = model.classes_
    class_idx = {c: i for i, c in enumerate(classes)}

    bankroll = initial_bankroll
    trajectory = [bankroll]
    total_staked = 0.0
    total_profit = 0.0
    bets = 0

    if len(X_test) > 0:
        probs = model.predict_proba(X_test)
        test_matches = matches.iloc[split_idx:].reset_index(drop=True)

        for i, prob_vec in enumerate(probs):
            p = {c: float(prob_vec[class_idx[c]]) for c in classes}
            odds = {
                "H": float(test_matches.iloc[i].get("B365H", 2.5) or 2.5),
                "D": float(test_matches.iloc[i].get("B365D", 3.1) or 3.1),
                "A": float(test_matches.iloc[i].get("B365A", 2.9) or 2.9),
            }

            edges = {c: p.get(c, 0.0) * odds[c] - 1.0 for c in ["H", "D", "A"]}
            side = max(edges, key=edges.get)
            if edges[side] < min_edge:
                trajectory.append(bankroll)
                continue

            f = min(_kelly_fraction(p.get(side, 0.0), odds[side]), max_kelly_fraction)
            stake = bankroll * f
            if stake <= 0:
                trajectory.append(bankroll)
                continue

            bets += 1
            total_staked += stake
            actual = y_test.iloc[i]
            profit = stake * (odds[side] - 1.0) if actual == side else -stake
            total_profit += profit
            bankroll += profit
            trajectory.append(bankroll)

    roi = (total_profit / total_staked) if total_staked else 0.0
    return BacktestResult(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        roi=float(roi),
        final_bankroll=float(bankroll),
        bets_placed=bets,
        trajectory=[float(v) for v in trajectory],
    )
