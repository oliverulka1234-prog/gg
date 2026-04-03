import pandas as pd

from football_betting_machine.backtest import run_backtest
from football_betting_machine.features import build_features


def _sample_matches(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "Date": f"2024-01-{(i % 28) + 1:02d}",
                "HomeTeam": f"H{i % 6}",
                "AwayTeam": f"A{(i + 1) % 6}",
                "FTR": ["H", "D", "A"][i % 3],
                "B365H": 2.2,
                "B365D": 3.2,
                "B365A": 3.1,
            }
        )
    return pd.DataFrame(rows)


def test_build_features_shape():
    X, y = build_features(_sample_matches(), lookback=3)
    assert len(X) == len(y) == 30
    assert {"home_form", "away_form", "form_delta"}.issubset(X.columns)


def test_run_backtest_smoke():
    result = run_backtest(_sample_matches(), lookback=3, train_fraction=0.7)
    assert isinstance(result.roi, float)
    assert result.final_bankroll > 0
    assert len(result.trajectory) >= 1
