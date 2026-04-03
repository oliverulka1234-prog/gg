from __future__ import annotations

import pandas as pd

RESULT_POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


def build_features(df: pd.DataFrame, lookback: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Build rolling-team-form and odds features.

    Inspired by:
    - rolling-form features (FootballBettingModel)
    - rank/points and simple bookmaker value setup (SportsBet)
    """

    games = df.copy().reset_index(drop=True)
    games["home_points"] = games["FTR"].map(lambda r: RESULT_POINTS[r][0])
    games["away_points"] = games["FTR"].map(lambda r: RESULT_POINTS[r][1])

    team_history: dict[str, list[int]] = {}
    feature_rows: list[dict[str, float]] = []

    for _, row in games.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        home_hist = team_history.get(home, [])[-lookback:]
        away_hist = team_history.get(away, [])[-lookback:]

        home_form = sum(home_hist) / (3 * len(home_hist)) if home_hist else 0.5
        away_form = sum(away_hist) / (3 * len(away_hist)) if away_hist else 0.5

        feat = {
            "home_form": home_form,
            "away_form": away_form,
            "form_delta": home_form - away_form,
            "has_b365": float(all(c in games.columns for c in ["B365H", "B365D", "B365A"])),
            "odds_home": float(row.get("B365H", 2.5) or 2.5),
            "odds_draw": float(row.get("B365D", 3.1) or 3.1),
            "odds_away": float(row.get("B365A", 2.9) or 2.9),
        }
        feature_rows.append(feat)

        team_history.setdefault(home, []).append(int(row["home_points"]))
        team_history.setdefault(away, []).append(int(row["away_points"]))

    X = pd.DataFrame(feature_rows)
    y = games["FTR"].astype(str)
    return X, y
