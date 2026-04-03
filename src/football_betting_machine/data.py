from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"


def load_matches(csv_path: str | None = None, *, country_code: str = "E0", season: str = "2324") -> pd.DataFrame:
    """Load football matches either from a local csv or football-data.co.uk.

    Expects columns used by classic betting datasets: Date, HomeTeam, AwayTeam,
    FTR, and odds columns such as B365H/B365D/B365A.
    """

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        url = f"{FOOTBALL_DATA_BASE}/{season}/{country_code}.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    required = {"HomeTeam", "AwayTeam", "FTR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"]).copy()
    return df.sort_values("Date").reset_index(drop=True)
