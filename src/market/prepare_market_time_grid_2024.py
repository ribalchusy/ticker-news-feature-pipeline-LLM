from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_MARKET_PATH = PROJECT_ROOT / "data" / "raw" / "market" / "df_final.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "market"
OUTPUT_PATH = OUTPUT_DIR / "market_time_grid_2024.parquet"


SELECTED_TICKERS = [
    "SBER",
    "GAZP",
    "ROSN",
    "NVTK",
    "GMKN",
    "PLZL",
    "TATN",
    "AFLT",
    "YDEX",
    "VKCO",
]


PROJECT_START = "2024-01-03 09:00:00"
PROJECT_END = "2024-12-13 23:00:00"


def prepare_market_time_grid() -> pd.DataFrame:
    market = pd.read_parquet(RAW_MARKET_PATH)

    required_columns = ["begin", "ticker"]
    missing_columns = [col for col in required_columns if col not in market.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    market = market.copy()
    market["begin"] = pd.to_datetime(market["begin"], errors="coerce")

    project_start = pd.Timestamp(PROJECT_START)
    project_end = pd.Timestamp(PROJECT_END)

    market = market[
        (market["begin"] >= project_start)
        & (market["begin"] <= project_end)
        & (market["ticker"].isin(SELECTED_TICKERS))
    ].copy()

    market_time_grid = (
        market[["begin", "ticker"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"begin": "datetime"})
        .sort_values(["ticker", "datetime"])
        .reset_index(drop=True)
    )

    return market_time_grid


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market_time_grid = prepare_market_time_grid()
    market_time_grid.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", market_time_grid.shape)
    print("Date range:", market_time_grid["datetime"].min(), "→", market_time_grid["datetime"].max())
    print("Tickers:")
    print(market_time_grid["ticker"].value_counts().sort_index())


if __name__ == "__main__":
    main()