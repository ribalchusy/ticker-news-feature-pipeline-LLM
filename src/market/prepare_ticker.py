from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_MARKET_DIR = PROJECT_ROOT / "data" / "raw" / "market"
PROCESSED_NEWS_DIR = PROJECT_ROOT / "data" / "processed" / "news"


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


def load_df_final() -> pd.DataFrame:
    """
    Loads df_final from data/raw/market.

    Supported formats:
    - df_final.csv
    - df_final.parquet
    - df_final.pkl
    - df_final.xlsx
    """
    possible_files = [
        RAW_MARKET_DIR / "df_final.csv",
        RAW_MARKET_DIR / "df_final.parquet",
        RAW_MARKET_DIR / "df_final.pkl",
        RAW_MARKET_DIR / "df_final.xlsx",
    ]

    existing_files = [path for path in possible_files if path.exists()]

    if not existing_files:
        raise FileNotFoundError(
            "df_final file was not found in data/raw/market. "
            "Expected one of: df_final.csv, df_final.parquet, df_final.pkl, df_final.xlsx"
        )

    path = existing_files[0]

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".pkl":
        return pd.read_pickle(path)

    if path.suffix == ".xlsx":
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def validate_columns(df: pd.DataFrame) -> None:
    required_cols = [
        "begin",
        "ticker",
        "open",
        "close",
        "high",
        "low",
        "value",
        "volume",
        "end",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def prepare_market_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    validate_columns(df)

    df["begin"] = pd.to_datetime(df["begin"])
    df["end"] = pd.to_datetime(df["end"])

    df = df[df["ticker"].isin(SELECTED_TICKERS)].copy()

    df = (
        df.sort_values(["ticker", "begin", "end"])
        .drop_duplicates(["ticker", "begin"], keep="last")
        .reset_index(drop=True)
    )

    return df


def build_ticker_universe(df: pd.DataFrame) -> pd.DataFrame:
    ticker_universe = (
        df.groupby("ticker")
        .agg(
            start_datetime=("begin", "min"),
            end_datetime=("begin", "max"),
            n_rows=("begin", "count"),
        )
        .reset_index()
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    return ticker_universe


def build_market_time_grid(df: pd.DataFrame) -> pd.DataFrame:
    market_time_grid = (
        df[["ticker", "begin"]]
        .rename(columns={"begin": "datetime"})
        .drop_duplicates(["ticker", "datetime"])
        .sort_values(["ticker", "datetime"])
        .reset_index(drop=True)
    )

    return market_time_grid


def main() -> None:
    PROCESSED_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_df_final()
    df = prepare_market_dataset(df)

    ticker_universe = build_ticker_universe(df)
    market_time_grid = build_market_time_grid(df)

    ticker_universe.to_parquet(
        PROCESSED_NEWS_DIR / "ticker_universe.parquet",
        index=False,
    )

    market_time_grid.to_parquet(
        PROCESSED_NEWS_DIR / "market_time_grid.parquet",
        index=False,
    )

    print("Ticker universe:")
    print(ticker_universe)

    print()
    print("Market time grid shape:")
    print(market_time_grid.shape)

    print()
    print("Saved files:")
    print(PROCESSED_NEWS_DIR / "ticker_universe.parquet")
    print(PROCESSED_NEWS_DIR / "market_time_grid.parquet")


if __name__ == "__main__":
    main()