from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MARKET_GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market"
    / "market_time_grid_2024.parquet"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "news"
    / "news_events_structured_2024.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "news"
OUTPUT_PATH = OUTPUT_DIR / "news_features_hourly_2024.parquet"


WINDOWS_HOURS = [1, 6, 24, 72]

EVENT_TYPES = [
    "dividends",
    "earnings",
    "debt",
    "sanctions",
    "production",
    "mna",
    "management",
    "forecast",
    "market_commentary",
    "macro",
    "legal",
    "other",
]

SENTIMENT_LABELS = [
    "positive",
    "negative",
    "neutral",
    "unknown",
]


# На первом этапе ставим 0.
# Если хочешь быть строже по leakage, можно поставить 15 или 30 минут.
NEWS_AVAILABILITY_DELAY_MINUTES = 0


def prepare_event_matrix(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    events["published_at"] = pd.to_datetime(events["published_at"], errors="coerce")
    events = events.dropna(subset=["published_at", "ticker"]).copy()

    events["available_at"] = events["published_at"] + pd.Timedelta(
        minutes=NEWS_AVAILABILITY_DELAY_MINUTES
    )

    events["news_count"] = 1

    events["sentiment_score"] = pd.to_numeric(
        events["sentiment_score"],
        errors="coerce",
    ).fillna(0.0)

    events["impact_score"] = pd.to_numeric(
        events["impact_score"],
        errors="coerce",
    ).fillna(0.0)

    events["impact_abs"] = events["impact_score"].abs()

    for event_type in EVENT_TYPES:
        events[f"event_{event_type}"] = (
            events["event_type"].eq(event_type).astype(int)
        )

    for sentiment in SENTIMENT_LABELS:
        events[f"sentiment_{sentiment}"] = (
            events["sentiment_label"].eq(sentiment).astype(int)
        )

    event_columns = (
        ["news_count", "sentiment_score", "impact_score", "impact_abs"]
        + [f"event_{event_type}" for event_type in EVENT_TYPES]
        + [f"sentiment_{sentiment}" for sentiment in SENTIMENT_LABELS]
    )

    event_matrix = (
        events.groupby(["ticker", "available_at"], as_index=False)[event_columns]
        .sum()
        .rename(columns={"available_at": "datetime"})
    )

    return event_matrix


def build_features_for_ticker(
    grid_ticker: pd.DataFrame,
    events_ticker: pd.DataFrame,
) -> pd.DataFrame:
    ticker = grid_ticker["ticker"].iloc[0]

    grid_ticker = grid_ticker.copy()
    grid_ticker["datetime"] = pd.to_datetime(grid_ticker["datetime"], errors="coerce")
    grid_ticker = grid_ticker.dropna(subset=["datetime"]).copy()
    grid_ticker = grid_ticker.sort_values("datetime")

    grid_times = pd.Index(grid_ticker["datetime"].unique())

    if events_ticker.empty:
        result = grid_ticker[["datetime", "ticker"]].copy()

        for window in WINDOWS_HOURS:
            result[f"news_count_{window}h"] = 0
            result[f"sentiment_avg_{window}h"] = 0.0
            result[f"impact_avg_{window}h"] = 0.0
            result[f"impact_abs_max_{window}h"] = 0.0

            for sentiment in SENTIMENT_LABELS:
                result[f"{sentiment}_count_{window}h"] = 0

            for event_type in EVENT_TYPES:
                result[f"{event_type}_count_{window}h"] = 0

        return result

    events_ticker = events_ticker.copy()
    events_ticker["datetime"] = pd.to_datetime(events_ticker["datetime"], errors="coerce")
    events_ticker = events_ticker.dropna(subset=["datetime"]).copy()
    events_ticker = events_ticker.sort_values("datetime")

    event_feature_columns = [
        col for col in events_ticker.columns
        if col not in ["ticker", "datetime"]
    ]

    timeline = pd.Index(
        sorted(set(grid_ticker["datetime"]).union(set(events_ticker["datetime"])))
    )

    ts = (
        events_ticker
        .set_index("datetime")[event_feature_columns]
        .reindex(timeline)
        .fillna(0.0)
        .sort_index()
    )

    features = pd.DataFrame(index=timeline)

    for window in WINDOWS_HOURS:
        window_str = f"{window}h"

        rolling_sum = ts.rolling(window=window_str, closed="both").sum()

        features[f"news_count_{window}h"] = rolling_sum["news_count"]

        features[f"sentiment_sum_{window}h"] = rolling_sum["sentiment_score"]
        features[f"impact_sum_{window}h"] = rolling_sum["impact_score"]

        features[f"sentiment_avg_{window}h"] = (
            features[f"sentiment_sum_{window}h"]
            / features[f"news_count_{window}h"].replace(0, pd.NA)
        ).fillna(0.0)

        features[f"impact_avg_{window}h"] = (
            features[f"impact_sum_{window}h"]
            / features[f"news_count_{window}h"].replace(0, pd.NA)
        ).fillna(0.0)

        features[f"impact_abs_max_{window}h"] = (
            ts["impact_abs"]
            .rolling(window=window_str, closed="both")
            .max()
            .fillna(0.0)
        )

        for sentiment in SENTIMENT_LABELS:
            source_col = f"sentiment_{sentiment}"
            features[f"{sentiment}_count_{window}h"] = rolling_sum[source_col]

        for event_type in EVENT_TYPES:
            source_col = f"event_{event_type}"
            features[f"{event_type}_count_{window}h"] = rolling_sum[source_col]

    features = features.loc[grid_times].copy()
    features = features.reset_index().rename(columns={"index": "datetime"})
    features["ticker"] = ticker

    # Убираем технические суммы, оставляем только понятные признаки.
    drop_cols = [
        col for col in features.columns
        if col.startswith("sentiment_sum_") or col.startswith("impact_sum_")
    ]

    features = features.drop(columns=drop_cols)

    ordered_cols = ["datetime", "ticker"] + [
        col for col in features.columns
        if col not in ["datetime", "ticker"]
    ]

    features = features[ordered_cols].sort_values(["ticker", "datetime"]).reset_index(drop=True)

    return features


def build_news_features() -> pd.DataFrame:
    market_grid = pd.read_parquet(MARKET_GRID_PATH)
    events = pd.read_parquet(EVENTS_PATH)

    required_grid_columns = ["datetime", "ticker"]
    required_event_columns = [
        "ticker",
        "published_at",
        "event_type",
        "sentiment_label",
        "sentiment_score",
        "impact_score",
    ]

    missing_grid_columns = [
        col for col in required_grid_columns
        if col not in market_grid.columns
    ]

    missing_event_columns = [
        col for col in required_event_columns
        if col not in events.columns
    ]

    if missing_grid_columns:
        raise ValueError(f"Missing market grid columns: {missing_grid_columns}")

    if missing_event_columns:
        raise ValueError(f"Missing event columns: {missing_event_columns}")

    market_grid = market_grid.copy()
    market_grid["datetime"] = pd.to_datetime(market_grid["datetime"], errors="coerce")
    market_grid = market_grid.dropna(subset=["datetime", "ticker"]).copy()

    event_matrix = prepare_event_matrix(events)

    all_features = []

    for ticker in sorted(market_grid["ticker"].unique()):
        print("Building features for:", ticker)

        grid_ticker = market_grid[market_grid["ticker"] == ticker].copy()
        events_ticker = event_matrix[event_matrix["ticker"] == ticker].copy()

        ticker_features = build_features_for_ticker(
            grid_ticker=grid_ticker,
            events_ticker=events_ticker,
        )

        all_features.append(ticker_features)

    result = pd.concat(all_features, ignore_index=True)

    result = result.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = build_news_features()
    result.to_parquet(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("Shape:", result.shape)
    print("Date range:", result["datetime"].min(), "→", result["datetime"].max())

    print("\nTickers:")
    print(result["ticker"].value_counts().sort_index())

    print("\nNon-zero 24h news count rows:")
    print((result["news_count_24h"] > 0).sum())

    print("\nSample:")
    print(result.head(20))


if __name__ == "__main__":
    main()