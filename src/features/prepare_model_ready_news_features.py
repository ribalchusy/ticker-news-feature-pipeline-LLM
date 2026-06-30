from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "news"
    / "news_features_hourly_2024.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features"
OUTPUT_PATH = OUTPUT_DIR / "model_ready_news_features_2024.parquet"


def prepare_model_ready_features() -> pd.DataFrame:
    features = pd.read_parquet(NEWS_FEATURES_PATH)

    required_columns = ["datetime", "ticker"]
    missing_columns = [col for col in required_columns if col not in features.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    features = features.copy()
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")

    features = features.dropna(subset=["datetime", "ticker"]).copy()

    feature_cols = [
        col for col in features.columns
        if col not in ["datetime", "ticker"]
    ]

    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    return features


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = prepare_model_ready_features()
    result.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", result.shape)
    print("Date range:", result["datetime"].min(), "→", result["datetime"].max())

    print("\nTickers:")
    print(result["ticker"].value_counts().sort_index())

    print("\nNon-zero news rows:")
    for window in [1, 6, 24, 72]:
        col = f"news_count_{window}h"
        print(col, ":", (result[col] > 0).sum())

    print("\nMissing values:")
    print(result.isna().sum().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()