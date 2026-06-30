from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "news"
    / "kaggle_news_canonical_2024.parquet"
)

LINKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "news"
    / "news_ticker_links_2024.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "news"
OUTPUT_PATH = OUTPUT_DIR / "news_for_llm_2024.parquet"


def prepare_news_for_llm() -> pd.DataFrame:
    news = pd.read_parquet(NEWS_PATH)
    links = pd.read_parquet(LINKS_PATH)

    required_news_columns = [
        "news_id",
        "published_at",
        "text",
        "tags",
        "source",
    ]

    required_links_columns = [
        "news_id",
        "ticker",
        "match_type",
        "matched_alias",
    ]

    missing_news_columns = [
        col for col in required_news_columns if col not in news.columns
    ]

    missing_links_columns = [
        col for col in required_links_columns if col not in links.columns
    ]

    if missing_news_columns:
        raise ValueError(f"Missing news columns: {missing_news_columns}")

    if missing_links_columns:
        raise ValueError(f"Missing links columns: {missing_links_columns}")

    result = links.merge(
        news[
            [
                "news_id",
                "raw_news_id",
                "published_at",
                "title",
                "text",
                "tags",
                "source",
                "raw_source",
            ]
        ],
        on="news_id",
        how="left",
        validate="many_to_one",
    )

    result = result.dropna(subset=["published_at", "text"]).copy()

    result["text_length"] = result["text"].astype(str).str.len()

    result = result[
        [
            "news_id",
            "raw_news_id",
            "ticker",
            "published_at",
            "title",
            "text",
            "text_length",
            "tags",
            "source",
            "raw_source",
            "match_type",
            "matched_alias",
        ]
    ].copy()

    result = result.sort_values(["ticker", "published_at", "news_id"]).reset_index(drop=True)

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = prepare_news_for_llm()
    result.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", result.shape)

    print("\nDate range:")
    print(result["published_at"].min(), "→", result["published_at"].max())

    print("\nRows by ticker:")
    print(result["ticker"].value_counts().sort_index())

    print("\nText length:")
    print(result["text_length"].describe())

    print("\nSample:")
    print(
        result[
            [
                "ticker",
                "published_at",
                "matched_alias",
                "match_type",
                "text",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()