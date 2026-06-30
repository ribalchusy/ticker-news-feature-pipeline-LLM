from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "kaggle"
    / "russian_financial_news"
    / "RussianFinancialNews"
    / "news_collection.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "news"
OUTPUT_PATH = OUTPUT_DIR / "kaggle_news_canonical_2024.parquet"


PROJECT_START = "2024-01-03 09:00:00"
PROJECT_END = "2024-12-13 23:00:00"


def clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value)
    text = " ".join(text.split())

    if not text:
        return None

    return text


def prepare_kaggle_news() -> pd.DataFrame:
    news = pd.read_parquet(RAW_NEWS_PATH)

    required_columns = ["title", "body", "date", "time", "tags", "source"]
    missing_columns = [col for col in required_columns if col not in news.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    news = news.copy()

    news["published_at"] = pd.to_datetime(
        news["date"].astype(str).str.strip()
        + " "
        + news["time"].astype(str).str.strip(),
        errors="coerce",
    )

    news["title"] = news["title"].replace({"no title": None})
    news["title"] = news["title"].apply(clean_text)

    news["text"] = news["body"].apply(clean_text)

    project_start = pd.Timestamp(PROJECT_START)
    project_end = pd.Timestamp(PROJECT_END)

    news = news[
        (news["published_at"] >= project_start)
        & (news["published_at"] <= project_end)
    ].copy()

    news = news.dropna(subset=["published_at", "text"]).copy()

    news = news.reset_index(drop=False).rename(columns={"index": "raw_news_id"})

    news["news_id"] = news["raw_news_id"].apply(lambda x: f"kaggle_rfn_{x}")
    news["raw_source"] = "kaggle_russian_financial_news"

    result = news[
        [
            "news_id",
            "raw_news_id",
            "published_at",
            "date",
            "time",
            "title",
            "text",
            "tags",
            "source",
            "raw_source",
        ]
    ].copy()

    result = result.sort_values("published_at").reset_index(drop=True)

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = prepare_kaggle_news()
    result.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", result.shape)
    print("Date range:", result["published_at"].min(), "→", result["published_at"].max())
    print("Sources:")
    print(result["source"].value_counts(dropna=False).head(20))
    print("Text length:")
    print(result["text"].astype(str).str.len().describe())


if __name__ == "__main__":
    main()