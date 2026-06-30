from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_FOR_LLM_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "news"
    / "news_for_llm_2024.parquet"
)

GPT4O_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "kaggle"
    / "russian_financial_news"
    / "RussianFinancialNews"
    / "news_descriptions"
    / "news_descriptions_GPT4o.json"
)

LLAMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "kaggle"
    / "russian_financial_news"
    / "RussianFinancialNews"
    / "news_descriptions"
    / "news_description_Llama3_8b.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "news"
OUTPUT_PATH = OUTPUT_DIR / "news_events_precomputed_llm_2024.parquet"


def load_description_json(path: Path, model_name: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []

    for raw_news_id, value in data.items():
        description = value.get("description", {}) or {}

        rows.append(
            {
                "raw_news_id": int(raw_news_id),
                f"{model_name}_thinking": value.get("thinking"),
                f"{model_name}_article_type": description.get("article_type"),
                f"{model_name}_country": description.get("country"),
                f"{model_name}_sectors": description.get("sectors"),
                f"{model_name}_llm_tickers": description.get("tickers"),
                f"{model_name}_sentiment_score": description.get("sentiment_score"),
            }
        )

    return pd.DataFrame(rows)


def choose_first_available(row: pd.Series, gpt_col: str, llama_col: str):
    gpt_value = row.get(gpt_col)
    llama_value = row.get(llama_col)

    if isinstance(gpt_value, list):
        if len(gpt_value) > 0:
            return gpt_value
    elif pd.notna(gpt_value):
        return gpt_value

    if isinstance(llama_value, list):
        if len(llama_value) > 0:
            return llama_value
    elif pd.notna(llama_value):
        return llama_value

    return None


def prepare_precomputed_llm_events() -> pd.DataFrame:
    news = pd.read_parquet(NEWS_FOR_LLM_PATH)

    required_columns = [
        "news_id",
        "raw_news_id",
        "ticker",
        "published_at",
        "text",
        "tags",
        "source",
        "matched_alias",
        "match_type",
    ]

    missing_columns = [col for col in required_columns if col not in news.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in news_for_llm: {missing_columns}")

    gpt4o = load_description_json(GPT4O_PATH, model_name="gpt4o")
    llama = load_description_json(LLAMA_PATH, model_name="llama3_8b")

    result = news.merge(gpt4o, on="raw_news_id", how="left")
    result = result.merge(llama, on="raw_news_id", how="left")

    result["article_type"] = result.apply(
        lambda row: choose_first_available(
            row,
            "gpt4o_article_type",
            "llama3_8b_article_type",
        ),
        axis=1,
    )

    result["sectors"] = result.apply(
        lambda row: choose_first_available(
            row,
            "gpt4o_sectors",
            "llama3_8b_sectors",
        ),
        axis=1,
    )

    result["llm_tickers"] = result.apply(
        lambda row: choose_first_available(
            row,
            "gpt4o_llm_tickers",
            "llama3_8b_llm_tickers",
        ),
        axis=1,
    )

    result["sentiment_score"] = result.apply(
        lambda row: choose_first_available(
            row,
            "gpt4o_sentiment_score",
            "llama3_8b_sentiment_score",
        ),
        axis=1,
    )

    result["llm_thinking"] = result.apply(
        lambda row: choose_first_available(
            row,
            "gpt4o_thinking",
            "llama3_8b_thinking",
        ),
        axis=1,
    )

    result["llm_model"] = result["gpt4o_sentiment_score"].apply(
        lambda value: "gpt4o" if pd.notna(value) else "llama3_8b"
    )

    result["sentiment_score"] = pd.to_numeric(
        result["sentiment_score"],
        errors="coerce",
    )

    # Для первой версии impact_score = sentiment_score.
    # Позже можно усложнить и учитывать event_type / confidence / sector.
    result["impact_score"] = result["sentiment_score"]

    result["confidence"] = result["sentiment_score"].notna().astype(float)

    result["reason"] = result["llm_thinking"]

    result = result[
        [
            "news_id",
            "raw_news_id",
            "ticker",
            "published_at",
            "text",
            "tags",
            "source",
            "matched_alias",
            "match_type",
            "article_type",
            "sectors",
            "llm_tickers",
            "sentiment_score",
            "impact_score",
            "confidence",
            "reason",
            "llm_model",
        ]
    ].copy()

    result = result.sort_values(["ticker", "published_at", "news_id"]).reset_index(drop=True)

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = prepare_precomputed_llm_events()
    result.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", result.shape)

    print("\nCoverage:")
    print("Rows:", len(result))
    print("With sentiment:", result["sentiment_score"].notna().sum())
    print("Missing sentiment:", result["sentiment_score"].isna().sum())

    print("\nRows by ticker:")
    print(result["ticker"].value_counts().sort_index())

    print("\nLLM model:")
    print(result["llm_model"].value_counts(dropna=False))

    print("\nArticle types:")
    print(result["article_type"].value_counts(dropna=False).head(20))

    print("\nSentiment score:")
    print(result["sentiment_score"].describe())


if __name__ == "__main__":
    main()