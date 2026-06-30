from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "news"
    / "news_events_precomputed_llm_2024.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "news"
OUTPUT_PATH = OUTPUT_DIR / "news_events_structured_2024.parquet"


EVENT_PATTERNS = {
    "dividends": [
        r"\bдивиденд",
        r"\bдивы\b",
        r"дивидендн",
        r"див\s*доход",
    ],
    "earnings": [
        r"\bвыручк",
        r"\bприбыл",
        r"\bebitda\b",
        r"\bмсфо\b",
        r"\bрсбу\b",
        r"финансов(ые|ых|ая|ую)? результат",
        r"\bотчетност",
        r"\bотч[её]т\b",
    ],
    "debt": [
        r"\bоблигац",
        r"\bбонды\b",
        r"\bдолг",
        r"\bкредит",
        r"кредитн(ый|ого|ому|ым)? рейтинг",
        r"\bзайм",
        r"\bкупон",
    ],
    "sanctions": [
        r"\bсанкц",
        r"\bограничен",
        r"\bэмбарго\b",
        r"\bблокирующ",
        r"\bsdн\b",
        r"\bofac\b",
    ],
    "production": [
        r"\bдобыч",
        r"\bпроизводств",
        r"\bвыпуск",
        r"\bпереработк",
        r"\bпоставк",
        r"\bэкспорт",
        r"\bзапас",
    ],
    "mna": [
        r"\bm&a\b",
        r"\bслияни",
        r"\bпоглощен",
        r"\bсделк",
        r"\bпокупк",
        r"\bпродаж[аиу]",
        r"\bактив",
        r"\bдол[яюи]\b",
    ],
    "management": [
        r"совет директор",
        r"\bсд\b",
        r"\bдиректор",
        r"\bменеджмент",
        r"\bгендиректор",
        r"\bакционер",
        r"\bсобрани",
    ],
    "forecast": [
        r"\bпрогноз",
        r"\bтаргет",
        r"целевая цена",
        r"\bрекомендац",
        r"\bаналитик",
        r"\bоценк",
        r"\bпотенциал",
    ],
    "legal": [
        r"\bсуд",
        r"\bиск",
        r"\bштраф",
        r"\bрасследован",
        r"\bрегулятор",
        r"\bлицензи",
        r"\bзакон",
    ],
    "macro": [
        r"\bинфляц",
        r"ключев(ая|ую|ой)? ставк",
        r"\bцб\b",
        r"\bфрс\b",
        r"\bдоллар",
        r"\bрубл",
        r"\bнефт",
        r"\bbrent\b",
        r"\bмакро",
        r"\bввп\b",
    ],
    "market_commentary": [
        r"главное к открытию",
        r"\bбрифинг",
        r"\bдайджест",
        r"какие акции",
        r"что купить",
        r"инвестор",
        r"\bрынок",
        r"\bторги",
        r"\bиндекс",
        r"\bмосбирж",
    ],
}


EVENT_PRIORITY = [
    "dividends",
    "earnings",
    "debt",
    "sanctions",
    "production",
    "mna",
    "management",
    "forecast",
    "legal",
    "macro",
    "market_commentary",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).lower()


def detect_event_type(text: str) -> tuple[str, str | None]:
    for event_type in EVENT_PRIORITY:
        patterns = EVENT_PATTERNS[event_type]

        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return event_type, pattern

    return "other", None


def prepare_structured_events() -> pd.DataFrame:
    events = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "news_id",
        "raw_news_id",
        "ticker",
        "published_at",
        "text",
        "tags",
        "source",
        "sentiment_score",
        "impact_score",
        "confidence",
        "reason",
        "llm_model",
    ]

    missing_columns = [col for col in required_columns if col not in events.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    result = events.copy()

    combined_text = (
        result["text"].apply(normalize_text)
        + " "
        + result["tags"].apply(normalize_text)
        + " "
        + result["reason"].apply(normalize_text)
    )

    detected = combined_text.apply(detect_event_type)

    result["event_type"] = detected.apply(lambda x: x[0])
    result["event_rule"] = detected.apply(lambda x: x[1])

    result["sentiment_label"] = pd.cut(
        result["sentiment_score"],
        bins=[-1.01, -0.2, 0.2, 1.01],
        labels=["negative", "neutral", "positive"],
    ).astype(str)

    result.loc[result["sentiment_score"].isna(), "sentiment_label"] = "unknown"

    result = result[
        [
            "news_id",
            "raw_news_id",
            "ticker",
            "published_at",
            "event_type",
            "event_rule",
            "sentiment_label",
            "sentiment_score",
            "impact_score",
            "confidence",
            "article_type",
            "sectors",
            "llm_tickers",
            "matched_alias",
            "match_type",
            "text",
            "tags",
            "source",
            "reason",
            "llm_model",
        ]
    ].copy()

    result = result.sort_values(["ticker", "published_at", "news_id"]).reset_index(drop=True)

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = prepare_structured_events()
    result.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", result.shape)

    print("\nEvent types:")
    print(result["event_type"].value_counts(dropna=False))

    print("\nSentiment labels:")
    print(result["sentiment_label"].value_counts(dropna=False))

    print("\nRows by ticker:")
    print(result["ticker"].value_counts().sort_index())

    print("\nSample:")
    print(
        result[
            [
                "ticker",
                "published_at",
                "event_type",
                "sentiment_label",
                "sentiment_score",
                "matched_alias",
                "text",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()