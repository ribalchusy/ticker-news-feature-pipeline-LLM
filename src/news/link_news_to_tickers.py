from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "news"
    / "kaggle_news_canonical_2024.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "news"
OUTPUT_PATH = OUTPUT_DIR / "news_ticker_links_2024.parquet"


ISSUER_ALIASES = {
    "SBER": [
        "сбербанк",
        "сбер",
        "sberbank",
    ],
    "GAZP": [
        "газпром",
        "gazprom",
    ],
    "ROSN": [
        "роснефть",
        "rosneft",
    ],
    "NVTK": [
        "новатэк",
        "novatek",
    ],
    "GMKN": [
        "норильский никель",
        "норникель",
        "nornickel",
        "гмк норильский никель",
    ],
    "PLZL": [
        "полюс",
        "polyus",
        "полюс золото",
    ],
    "TATN": [
        "татнефть",
        "tatneft",
    ],
    "AFLT": [
        "аэрофлот",
        "aeroflot",
    ],
    "YDEX": [
        "яндекс",
        "yandex",
    ],
    "VKCO": [
        "вконтакте",
        "vk company",
        "vk group",
        "группа vk",
        "мэйл.ру",
        "mail.ru",
        "mail.ru group",
    ],
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).lower()


def contains_alias(text: str, alias: str) -> bool:
    """
    Checks alias occurrence with safe token boundaries.

    This prevents false matches where a short alias is found inside another word.
    Example: short aliases like 'vk' or 'вк' are intentionally avoided in the alias list.
    """
    alias = alias.lower().strip()

    if not alias:
        return False

    pattern = rf"(?<![a-zа-яё0-9]){re.escape(alias)}(?![a-zа-яё0-9])"

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def find_matches_in_text(news_id: str, text: str, tags: str) -> list[dict]:
    matches = []

    normalized_text = normalize_text(text)
    normalized_tags = normalize_text(tags)

    for ticker, aliases in ISSUER_ALIASES.items():
        for alias in aliases:
            if contains_alias(normalized_tags, alias):
                matches.append(
                    {
                        "news_id": news_id,
                        "ticker": ticker,
                        "match_type": "tag",
                        "matched_alias": alias,
                    }
                )
                break

            if contains_alias(normalized_text, alias):
                matches.append(
                    {
                        "news_id": news_id,
                        "ticker": ticker,
                        "match_type": "text",
                        "matched_alias": alias,
                    }
                )
                break

    return matches


def link_news_to_tickers(news: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["news_id", "text", "tags"]
    missing_columns = [col for col in required_columns if col not in news.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    all_matches = []

    for row in news.itertuples(index=False):
        row_matches = find_matches_in_text(
            news_id=row.news_id,
            text=row.text,
            tags=row.tags,
        )

        all_matches.extend(row_matches)

    links = pd.DataFrame(all_matches)

    if links.empty:
        return pd.DataFrame(
            columns=["news_id", "ticker", "match_type", "matched_alias"]
        )

    links = (
        links.drop_duplicates(["news_id", "ticker"])
        .sort_values(["ticker", "news_id"])
        .reset_index(drop=True)
    )

    return links


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    news = pd.read_parquet(NEWS_PATH)
    links = link_news_to_tickers(news)

    links.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", links.shape)

    if not links.empty:
        print("\nLinks by ticker:")
        print(links["ticker"].value_counts().sort_index())

        print("\nMatch types:")
        print(links["match_type"].value_counts())

        print("\nMatched aliases:")
        print(links["matched_alias"].value_counts().head(30))

        print("\nSample:")
        print(links.head(20))


if __name__ == "__main__":
    main()