from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "news"
    / "news_for_llm_2024.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "news"
OUTPUT_PATH = OUTPUT_DIR / "news_for_llm_sample_2024.parquet"


N_PER_TICKER = 5
RANDOM_STATE = 42


def prepare_sample() -> pd.DataFrame:
    news = pd.read_parquet(INPUT_PATH)

    required_columns = ["news_id", "ticker", "published_at", "text"]

    missing_columns = [
        col for col in required_columns
        if col not in news.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    samples = []

    for ticker, group in news.groupby("ticker"):
        group_sample = group.sample(
            n=min(N_PER_TICKER, len(group)),
            random_state=RANDOM_STATE,
        )
        samples.append(group_sample)

    sample = pd.concat(samples, ignore_index=True)

    sample = (
        sample
        .sort_values(["ticker", "published_at", "news_id"])
        .reset_index(drop=True)
    )

    return sample


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = prepare_sample()
    sample.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", sample.shape)
    print("\nRows by ticker:")
    print(sample["ticker"].value_counts().sort_index())

    print("\nSample:")
    print(
        sample[
            [
                "news_id",
                "ticker",
                "published_at",
                "matched_alias",
                "text",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()