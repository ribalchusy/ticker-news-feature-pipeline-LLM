from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


BASE_URL = "https://disclosure.1prime.ru"
SEARCH_URL = "https://disclosure.1prime.ru/news/"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports" / "samples"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_message_links(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    links = []

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"]
        url = urljoin(BASE_URL, href)

        links.append(
            {
                "text": text,
                "href": href,
                "url": url,
            }
        )

    links_df = pd.DataFrame(links)

    if links_df.empty:
        return links_df

    message_links = links_df[
        links_df["href"].astype(str).str.contains(
            r"/news/-203/.*\.uif",
            regex=True,
            na=False,
        )
    ].copy()

    message_links = message_links.drop_duplicates("url").reset_index(drop=True)

    return message_links


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)

        page.locator(
            'input[name="ctl00$LeftColumn$LeftSearchBlock$SName"]'
        ).fill("Сбербанк")

        page.locator(
            'input[name="ctl00$LeftColumn$LeftSearchBlock$SearchD1"]'
        ).fill("30.05.2024")

        page.locator(
            'input[name="ctl00$LeftColumn$LeftSearchBlock$SearchD2"]'
        ).fill("30.06.2025")

        page.locator(
            'input[name="ctl00$LeftColumn$LeftSearchBlock$STargetDocs"][value="0"]'
        ).check()

        page.locator(
            'input[name="ctl00$LeftColumn$LeftSearchBlock$SubmitSearchBlockForm"]'
        ).click()

        page.wait_for_load_state("domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)

        html = page.content()
        current_url = page.url

        print("Current URL:", current_url)

        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)

        print("\nPAGE TEXT SAMPLE:")
        print(text[:3000])

        message_links = extract_message_links(html)

        print("\nMESSAGE LINKS:")
        print(message_links[["text", "href", "url"]].head(30))

        output_path = REPORTS_DIR / "prime_sberbank_links_sample.csv"
        message_links.to_csv(output_path, index=False, encoding="utf-8-sig")

        print("\nSaved:", output_path)

        browser.close()


if __name__ == "__main__":
    main()