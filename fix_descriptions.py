"""
fix_descriptions.py
====================
Re-fetches description from each individual assessment page, using a
tighter parser that skips navigation/boilerplate text.
Overwrites data/catalog.json in place.
"""
import json
import sys
import time
import re
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

CATALOG_PATH = Path("data/catalog.json")

# Phrases that indicate boilerplate / navigation text to skip
SKIP_PHRASES = [
    "if you choose to continue",
    "cookie",
    "privacy",
    "copyright",
    "javascript",
    "upgrade to a modern browser",
    "we recommend",
    "browser",
    "gdpr",
    "terms of use",
]


def is_boilerplate(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in SKIP_PHRASES) or len(text) < 40


def fetch_description(url: str) -> str:
    """Fetch a detail page and extract a clean description."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Strategy: find the main content block by looking for the div
        # that appears AFTER the breadcrumb on SHL detail pages.
        # SHL detail pages have a main article section with class patterns.

        # Remove all navigation, header, footer, script, style elements
        for tag in soup.find_all(["nav", "header", "footer", "script", "style",
                                   "noscript", "aside"]):
            tag.decompose()

        # Also remove elements with nav-like class names
        for tag in soup.find_all(class_=re.compile(
            r"nav|menu|header|footer|breadcrumb|cookie|banner|alert|popup",
            re.I
        )):
            tag.decompose()

        # Now extract paragraphs
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            text = re.sub(r'\s+', ' ', text)
            if not is_boilerplate(text):
                paragraphs.append(text)

        # Cap at 3 paragraphs, join them
        description = " ".join(paragraphs[:3])

        # If still empty, try h2/h3 + first paragraph after it
        if not description:
            for h in soup.find_all(["h2", "h3"]):
                next_p = h.find_next_sibling("p")
                if next_p:
                    text = next_p.get_text(separator=" ", strip=True)
                    if not is_boilerplate(text):
                        description = text
                        break

        return description.strip()

    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return ""


def main():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    log.info(f"Re-enriching descriptions for {len(catalog)} assessments...")

    for i, assessment in enumerate(catalog, 1):
        name = assessment["name"]
        url = assessment["url"]
        log.info(f"[{i}/{len(catalog)}] {name}")

        desc = fetch_description(url)
        if desc:
            assessment["description"] = desc
        else:
            log.warning(f"  No description found for {name}")

        if i < len(catalog):
            time.sleep(0.4)

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    # Verify
    with_desc = sum(1 for a in catalog if a.get("description", "").strip())
    log.info(f"\nDone. {with_desc}/{len(catalog)} assessments have descriptions.")
    log.info(f"Saved to {CATALOG_PATH}")

    # Print 3 samples
    print("\n── Samples ──")
    for a in catalog[:3]:
        print(f"  {a['name']}")
        print(f"  DESC: {a['description'][:120]}")
        print()


if __name__ == "__main__":
    main()
