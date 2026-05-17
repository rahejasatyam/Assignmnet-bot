"""
scraper/scrape_catalog.py
=========================
Scrapes ALL Individual Test Solutions (type=1) from the SHL product catalog.

Design decisions:
- Uses requests + BeautifulSoup (no Playwright needed — server-side rendered).
- Only scrapes type=1 (Individual Test Solutions), not type=2 (Pre-packaged Job Solutions).
- Paginates through all pages (start=0,12,24,...) until no more rows appear.
- For each assessment on the listing page, we grab: name, url, remote_testing,
  adaptive_irt, test_types (list of letter codes).
- Then fetches each individual assessment page for: description, job_levels,
  languages, and any other fields available (duration, key_features).
- Saves final list to data/catalog.json.
- Includes retry logic and polite rate-limiting so SHL doesn't block us.
"""

import json
import time
import re
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/"

# type=1 → Individual Test Solutions (what we want)
# type=2 → Pre-packaged Job Solutions (excluded per assignment brief)
SECTION_TYPE = 1
PAGE_SIZE = 12          # SHL shows 12 items per page
DELAY_BETWEEN_PAGES = 1.0   # seconds — be polite
DELAY_BETWEEN_DETAIL = 0.5  # seconds for detail page fetches
MAX_RETRIES = 3

# Output path (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Test type code → human-readable label mapping
# Source: SHL's own legend shown on the catalog page
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


# ── HTTP helper ────────────────────────────────────────────────────────────────

def fetch(url: str, params: Optional[dict] = None, retries: int = MAX_RETRIES) -> Optional[BeautifulSoup]:
    """
    Fetch a URL and return a BeautifulSoup object, or None on failure.
    Retries up to `retries` times with exponential back-off.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as exc:
            log.warning(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential back-off
    log.error(f"All {retries} attempts failed for {url}")
    return None


# ── Listing page parser ────────────────────────────────────────────────────────

def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    """
    Parse one page of the catalog listing.

    The SHL catalog renders two separate tables: one for Pre-packaged Job
    Solutions (type=2) and one for Individual Test Solutions (type=1).
    Each table uses class="custom__table-responsive".

    Each <tr> in the Individual Test Solutions table has:
      - Column 0: assessment name with <a> link
      - Column 1: Remote Testing indicator (span.catalogue__circle.-yes or absent)
      - Column 2: Adaptive/IRT indicator (span.catalogue__circle.-yes or absent)
      - Columns 3+: Test type letter badges (span.product-catalogue__key)
    
    IMPORTANT: The page renders BOTH tables when both type params are present.
    We use `?start=X&type=1` to get Individual Test Solutions in the SECOND table
    (the first table may be Pre-packaged or empty depending on the page).
    We identify the correct table heuristically by taking the table with the
    most rows, or by checking which table's first-row link points to the
    correct section structure.
    """
    assessments = []

    # Find all catalog tables on the page
    tables = soup.find_all("table", class_=lambda c: c and "custom__table" in c)
    
    if not tables:
        # Fallback: look for any table with data-entity-id rows
        all_rows = soup.find_all("tr", attrs={"data-entity-id": True})
        if not all_rows:
            log.warning("No assessment rows found on this page.")
            return []
        # treat all rows as a single virtual table
        tables = [{"rows": all_rows}]

    # We want Individual Test solutions; when ?type=1 is passed the FIRST
    # rendered table is Individual tests. But to be safe we parse ALL tables
    # and deduplicate by URL later.
    seen_urls = set()
    
    for table in tables:
        if isinstance(table, dict):
            rows = table["rows"]
        else:
            rows = table.find_all("tr", attrs={"data-entity-id": True})
        
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            # ── Name & URL ─────────────────────────────────────────────────
            name_cell = cells[0]
            link_tag = name_cell.find("a")
            if not link_tag:
                continue
            name = link_tag.get_text(strip=True)
            href = link_tag.get("href", "")
            if href.startswith("/"):
                href = BASE_URL + href
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)

            # ── Remote Testing (column 1) ──────────────────────────────────
            remote_testing = False
            if len(cells) > 1:
                yes_span = cells[1].find("span", class_=lambda c: c and "-yes" in c)
                remote_testing = yes_span is not None

            # ── Adaptive / IRT (column 2) ──────────────────────────────────
            adaptive_irt = False
            if len(cells) > 2:
                yes_span = cells[2].find("span", class_=lambda c: c and "-yes" in c)
                adaptive_irt = yes_span is not None

            # ── Test Type Codes (columns 3 onwards) ────────────────────────
            test_types = []
            for cell in cells[3:]:
                badges = cell.find_all("span", class_=lambda c: c and "product-catalogue__key" in c)
                for badge in badges:
                    code = badge.get_text(strip=True).upper()
                    if code and code in TEST_TYPE_LABELS:
                        test_types.append(code)

            assessments.append({
                "name": name,
                "url": href,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "test_types": test_types,
                # Fields to be filled by detail page scrape:
                "description": "",
                "job_levels": [],
                "languages": [],
                "duration": "",
                "key_features": [],
            })

    return assessments


# ── Detail page parser ─────────────────────────────────────────────────────────

def parse_detail_page(soup: BeautifulSoup, assessment: dict) -> dict:
    """
    Enrich an assessment dict with data from its individual page.
    
    Individual assessment pages (e.g. /products/product-catalog/view/opq32r/)
    typically contain:
      - Description paragraph(s) in the main content area
      - Job levels / relevant roles in a sidebar or detail section
      - Languages in a detail section
      - Duration (sometimes) in a detail section
      - Key features as bullet list (sometimes)

    We use broad selectors and fall through gracefully if a field is missing.
    """
    enriched = dict(assessment)

    # ── Description ────────────────────────────────────────────────────────
    # Look for main content area with paragraphs
    description_parts = []
    
    # Common pattern: main article or section with class containing "content"
    content_areas = soup.select(
        "div.product-catalogue__content, "
        "div.product__content, "
        "div.catalogue__content, "
        "section.content, "
        "div.rich-text, "
        "article"
    )
    
    if content_areas:
        for area in content_areas[:1]:  # take only the first match
            paras = area.find_all("p")
            for p in paras:
                text = p.get_text(strip=True)
                if text and len(text) > 30:  # skip very short lines
                    description_parts.append(text)
    
    if not description_parts:
        # Broader fallback: take all <p> tags in body that look substantive
        all_paras = soup.find_all("p")
        for p in all_paras:
            text = p.get_text(strip=True)
            if (len(text) > 60 
                and "cookie" not in text.lower() 
                and "privacy" not in text.lower()
                and "copyright" not in text.lower()):
                description_parts.append(text)
                if len(description_parts) >= 3:
                    break

    enriched["description"] = " ".join(description_parts[:3])

    # ── Job Levels / Relevant Roles ───────────────────────────────────────
    job_levels = []
    
    # Look for structured key-value pairs in the assessment detail section
    detail_items = soup.select(
        "div.product-catalogue__detail, "
        "ul.product-catalogue__list, "
        "div.catalogue__details"
    )
    
    for item in detail_items:
        text = item.get_text(separator=" | ", strip=True).lower()
        if "job level" in text or "level" in text or "role" in text:
            # Extract list items
            lis = item.find_all("li")
            for li in lis:
                val = li.get_text(strip=True)
                if val:
                    job_levels.append(val)

    # Fallback: look for any element mentioning "Job Level" label
    if not job_levels:
        labels = soup.find_all(string=re.compile(r"job\s+level", re.I))
        for label in labels[:2]:
            parent = label.parent
            if parent:
                sibling = parent.find_next_sibling()
                if sibling:
                    lis = sibling.find_all("li")
                    for li in lis:
                        val = li.get_text(strip=True)
                        if val:
                            job_levels.append(val)
                    if not lis:
                        val = sibling.get_text(strip=True)
                        if val:
                            job_levels.extend([v.strip() for v in val.split(",") if v.strip()])

    enriched["job_levels"] = list(dict.fromkeys(job_levels))  # dedup preserving order

    # ── Languages ─────────────────────────────────────────────────────────
    languages = []
    lang_labels = soup.find_all(string=re.compile(r"language", re.I))
    for label in lang_labels[:2]:
        parent = label.parent
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                lis = sibling.find_all("li")
                for li in lis:
                    val = li.get_text(strip=True)
                    if val:
                        languages.append(val)
                if not lis:
                    val = sibling.get_text(strip=True)
                    if val:
                        languages.extend([v.strip() for v in val.split(",") if v.strip()])
    
    enriched["languages"] = list(dict.fromkeys(languages))

    # ── Duration ──────────────────────────────────────────────────────────
    # Look for minutes pattern in the page text
    duration = ""
    duration_labels = soup.find_all(string=re.compile(r"duration|minutes|timing", re.I))
    for label in duration_labels[:3]:
        parent = label.parent
        if parent:
            # Check current element or next sibling
            for el in [parent, parent.find_next_sibling()]:
                if el:
                    text = el.get_text(strip=True)
                    # Pattern: "25 minutes" or "25-35 mins" or "Approximately 30 min"
                    m = re.search(r"(\d+(?:\s*[-–]\s*\d+)?)\s*(?:minutes?|mins?)", text, re.I)
                    if m:
                        duration = m.group(0).strip()
                        break
        if duration:
            break

    # Also scan the full description for a duration mention
    if not duration and enriched["description"]:
        m = re.search(r"(\d+(?:\s*[-–]\s*\d+)?)\s*(?:minutes?|mins?)", enriched["description"], re.I)
        if m:
            duration = m.group(0).strip()

    enriched["duration"] = duration

    # ── Key Features ──────────────────────────────────────────────────────
    key_features = []
    # Look for bullet lists that seem to describe features
    feature_headers = soup.find_all(
        string=re.compile(r"key\s+feature|highlight|benefit|measure", re.I)
    )
    for header in feature_headers[:2]:
        parent = header.parent
        if parent:
            ul = parent.find_next("ul")
            if ul:
                for li in ul.find_all("li"):
                    val = li.get_text(strip=True)
                    if val and len(val) > 5:
                        key_features.append(val)

    enriched["key_features"] = key_features[:10]  # cap at 10

    return enriched


# ── Main scraping orchestrator ─────────────────────────────────────────────────

def scrape_all_listings() -> list[dict]:
    """
    Paginate through all Individual Test Solutions pages and collect
    basic assessment data (name, url, remote_testing, adaptive_irt, test_types).
    """
    all_assessments = []
    seen_urls = set()
    start = 0

    log.info("Starting catalog listing scrape (Individual Test Solutions only)...")

    while True:
        params = {"start": start, "type": SECTION_TYPE}
        log.info(f"Fetching listing page start={start} ...")
        soup = fetch(CATALOG_URL, params=params)
        
        if soup is None:
            log.error(f"Failed to fetch page at start={start}. Stopping.")
            break

        page_assessments = parse_listing_page(soup)

        if not page_assessments:
            log.info(f"No assessments found at start={start}. Reached end of catalog.")
            break

        # Deduplicate across pages
        new_count = 0
        for assessment in page_assessments:
            if assessment["url"] not in seen_urls:
                seen_urls.add(assessment["url"])
                all_assessments.append(assessment)
                new_count += 1

        log.info(f"  → Found {new_count} new assessments (total: {len(all_assessments)})")

        # Check if there's a next page by looking for "Next" link in pagination
        next_link = soup.find("a", string=re.compile(r"next", re.I))
        if not next_link:
            log.info("No 'Next' pagination link found. Catalog fully scraped.")
            break

        start += PAGE_SIZE
        time.sleep(DELAY_BETWEEN_PAGES)

    return all_assessments


def enrich_with_detail_pages(assessments: list[dict]) -> list[dict]:
    """
    Fetch each assessment's individual page and enrich the basic data.
    Uses a checkpoint: if output already exists, skip already-enriched items.
    """
    enriched = []
    total = len(assessments)

    log.info(f"Enriching {total} assessments with detail page data...")

    for i, assessment in enumerate(assessments, 1):
        url = assessment["url"]
        log.info(f"[{i}/{total}] Fetching detail: {assessment['name']}")
        
        soup = fetch(url)
        if soup:
            assessment = parse_detail_page(soup, assessment)
        else:
            log.warning(f"  Skipped detail page for: {assessment['name']}")
        
        enriched.append(assessment)
        
        if i < total:
            time.sleep(DELAY_BETWEEN_DETAIL)

    return enriched


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # Step 1: Scrape all listing pages
    assessments = scrape_all_listings()

    if not assessments:
        log.error("No assessments scraped. Check the catalog URL and HTML structure.")
        sys.exit(1)

    log.info(f"\n✓ Scraped {len(assessments)} Individual Test Solutions from listings.\n")

    # Step 2: Enrich with detail page data
    enriched = enrich_with_detail_pages(assessments)

    # Step 3: Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    log.info(f"\n✓ Saved {len(enriched)} assessments to {OUTPUT_PATH}")
    
    # Print a quick summary
    log.info("\n── Sample entries ──────────────────────────────────────────────")
    for item in enriched[:3]:
        log.info(f"  {item['name']}")
        log.info(f"    URL: {item['url']}")
        log.info(f"    Types: {item['test_types']}")
        log.info(f"    Remote: {item['remote_testing']} | Adaptive: {item['adaptive_irt']}")
        log.info(f"    Description: {item['description'][:120]}...")
        log.info("")


if __name__ == "__main__":
    main()
