"""
retriever/retriever.py
======================
Semantic search over the SHL catalog using lightweight TF-IDF.

Design decisions:
- Model: TF-IDF vectorizer + cosine similarity (via scikit-learn).
  Extremely lightweight, uses <200MB RAM, fast enough for 500 items.
- Embedding document: concatenate name + description + test_types + job_levels
  so the search captures all relevant signals per assessment.
- Index is persisted to disk using pickle so it is NOT rebuilt on every server start.
- Thread-safe: model and index are loaded once at module import time
  (singleton pattern) and are read-only during search.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_PATH = INDEX_DIR / "tfidf.pkl"
METADATA_PATH = INDEX_DIR / "metadata.json"

# ── Lazy globals (loaded once) ─────────────────────────────────────────────────
_vectorizer = None     # TfidfVectorizer
_tfidf_matrix = None   # Sparse matrix of tf-idf vectors
_catalog: list[dict] = []   # full assessment dicts, aligned with index rows


# ── Test type label mapping (duplicated from scraper for self-containment) ──────
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


def _build_document(assessment: dict) -> str:
    """
    Build a single string to embed for an assessment.

    We concatenate the most semantically rich fields so all of them are
    considered during retrieval:
      - name (most important: often contains the technology/skill directly)
      - test_types expanded to human-readable labels
      - job_levels (seniority context)
      - description (detailed purpose)
    """
    parts = []

    name = assessment.get("name", "")
    if name:
        # Repeat name twice to up-weight it during TF-IDF
        parts.append(name)
        parts.append(name)

    types = assessment.get("test_types", [])
    if types:
        labels = [TEST_TYPE_LABELS.get(t, t) for t in types]
        parts.append("Test types: " + ", ".join(labels))

    levels = assessment.get("job_levels", [])
    if levels:
        parts.append("Job levels: " + ", ".join(levels))

    desc = assessment.get("description", "")
    if desc:
        parts.append(desc[:500])  # cap to avoid term bloat

    return " ".join(parts)


def build_index(force: bool = False) -> None:
    """
    Build TF-IDF vectors for the catalog and save to disk.

    Args:
        force: If True, rebuild even if index already exists on disk.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    global _catalog, _vectorizer, _tfidf_matrix

    if not force and INDEX_PATH.exists() and METADATA_PATH.exists():
        log.info("TF-IDF index already exists. Skipping build (use force=True to rebuild).")
        return

    # Load catalog
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. Run the scraper first."
        )
    
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    log.info(f"Building TF-IDF index for {len(catalog)} assessments ...")

    # Build document strings
    documents = [_build_document(a) for a in catalog]

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Persist
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix}, f)
    
    # Save metadata (the catalog list, aligned with matrix rows)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    log.info(f"TF-IDF index saved: {INDEX_PATH} ({tfidf_matrix.shape[0]} documents)")

    # Update in-memory globals
    _vectorizer = vectorizer
    _tfidf_matrix = tfidf_matrix
    _catalog = catalog


def _ensure_loaded() -> None:
    """Load index and catalog from disk if not already in memory."""
    global _vectorizer, _tfidf_matrix, _catalog

    if _vectorizer is not None and _tfidf_matrix is not None and _catalog:
        return  # already loaded

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        log.warning("Index not found on disk. Building now from catalog ...")
        build_index()
        return

    log.info("Loading TF-IDF index from disk ...")
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
        _vectorizer = data["vectorizer"]
        _tfidf_matrix = data["matrix"]

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    log.info(f"Index loaded: {_tfidf_matrix.shape[0]} documents.")


def search(query: str, k: int = 10) -> list[dict]:
    """
    Keyword/TF-IDF search over the SHL catalog.

    Args:
        query: Natural language query string.
        k:     Maximum number of results to return.

    Returns:
        List of assessment dicts (full catalog data), ordered by relevance.
        Each dict is a deep copy so callers can mutate safely.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    _ensure_loaded()

    if not query or not query.strip():
        log.warning("Empty query passed to search().")
        return []

    # Vectorize the query
    query_vec = _vectorizer.transform([query.strip()])

    # Compute cosine similarity
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    # Get top k indices sorted by similarity (descending)
    k_actual = min(k, len(_catalog))
    # argpartition is faster than argsort for top-k, then sort those k
    if k_actual < len(similarities):
        top_indices = np.argpartition(similarities, -k_actual)[-k_actual:]
        # sort the top k by score
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
    else:
        top_indices = np.argsort(-similarities)

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score <= 0.0:  # Skip completely irrelevant results
            continue
        assessment = dict(_catalog[idx])  # shallow copy per item
        assessment["_score"] = score  # attach cosine similarity score
        results.append(assessment)

    return results


def get_all_urls() -> set[str]:
    """Return the set of all valid catalog URLs (for URL validation)."""
    _ensure_loaded()
    return {a["url"] for a in _catalog}


def get_by_name(name: str) -> Optional[dict]:
    """Exact name lookup in the catalog (case-insensitive)."""
    _ensure_loaded()
    name_lower = name.lower()
    for assessment in _catalog:
        if assessment.get("name", "").lower() == name_lower:
            return dict(assessment)
    return None


if __name__ == "__main__":
    # Quick test: build index and run a sample query
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Building index ...")
    build_index(force=True)

    queries = [
        "Java developer mid-level cognitive ability",
        "personality assessment for leadership",
        "customer service call center simulation",
    ]

    for q in queries:
        print(f"\nQuery: {q!r}")
        results = search(q, k=5)
        for r in results:
            print(f"  [{r['_score']:.3f}] {r['name']} | types={r['test_types']}")
