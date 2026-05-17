"""
retriever/retriever.py
======================
Semantic search over the SHL catalog using sentence-transformers + FAISS.

Design decisions:
- Model: all-MiniLM-L6-v2 (384-dim, fast, free, strong retrieval quality).
  Chosen over larger models because speed matters (30s API timeout) and
  the catalog is small enough that a lighter model is sufficient.
- Index: FAISS IndexFlatIP with L2-normalized vectors = cosine similarity.
  "Flat" (brute-force) is fine for <500 items; no approximation needed.
- Embedding document: concatenate name + description + test_types + job_levels
  so the search captures all relevant signals per assessment.
- Index is persisted to disk so it is NOT rebuilt on every server start (cold
  start would be too slow on Render free tier).
- Thread-safe: model and index are loaded once at module import time
  (singleton pattern) and are read-only during search.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# ── Lazy globals (loaded once) ─────────────────────────────────────────────────
_model = None          # SentenceTransformer model
_index = None          # faiss.Index
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
        # Repeat name twice to up-weight it during embedding
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
        parts.append(desc[:500])  # cap to avoid token overflow

    return " | ".join(parts)


def _load_model():
    """Load the sentence-transformer model (lazy, singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model all-MiniLM-L6-v2 ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embedding model loaded.")
    return _model


def build_index(force: bool = False) -> None:
    """
    Embed the catalog and save a FAISS index to disk.

    Args:
        force: If True, rebuild even if index already exists on disk.
    """
    import faiss

    global _catalog, _index

    if not force and INDEX_PATH.exists() and METADATA_PATH.exists():
        log.info("FAISS index already exists. Skipping build (use force=True to rebuild).")
        return

    # Load catalog
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. Run the scraper first."
        )
    
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    log.info(f"Building FAISS index for {len(catalog)} assessments ...")

    # Build embedding documents
    documents = [_build_document(a) for a in catalog]

    # Embed
    model = _load_model()
    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize → inner product = cosine sim
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]

    # Create FAISS flat index (exact cosine via inner product on normalized vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    
    # Save metadata (the catalog list, aligned with index rows)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    log.info(f"FAISS index saved: {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")

    # Update in-memory globals
    _index = index
    _catalog = catalog


def _ensure_loaded() -> None:
    """Load index and catalog from disk if not already in memory."""
    global _model, _index, _catalog
    import faiss

    if _index is not None and _catalog:
        return  # already loaded

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        log.warning("Index not found on disk. Building now from catalog ...")
        build_index()
        return

    log.info("Loading FAISS index from disk ...")
    _index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    _load_model()  # ensure model is loaded for future queries
    log.info(f"Index loaded: {_index.ntotal} vectors.")


def search(query: str, k: int = 10) -> list[dict]:
    """
    Semantic search over the SHL catalog.

    Args:
        query: Natural language query string.
        k:     Maximum number of results to return.

    Returns:
        List of assessment dicts (full catalog data), ordered by relevance.
        Each dict is a deep copy so callers can mutate safely.
    """
    _ensure_loaded()

    if not query or not query.strip():
        log.warning("Empty query passed to search().")
        return []

    model = _load_model()

    # Embed the query (must also be L2-normalized to match index)
    query_vec = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # FAISS search — returns distances and indices
    k_actual = min(k, _index.ntotal)
    distances, indices = _index.search(query_vec, k_actual)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        assessment = dict(_catalog[idx])  # shallow copy per item
        assessment["_score"] = float(dist)  # attach cosine similarity score
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
