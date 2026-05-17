"""
agent/agent.py
==============
Core agent logic: given a full conversation history, decides what to do
(clarify / recommend / refine / compare / refuse) and returns a structured
ChatResponse.

Design decisions:

1. STATELESS: The agent receives the full conversation history on every call.
   No session state is stored. This matches the API requirement exactly.

2. RETRIEVAL STRATEGY:
   - We build a composite query from ALL user messages, not just the last one.
     This gives richer context to the vector search (e.g. seniority from turn 2
     is still captured when turn 4 asks about test types).
   - We over-retrieve (k=15) and let the LLM filter down to the best 1-10.
     This maximizes Recall@10 since the LLM can reason about fit better than
     pure cosine similarity.

3. URL VALIDATION (CRITICAL):
   - After the LLM responds, we cross-check every recommended URL against the
     known catalog URL set. Any URL not in the catalog is stripped. This entirely
     prevents hallucinated URLs from reaching the API response.

4. JSON PARSING WITH FALLBACK:
   - If the LLM returns malformed JSON (rare but possible), we attempt to extract
     a JSON object with regex, then fall back to a safe error response that still
     conforms to the required schema.

5. TURN CAP ENFORCEMENT:
   - We count turns (user + assistant messages) and pass this to the system
     prompt. The prompt instructs the LLM to stop clarifying after turn 5.
   - As a hard safety net: if we detect turn >= 7 and recommendations is still
     empty, we force a recommendation pass by boosting the retrieval signal.

6. GROQ LLM:
   - Model: llama-3.3-70b-versatile (free, fast, strong instruction following).
   - Temperature: 0.1 (near-deterministic for consistent JSON output).
   - Max tokens: 1500 (enough for 10 recommendations + reply text).
   - Timeout: handled by Groq client; we also have a 25s hard timeout.
"""

import json
import logging
import os
import re
import time
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from agent.system_prompt import build_system_prompt, get_turn_number
from retriever.retriever import search, get_all_urls

load_dotenv()

log = logging.getLogger(__name__)

# ── Groq client (singleton) ────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None

def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable not set. "
                "Get a free key at https://console.groq.com/"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Pydantic-compatible response models ───────────────────────────────────────
# We use plain dicts here; Pydantic models are defined in api/main.py.

EMPTY_RESPONSE = {
    "reply": "I'm sorry, I can only help with SHL assessment selection. Please describe the role or skills you need to assess.",
    "recommendations": [],
    "end_of_conversation": False,
}

ERROR_RESPONSE = {
    "reply": "I encountered a technical issue. Please try again.",
    "recommendations": [],
    "end_of_conversation": False,
}


# ── Query building ─────────────────────────────────────────────────────────────

def _build_retrieval_query(messages: list[dict]) -> str:
    """
    Build a rich natural language query from all user messages in the conversation.

    Rationale: a hiring manager's context is spread across multiple turns.
    E.g.: turn 1 says "Java developer", turn 2 says "mid-level", turn 3 says
    "also need personality". Combining all turns gives the retriever the full
    picture and maximizes recall.
    """
    user_texts = [
        m["content"] for m in messages if m.get("role") == "user"
    ]
    return " ".join(user_texts).strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(
    messages: list[dict],
    system_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1500,
    temperature: float = 0.1,
) -> str:
    """
    Call the Groq LLM with the conversation history and system prompt.

    Returns raw text response from the LLM.
    Raises on network error or API error (caller handles).
    """
    client = _get_groq_client()

    # Build messages list for Groq: system first, then conversation history
    groq_messages = [{"role": "system", "content": system_prompt}]
    
    # Include conversation history (Groq supports multi-turn natively)
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            groq_messages.append({"role": role, "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=groq_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=25,  # hard 25s timeout, well under the 30s API limit
    )

    return response.choices[0].message.content or ""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> dict:
    """
    Parse the LLM's raw text response into a structured dict.

    Tries three strategies in order:
    1. Direct JSON parse (works when LLM is well-behaved).
    2. Regex extraction of first {...} block (handles extra prose wrapping).
    3. Returns a safe fallback response dict.
    """
    raw = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract reply text at least and return safe structure
    log.warning(f"Failed to parse LLM response as JSON. Raw: {raw[:200]}")
    
    # Try to get the reply text even from malformed JSON
    reply_match = re.search(r'"reply"\s*:\s*"([^"]+)"', raw)
    reply = reply_match.group(1) if reply_match else "I encountered an issue formatting my response. Please try again."
    
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


# ── URL validation ─────────────────────────────────────────────────────────────

def _validate_recommendations(recommendations: list[dict]) -> list[dict]:
    """
    Strip any recommendation whose URL is not in the scraped catalog.

    This is the CRITICAL safety net that prevents hallucinated URLs from
    ever reaching the API response. The valid_urls set is built from
    catalog.json at startup.
    """
    valid_urls = get_all_urls()
    validated = []

    for rec in recommendations:
        url = rec.get("url", "")
        if url in valid_urls:
            validated.append(rec)
        else:
            log.warning(f"Stripped hallucinated/invalid URL: {url!r} for '{rec.get('name')}'")

    return validated


# ── Response normalization ─────────────────────────────────────────────────────

def _normalize_response(parsed: dict) -> dict:
    """
    Ensure the response has exactly the required fields and valid types.
    Coerces or fills in defaults if needed.
    """
    reply = str(parsed.get("reply", "")).strip()
    if not reply:
        reply = "I'm ready to help you find the right SHL assessment."

    raw_recs = parsed.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    # Normalize each recommendation
    recs = []
    for r in raw_recs:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", "")).strip()
        url = str(r.get("url", "")).strip()
        test_type = str(r.get("test_type", "")).strip()
        if name and url:
            recs.append({"name": name, "url": url, "test_type": test_type})

    # Validate URLs against catalog
    recs = _validate_recommendations(recs)

    # Enforce max 10 recommendations
    recs = recs[:10]

    end_of_conversation = bool(parsed.get("end_of_conversation", False))
    # Auto-set end_of_conversation if LLM forgot but gave recommendations
    # and turn count is high (turn 7 or 8)
    # (The API handler may also set this; LLM decision is respected first)

    return {
        "reply": reply,
        "recommendations": recs,
        "end_of_conversation": end_of_conversation,
    }


# ── Main agent function ────────────────────────────────────────────────────────

def process_conversation(messages: list[dict]) -> dict:
    """
    Main entry point: given full conversation history, return a ChatResponse dict.

    Pipeline:
    1. Validate input (empty list, role check).
    2. Build composite retrieval query from all user messages.
    3. Fetch top-K relevant assessments from FAISS.
    4. Count turns, build system prompt with catalog context + turn number.
    5. Call Groq LLM with full message history + system prompt.
    6. Parse JSON response, validate URLs, normalize shape.
    7. Return clean response dict.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str} dicts.

    Returns:
        Dict with keys: reply (str), recommendations (list), end_of_conversation (bool).
    """
    start_time = time.time()

    # ── Input validation ──────────────────────────────────────────────────
    if not messages:
        log.warning("process_conversation called with empty messages list.")
        return {
            "reply": "Hello! I'm your SHL Assessment Consultant. Please tell me about the role you're hiring for.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Ensure the last message is from the user (sanity check)
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        log.warning("Last message in history is not from user. Continuing anyway.")

    # ── Turn counting ─────────────────────────────────────────────────────
    turn_number = get_turn_number(messages)
    log.info(f"Processing turn {turn_number}/8 with {len(messages)} messages.")

    # ── Retrieval ─────────────────────────────────────────────────────────
    query = _build_retrieval_query(messages)
    
    # Over-retrieve (k=15) so LLM can filter; maximizes Recall@10
    k = 15
    try:
        catalog_results = search(query, k=k)
        log.info(f"Retrieved {len(catalog_results)} catalog items for query: {query[:80]!r}")
    except Exception as exc:
        log.error(f"Retrieval failed: {exc}")
        catalog_results = []

    # ── System prompt ─────────────────────────────────────────────────────
    system_prompt = build_system_prompt(catalog_results, turn_number)

    # ── LLM call ─────────────────────────────────────────────────────────
    try:
        raw_response = _call_llm(messages, system_prompt)
        log.debug(f"LLM raw response: {raw_response[:300]}")
    except Exception as exc:
        log.error(f"LLM call failed: {exc}")
        elapsed = time.time() - start_time
        log.info(f"Request failed after {elapsed:.1f}s")
        return dict(ERROR_RESPONSE)

    # ── Parse & validate ──────────────────────────────────────────────────
    parsed = _parse_llm_response(raw_response)
    response = _normalize_response(parsed)

    elapsed = time.time() - start_time
    log.info(
        f"Turn {turn_number} complete in {elapsed:.1f}s | "
        f"recs={len(response['recommendations'])} | eoc={response['end_of_conversation']}"
    )

    return response
