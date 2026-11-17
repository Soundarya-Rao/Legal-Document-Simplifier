import os
import re
import json
import hashlib
from langchain_core.documents import Document
from src.config import CACHE_DIR, LANGUAGE

# Ensure cache directory exists when module loads
os.makedirs(CACHE_DIR, exist_ok=True)


def calculate_file_hash(file_input: bytes | str) -> str:
    """
    Computes a SHA-256 hash from raw file bytes or a file path.
    
    Why: Using cryptographic content hashing guarantees cache hits even if the uploaded
    file is renamed, while ensuring any modified document produces a fresh cache key.
    """
    if isinstance(file_input, str):
        with open(file_input, "rb") as f:
            content = f.read()
    elif hasattr(file_input, "getvalue"):
        content = file_input.getvalue()
    elif hasattr(file_input, "read"):
        content = file_input.read()
    elif isinstance(file_input, (bytes, bytearray)):
        content = bytes(file_input)
    else:
        content = str(file_input).encode("utf-8")

    return hashlib.sha256(content).hexdigest()


def _normalize_key(file_hash: str, language: str, doc_type: str = "") -> str:
    """
    Constructs a filesystem-safe composite key from (file content hash, language, doc type).
    
    Why: Keeps cache files organized, deterministic, and safe across Windows/Linux filesystems.
    """
    safe_hash = file_hash.strip().lower()
    safe_lang = language.strip().lower()
    safe_doc = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_type.strip().lower()) if doc_type else ""
    if safe_doc:
        return f"{safe_hash}_{safe_lang}_{safe_doc}"
    return f"{safe_hash}_{safe_lang}"


def save_cached_stage(
    file_hash: str,
    language: str = LANGUAGE,
    doc_info: dict | None = None,
    summary: str | None = None,
    risky_clauses: list[dict] | None = None,
) -> str:
    """
    Saves or updates individual analysis stages (doc_info, summary, risky_clauses)
    to the local JSON cache as soon as each finishes.
    
    Why: If downstream processing (e.g. vector store embedding) hits a rate limit or fails,
    all previously completed Gemini results remain cached. Retrying skips all completed
    Gemini calls and only redoes the failed stage.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load existing cached data if present to preserve previously completed stages
    existing = get_cached_analysis(file_hash, language) or {}
    
    current_doc_info = (
        doc_info if doc_info is not None
        else existing.get("doc_info", {"doc_type": "Legal Document", "category": "other", "parties": []})
    )
    current_summary = summary if summary is not None else existing.get("summary", "")
    current_risks = risky_clauses if risky_clauses is not None else existing.get("risky_clauses", [])
    
    doc_type = current_doc_info.get("doc_type", "Legal Document") if isinstance(current_doc_info, dict) else str(current_doc_info)
    
    payload = {
        "file_hash": file_hash,
        "language": language,
        "doc_type": doc_type,
        "doc_info": current_doc_info,
        "summary": current_summary,
        "risky_clauses": current_risks,
    }

    # Save under base key (file_hash, language)
    base_key = _normalize_key(file_hash, language)
    base_cache_path = os.path.join(CACHE_DIR, f"{base_key}.json")
    with open(base_cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Also save under primary composite key: (file content hash, language, doc type)
    if doc_type:
        key = _normalize_key(file_hash, language, doc_type)
        composite_path = os.path.join(CACHE_DIR, f"{key}.json")
        if composite_path != base_cache_path:
            with open(composite_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    return base_cache_path


def save_cached_analysis(
    file_hash: str,
    language: str = LANGUAGE,
    doc_info: dict = None,
    summary: str = "",
    risky_clauses: list[dict] = None
) -> str:
    """
    Saves the summary, risky clauses, and detected doc type to a local JSON cache
    keyed by (file content hash, language, doc type).
    
    Why: Persists expensive Gemini analysis results locally so that re-analyzing
    the same document in the same language loads instantaneously with zero API calls.
    Delegates to save_cached_stage for consistent storage.
    """
    return save_cached_stage(
        file_hash=file_hash,
        language=language,
        doc_info=doc_info,
        summary=summary,
        risky_clauses=risky_clauses,
    )


def get_cached_analysis(
    file_hash: str,
    language: str = LANGUAGE,
    doc_type: str = None
) -> dict | None:
    """
    Retrieves analysis results from the local JSON cache for (file content hash, language, doc type).
    If doc_type is omitted, looks up the cache by (file_hash, language).
    
    Why: Enables checking the cache BEFORE any LLM call, ensuring that re-analyzing
    the same file in the same language makes zero Gemini calls.
    
    Returns:
        dict containing 'doc_info', 'summary', 'risky_clauses' if cached, or None on cache miss.
    """
    if not os.path.exists(CACHE_DIR):
        return None

    # Check exact match if doc_type was provided
    if doc_type:
        key = _normalize_key(file_hash, language, doc_type)
        exact_path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(exact_path):
            try:
                with open(exact_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    # Check base key (file_hash, language)
    base_key = _normalize_key(file_hash, language)
    base_path = os.path.join(CACHE_DIR, f"{base_key}.json")
    if os.path.exists(base_path):
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Search for any cache file matching file_hash and language prefix
    prefix = f"{file_hash.strip().lower()}_{language.strip().lower()}_"
    try:
        for fname in os.listdir(CACHE_DIR):
            if fname.startswith(prefix) and fname.endswith(".json"):
                fpath = os.path.join(CACHE_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception:
        pass

    return None


def get_cached_vector_store(file_hash: str, embeddings):
    """
    Loads an in-memory FAISS vector index from disk if previously saved for this file.
    
    Why: Bypasses Google embedding API calls entirely when a document is re-analyzed,
    ensuring 100% zero Gemini calls during cache hits.
    """
    faiss_dir = os.path.join(CACHE_DIR, f"{file_hash[:16]}_faiss")
    if os.path.exists(faiss_dir):
        try:
            from langchain_community.vectorstores import FAISS
            return FAISS.load_local(
                faiss_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            return None
    return None


def save_cached_vector_store(file_hash: str, vector_store) -> None:
    """
    Persists an in-memory FAISS index to the local .cache/ folder.
    
    Why: Reusable for future Q&A sessions on the same document without re-embedding.
    """
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        faiss_dir = os.path.join(CACHE_DIR, f"{file_hash[:16]}_faiss")
        vector_store.save_local(faiss_dir)
    except Exception:
        pass
