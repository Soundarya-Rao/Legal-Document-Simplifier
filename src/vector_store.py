import re
import time
import contextlib
from typing import Callable, Any
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Import configuration constants and rate limit helpers
# Why: Rule 5 forbids hardcoded keys; config.py ensures a single source of truth
from src.config import (
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PAUSE_SECONDS,
    is_quota_error,
)


def extract_retry_delay(exc: Exception, default_delay: float = 30.0) -> float:
    """
    Extracts the recommended retry delay in seconds from a 429 / RESOURCE_EXHAUSTED error.
    
    Why: Google Generative AI errors include retryDelay in error text or attributes
    (e.g., retryDelay: "34s"). Respecting this delay prevents repetitive immediate 429 rejections.
    """
    err_str = str(exc)

    # 1. Look for retryDelay: "34s" or 'retryDelay': '34s' or retry_delay: 34
    match = re.search(
        r'retry[_-]?delay["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)\s*s?["\']?',
        err_str,
        re.IGNORECASE,
    )
    if match:
        try:
            val = float(match.group(1))
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    # 2. Look for "retry after X seconds" or "retry in Xs"
    match_after = re.search(
        r'retry\s+(?:after|in)\s+(\d+(?:\.\d+)?)\s*s',
        err_str,
        re.IGNORECASE,
    )
    if match_after:
        try:
            val = float(match_after.group(1))
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    # 3. Check for retry_delay attribute on the exception object
    if hasattr(exc, "retry_delay"):
        val = getattr(exc, "retry_delay")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
        if hasattr(val, "total_seconds"):
            sec = float(val.total_seconds())
            if sec > 0:
                return sec

    return default_delay


def build_vector_store(
    chunks: list[Document],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    pause_seconds: float = EMBEDDING_PAUSE_SECONDS,
    progress_callback: Callable[[int, int], Any] | None = None,
) -> FAISS:
    """
    Builds an in-memory FAISS vector index from document chunks in rate-limited batches.
    
    Why batching (batch_size ~30):
    gemini-embedding-2 has a free-tier rate limit of 100 requests per minute.
    Embedding chunks in batches of ~30 with pauses between batches prevents bursting
    the rate limit when indexing larger multi-page documents (~25 pages).
    
    Why retry with retryDelay:
    If a batch encounters a 429 RESOURCE_EXHAUSTED error, the error's retryDelay
    (defaulting to 30s) is respected, sleeping that long and retrying that batch up to 3 times.
    
    Why progress_callback:
    Allows Streamlit UI to show "Indexing part X of Y..." in a spinner while batches are processed.
    
    Parameters:
        chunks: List of LangChain Document chunks with text and page metadata.
        batch_size: Number of chunks per embedding batch (defaults to 30).
        pause_seconds: Short pause between batches to stay under 100 req/min (defaults to 2.0s).
        progress_callback: Optional callable or contextmanager called with (current_part, total_parts).
        
    Returns:
        FAISS vector store ready for similarity search.
    """
    if not chunks:
        raise ValueError("Cannot build vector store: chunks list is empty.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )

    vector_store: FAISS | None = None
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    max_retries = 3

    for i in range(num_batches):
        batch = chunks[i * batch_size : (i + 1) * batch_size]
        current_part = i + 1

        # Check if progress_callback is a contextmanager or regular callable
        cm = (
            progress_callback(current_part, num_batches)
            if progress_callback
            else contextlib.nullcontext()
        )
        if not hasattr(cm, "__enter__"):
            cm = contextlib.nullcontext()

        with cm:
            for attempt in range(max_retries + 1):
                try:
                    if vector_store is None:
                        vector_store = FAISS.from_documents(
                            documents=batch,
                            embedding=embeddings,
                        )
                    else:
                        vector_store.add_documents(documents=batch)
                    break  # Success!
                except Exception as exc:
                    if is_quota_error(exc) and attempt < max_retries:
                        delay = extract_retry_delay(exc, default_delay=30.0)
                        print(
                            f"[Vector Store] Rate limit (429) on batch {current_part}/{num_batches}. "
                            f"Sleeping {delay:.1f}s before retry {attempt + 1}/{max_retries}..."
                        )
                        time.sleep(delay)
                        continue
                    raise exc

        # Short pause between batches to stay under 100 requests per minute
        if current_part < num_batches and pause_seconds > 0:
            time.sleep(pause_seconds)

    return vector_store
