import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file in the project root
# Why: Keeps sensitive credentials like API keys out of source control
load_dotenv()

# Read the Gemini API key from the environment
# Why: Rule 5 requires never hardcoding the API key directly in code
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fail early if the API key is missing
# Why: Catch configuration problems immediately at import time instead of failing deep inside model calls
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. Please ensure it is defined in your .env file."
    )

APP_NAME = "Legal Document Simplifier"
APP_TAGLINE = "Understand what you're signing."

# Supported application language
# Why: English-only simplification while maintaining cache key compatibility
LANGUAGE = "English"

# Model fallback list for rate limit / quota management
# Why: On the Gemini API Free Tier, each model has its own daily request quota.
# If gemini-3.6-flash reaches 429 (RESOURCE_EXHAUSTED), falling back to gemini-3.5-flash-lite
# allows user workflows to continue without failing.
MODEL_FALLBACK_LIST = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]

# Centralized model identifier pointing to the default model
LLM_MODEL = MODEL_FALLBACK_LIST[0]

# The embedding model used to convert text chunks into vector representations
# Why: Standardized model identifier for generating vector embeddings across document simplification pipelines
EMBEDDING_MODEL = "models/gemini-embedding-2"

# Character threshold for direct single-prompt document analysis (~15-20 pages)
# Why: Contracts longer than this threshold are processed in segmented page groups to respect context limits
MAX_DIRECT_CHARS = 60000

# Local cache directory path for document summaries, risky clauses, and doc types
# Why: Caches analysis results to make zero Gemini calls when re-analyzing the same file and language
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")


# Text chunking configuration for RAG and vector store
# Why: chunk_size=1500 reduces the total number of chunks for long documents (~25 pages),
# keeping embedding API calls well below the 100 requests per minute rate limit.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Batching and throttling configuration for embedding generation
# Why: gemini-embedding-2 has a free-tier limit of 100 requests per minute.
# Batching in ~30 chunks with a short pause between batches ensures we stay within quota.
EMBEDDING_BATCH_SIZE = 30
EMBEDDING_PAUSE_SECONDS = 2.0


class ModelQuotaExhaustedError(Exception):
    """
    Raised when all models in MODEL_FALLBACK_LIST have exhausted their 429 quota.
    
    Why: Distinguishes daily rate-limit exhaustion from generic unexpected errors,
    enabling specific user-friendly error banners in the UI.
    """
    pass


def is_quota_error(exc: Exception) -> bool:
    """
    Detects whether an exception was caused by a 429 quota or RESOURCE_EXHAUSTED error.
    
    Why: Handles GoogleRateLimitError, ClientError, and generic exceptions containing
    429 or RESOURCE_EXHAUSTED strings across different Google GenAI / LangChain versions.
    """
    if isinstance(exc, ModelQuotaExhaustedError):
        return True
    err_str = str(exc).lower()
    return (
        "429" in err_str
        or "resource_exhausted" in err_str
        or "quota" in err_str
        or "resourcelimitexceeded" in err_str
    )


def get_quota_error_message(exc: Exception) -> str | None:
    """
    Inspects an exception for rate limit / quota details and returns a specific user-friendly message.
    
    Why:
    - If the error mentions "PerMinute" or "per minute":
      "The AI service is busy (per-minute limit). Please wait a minute and click Analyze again."
    - If the error mentions "PerDay" or "per day":
      "Daily AI usage limit reached. Please try again later."
    - Returns None if neither per-minute nor per-day is specifically identified.
    """
    err_str = str(exc).lower()
    if "perminute" in err_str or "per minute" in err_str:
        return "The AI service is busy (per-minute limit). Please wait a minute and click Analyze again."
    if "perday" in err_str or "per day" in err_str:
        return "Daily AI usage limit reached. Please try again later."
    return None


def extract_response_text(response) -> str:
    """
    Extracts plain text from the LLM response object.
    
    Why: langchain-google-genai can return either a plain string or a list of content blocks
    (e.g., [{'type': 'text', 'text': '...'}]). This helper normalizes it to a clean string.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


def invoke_with_model_fallback(prompt_text: str, temperature: float = 0.2) -> str:
    """
    Invokes Gemini with automatic 429 / RESOURCE_EXHAUSTED fallback across MODEL_FALLBACK_LIST.
    
    Why: If gemini-3.6-flash fails with 429 / RESOURCE_EXHAUSTED, retries once with the next
    model in MODEL_FALLBACK_LIST (gemini-3.5-flash-lite). If all models fail with 429,
    raises ModelQuotaExhaustedError with a clear description.
    Also handles transient 503 Service Unavailable spikes with exponential backoff.
    """
    last_quota_error = None

    for idx, model_name in enumerate(MODEL_FALLBACK_LIST):
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
            max_retries=2,
            timeout=120,
        )

        try:
            # Handle transient 503 errors on this model with up to 3 attempts
            for attempt_503 in range(3):
                try:
                    response = llm.invoke(prompt_text)
                    return extract_response_text(response)
                except Exception as e:
                    if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt_503 < 2:
                        time.sleep(2 * (attempt_503 + 1))
                        continue
                    raise e
        except Exception as exc:
            if is_quota_error(exc):
                last_quota_error = exc
                if idx + 1 < len(MODEL_FALLBACK_LIST):
                    next_model = MODEL_FALLBACK_LIST[idx + 1]
                    print(
                        f"[Model Fallback] {model_name} exhausted quota (429 / RESOURCE_EXHAUSTED). "
                        f"Retrying once with next model: {next_model}..."
                    )
                    continue
                else:
                    # Every model in the list is exhausted
                    raise ModelQuotaExhaustedError(
                        f"All models in fallback list ({MODEL_FALLBACK_LIST}) have exhausted their daily quota (429 / RESOURCE_EXHAUSTED)."
                    ) from exc
            else:
                # Other non-quota errors (e.g. prompt errors or network drop) - re-raise
                raise exc

    if last_quota_error:
        raise ModelQuotaExhaustedError(
            f"All models in fallback list ({MODEL_FALLBACK_LIST}) are exhausted."
        ) from last_quota_error

    raise RuntimeError("Unexpected end of model fallback loop without response or error.")
