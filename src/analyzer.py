import json
import re
from langchain_core.documents import Document

# Import configuration, model fallback caller, and prompt templates
# Why: Guarantees consistent model fallback on 429 errors, avoids hardcoding API keys, and separates prompts from logic
from src.config import MAX_DIRECT_CHARS, invoke_with_model_fallback
from src.prompts import (
    DOC_TYPE_CHECKLISTS,
    DETECT_DOC_TYPE_PROMPT,
    DOCUMENT_SUMMARY_PROMPT,
    MERGE_SUMMARIES_PROMPT,
    RISKY_CLAUSES_PROMPT,
)

# Validated document categories
VALID_CATEGORIES = {
    "rental_lease",
    "employment",
    "loan_credit",
    "insurance",
    "property_sale",
    "business_contract_nda",
    "terms_privacy",
    "will_family",
    "legal_notice_court",
    "other",
}


def _format_full_document(documents: list[Document]) -> str:
    """
    Combines text from all pages into a single string with clear page labels.
    
    Why: Full document tasks (summarization and risk analysis) require page headers
    so the model can cite exact page numbers in its output.
    """
    page_blocks = []
    for doc in documents:
        page_num = doc.metadata.get("page", "Unknown")
        page_blocks.append(f"--- [Page {page_num}] ---\n{doc.page_content}")
    return "\n\n".join(page_blocks)


def _split_into_page_groups(documents: list[Document], max_chars: int = MAX_DIRECT_CHARS) -> list[list[Document]]:
    """
    Splits a large document's pages into smaller sequential groups that each fit within max_chars.
    
    Why: Very long contracts (> 60,000 characters) exceed reliable single-prompt attention spans;
    grouping pages enables reliable map-reduce analysis without loss of detail.
    """
    groups = []
    current_group = []
    current_length = 0

    for doc in documents:
        doc_len = len(doc.page_content)
        if current_group and (current_length + doc_len > max_chars):
            groups.append(current_group)
            current_group = [doc]
            current_length = doc_len
        else:
            current_group.append(doc)
            current_length += doc_len

    if current_group:
        groups.append(current_group)

    return groups


def _parse_json_safely(raw_text: str):
    """
    Safely extracts and parses JSON data (either an array or an object) from the model's output.
    
    Why: LLMs occasionally wrap JSON inside markdown fences (```json ... ```) or prefix text;
    this helper cleans fences and falls back to regex pattern matching before giving up.
    """
    cleaned = raw_text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback 1: Search for any JSON array [...]
        match_arr = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match_arr:
            try:
                return json.loads(match_arr.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback 2: Search for any JSON object {...}
        match_obj = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match_obj:
            try:
                return json.loads(match_obj.group(0))
            except json.JSONDecodeError:
                pass

        return None


def _call_llm_with_retry(prompt_text: str, temperature: float = 0.2) -> str:
    """
    Invokes Gemini with automatic 429 quota fallback to MODEL_FALLBACK_LIST and 503 backoff.
    
    Why: Centralizes LLM invocation so any rate limit (429 / RESOURCE_EXHAUSTED) automatically
    retries once with the next model (e.g. gemini-3.5-flash-lite) instead of failing.
    """
    return invoke_with_model_fallback(prompt_text, temperature=temperature)



def detect_document_type(documents: list[Document]) -> dict:
    """
    Analyzes the opening ~3 pages of a document to identify its type, category, and parties.
    
    Why: Knowing the category (e.g. employment vs loan) unlocks targeted risk checklists
    and avoids analyzing a 50-page document just to determine what kind of agreement it is.
    
    Returns:
        dict: {"doc_type": "...", "category": "...", "parties": ["..."]}
    """
    # Sample the first 3 pages (or fewer if the document is short)
    sample_pages = documents[:3]
    sample_text = _format_full_document(sample_pages)

    prompt = DETECT_DOC_TYPE_PROMPT.format(document_sample=sample_text)
    raw_output = _call_llm_with_retry(prompt, temperature=0.1)
    parsed = _parse_json_safely(raw_output)

    # Validate output structure; fall back gracefully to "other" if parsing fails
    if isinstance(parsed, dict) and "category" in parsed:
        cat = parsed["category"].strip().lower()
        if cat in VALID_CATEGORIES:
            return {
                "doc_type": str(parsed.get("doc_type", "Legal Agreement")).strip(),
                "category": cat,
                "parties": parsed.get("parties", []),
            }

    return {
        "doc_type": "Legal Document",
        "category": "other",
        "parties": [],
    }


def _is_horizontal_rule(line: str) -> bool:
    """Checks if a string is a markdown horizontal rule (e.g. '---', '***', '___', '- - -')."""
    return bool(re.match(r"^\s*([*\-_])(?:\s*\1){2,}\s*$", line))


def strip_summary_filler(text: str) -> str:
    """
    Strips conversational filler introductions (e.g. 'Here is', "Here's", 'Below is', 'Sure'),
    leading and trailing horizontal rules ('---', '***', '___'), and blank lines from summaries.
    
    Why: AI summaries often open with filler greetings or markdown dividers (---) before section 1.
    Stripping these ensures the summary card opens cleanly and directly at '1. What this document is',
    with no stray divider rules at the top or bottom.
    """
    if not text:
        return ""
    lines = text.strip().splitlines()
    filler_prefixes = ("here is", "here's", "below is", "sure")

    # 1. Strip leading filler lines, horizontal rules, and blank lines
    while lines:
        raw_line = lines[0].strip()
        if not raw_line or _is_horizontal_rule(raw_line):
            lines.pop(0)
            continue
        clean_first = re.sub(r"^[\*#_\"'\s]+", "", raw_line).strip().lower()
        if any(clean_first.startswith(prefix) for prefix in filler_prefixes):
            lines.pop(0)
            continue
        break

    # 2. Strip trailing horizontal rules and blank lines
    while lines:
        raw_line = lines[-1].strip()
        if not raw_line or _is_horizontal_rule(raw_line):
            lines.pop()
        else:
            break

    return "\n".join(lines).strip()


def summarize_document(
    documents: list[Document],
    language: str = "English",
    doc_type: str = "Legal Document"
) -> str:
    """
    Generates a structured, plain-language summary covering 7 core sections.
    Handles long documents (> MAX_DIRECT_CHARS) via page-group chunking and merge synthesis.
    
    Parameters:
        documents: List of LangChain Document objects representing the full PDF.
        language: Target language for the summary (defaults to "English").
        doc_type: Human-readable name of the document detected in step 1.
        
    Returns:
        Structured summary text covering 7 key pillars.
    """
    full_text = _format_full_document(documents)

    # Case A: Document fits within direct single-prompt limit
    if len(full_text) <= MAX_DIRECT_CHARS:
        prompt = DOCUMENT_SUMMARY_PROMPT.format(
            document_text=full_text,
            language=language,
            doc_type=doc_type,
        )
        raw_summary = _call_llm_with_retry(prompt, temperature=0.2)
        return strip_summary_filler(raw_summary)

    # Case B: Long document - segment by page groups and merge
    page_groups = _split_into_page_groups(documents, MAX_DIRECT_CHARS)
    partial_summaries = []

    for idx, group in enumerate(page_groups, start=1):
        group_text = _format_full_document(group)
        group_prompt = DOCUMENT_SUMMARY_PROMPT.format(
            document_text=group_text,
            language=language,
            doc_type=f"{doc_type} (Part {idx} of {len(page_groups)})",
        )
        part_summary = _call_llm_with_retry(group_prompt, temperature=0.2)
        partial_summaries.append(f"### Document Section {idx}\n{part_summary}")

    # Run final unified synthesis
    merge_prompt = MERGE_SUMMARIES_PROMPT.format(
        partial_summaries="\n\n---\n\n".join(partial_summaries),
        language=language,
        doc_type=doc_type,
    )
    raw_merged = _call_llm_with_retry(merge_prompt, temperature=0.2)
    return strip_summary_filler(raw_merged)


def find_risky_clauses(
    documents: list[Document],
    language: str = "English",
    doc_type: str = "Legal Document",
    category: str = "other"
) -> list[dict]:
    """
    Scans the document for potentially unfair or one-sided terms against domain-specific checklists.
    Handles long documents by scanning page groups, removing duplicate clauses, and sorting High-to-Low.
    
    Parameters:
        documents: List of LangChain Document objects representing the full PDF.
        language: Target language for explanations and advice questions (defaults to "English").
        doc_type: Human-readable document name.
        category: Document category key to retrieve relevant checklist.
        
    Returns:
        A list of dictionaries with keys: clause, page, risk_level, why_it_matters, what_to_ask.
    """
    checklist = DOC_TYPE_CHECKLISTS.get(category, DOC_TYPE_CHECKLISTS["other"])
    full_text = _format_full_document(documents)

    def _parse_clauses_list(raw_json) -> list[dict]:
        parsed = _parse_json_safely(raw_json)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            for key in ["clauses", "risky_clauses", "risks"]:
                if key in parsed and isinstance(parsed[key], list):
                    return [item for item in parsed[key] if isinstance(item, dict)]
            return [parsed]
        return []

    def _scan_prompt(prompt_str: str) -> list[dict]:
        raw_output = _call_llm_with_retry(prompt_str, temperature=0.1)
        return _parse_clauses_list(raw_output)

    # Case A: Document fits within single-prompt limit
    if len(full_text) <= MAX_DIRECT_CHARS:
        prompt = RISKY_CLAUSES_PROMPT.format(
            document_text=full_text,
            language=language,
            doc_type=doc_type,
            checklist=checklist,
        )
        clauses = _scan_prompt(prompt)
    else:
        # Case B: Long document - scan each page group separately
        page_groups = _split_into_page_groups(documents, MAX_DIRECT_CHARS)
        clauses = []
        for group in page_groups:
            group_text = _format_full_document(group)
            prompt = RISKY_CLAUSES_PROMPT.format(
                document_text=group_text,
                language=language,
                doc_type=doc_type,
                checklist=checklist,
            )
            clauses.extend(_scan_prompt(prompt))

    # Deduplicate clauses by normalized quote text
    seen_clauses = set()
    unique_clauses = []
    for item in clauses:
        clause_key = str(item.get("clause", "")).strip().lower()
        if clause_key and clause_key not in seen_clauses:
            seen_clauses.add(clause_key)
            unique_clauses.append(item)

    # Sort High first, then Medium, then Low
    risk_priority = {"High": 0, "Medium": 1, "Low": 2}
    sorted_clauses = sorted(
        unique_clauses,
        key=lambda x: risk_priority.get(str(x.get("risk_level", "Low")).capitalize(), 3)
    )

    return sorted_clauses
