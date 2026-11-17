import re
from langchain_community.vectorstores import FAISS

# Import model fallback invoker and prompt template
# Why: Automatically falls back from gemini-3.6-flash to gemini-3.5-flash-lite on 429 quota exhaustion
from src.config import invoke_with_model_fallback
from src.prompts import LEGAL_QA_PROMPT


def normalize_citations(text: str) -> str:
    """
    Normalizes citation formats in answer text into consistent single brackets:
    - [[Page 6]] -> [Page 6]
    - [Pages 6] -> [Page 6]
    - [[Page 6, 7]] -> [Pages 6, 7]
    - [Pages 6-7] / [[Pages 6-7]] -> [Pages 6, 7]
    
    Why: Models may output inconsistent double brackets or singular/plural mismatches.
    This ensures clean, uniform [Page X] or [Pages X, Y] citations for the user.
    """
    if not text:
        return ""

    def replace_citation(match):
        inner = match.group(1).strip()
        # Replace 'and' or '&' with comma
        clean_inner = re.sub(r"\s*(?:and|&)\s*", ",", inner, flags=re.IGNORECASE)
        parts = [p.strip() for p in clean_inner.split(",") if p.strip()]
        pages = []
        for part in parts:
            range_match = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", part)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if 0 <= end - start <= 10:
                    pages.extend(range(start, end + 1))
                else:
                    pages.append(f"{start}-{end}")
            elif part.isdigit():
                pages.append(int(part))
            else:
                pages.append(part)

        # Remove duplicates while preserving order
        unique_pages = []
        for p in pages:
            if p not in unique_pages:
                unique_pages.append(p)

        if len(unique_pages) == 1:
            return f"[Page {unique_pages[0]}]"
        elif len(unique_pages) > 1:
            page_str = ", ".join(str(p) for p in unique_pages)
            return f"[Pages {page_str}]"
        return match.group(0)

    # Match [[Page 6]], [Page 6], [[Pages 6, 7]], [Pages 6-7], etc.
    pattern = r"\[\[?\s*Pages?\s*([0-9\s,\-–and&]+?)\s*\]\]?"
    return re.sub(pattern, replace_citation, text, flags=re.IGNORECASE)


def answer_question(
    vectorstore: FAISS,
    question: str,
    language: str = "English",
    chat_history: list = None
) -> str:
    """
    Retrieves the top 4 relevant chunks from the FAISS vector store and generates
    a simplified, factual answer citing page numbers.
    Automatically retries with fallback models if 429 quota is encountered.
    
    Why top 4 chunks (k=4)?
    4 chunks (~4000 characters) provide sufficient context for specific clauses
    without overwhelming the prompt or introducing irrelevant text.
    
    Parameters:
        vectorstore: The FAISS vector store containing indexed PDF chunks.
        question: The user's query about the legal document.
        language: The target language for the explanation (defaults to "English").
        chat_history: Optional list of previous chat messages for conversational context.
        
    Returns:
        The simplified answer text from Gemini with normalized citations.
    """
    # 1. Retrieve the top 4 most semantically relevant chunks for the question
    # Why: Similarity search finds the exact paragraphs related to the question across all pages
    relevant_docs = vectorstore.similarity_search(question, k=4)

    # 2. Format the retrieved chunks with clear page markers
    # Why: The LLM needs explicit page tags so it can accurately cite where each fact originated
    context_blocks = []
    for doc in relevant_docs:
        page_num = doc.metadata.get("page", "Unknown")
        context_blocks.append(f"[Page {page_num}]\n{doc.page_content}")
    formatted_context = "\n\n".join(context_blocks)

    # 3. Format the chat history into a readable transcript
    # Why: Multi-turn chat allows users to ask follow-up questions referencing earlier answers
    if not chat_history:
        formatted_history = "No previous conversation."
    elif isinstance(chat_history, str):
        formatted_history = chat_history
    else:
        history_lines = []
        for item in chat_history:
            if isinstance(item, tuple) and len(item) == 2:
                history_lines.append(f"User: {item[0]}\nAssistant: {item[1]}")
            elif isinstance(item, dict) and "role" in item and "content" in item:
                history_lines.append(f"{item['role'].capitalize()}: {item['content']}")
            else:
                history_lines.append(str(item))
        formatted_history = "\n".join(history_lines)

    # 4. Build the complete prompt using our predefined template
    prompt_text = LEGAL_QA_PROMPT.format(
        context=formatted_context,
        chat_history=formatted_history,
        question=question,
        language=language,
    )

    # 5. Generate answer with model fallback on 429 and normalize citations
    raw_answer = invoke_with_model_fallback(prompt_text, temperature=0.2)
    return normalize_citations(raw_answer)
