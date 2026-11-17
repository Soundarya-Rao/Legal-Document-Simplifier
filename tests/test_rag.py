import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reconfigure stdout to UTF-8 so non-ASCII characters don't crash the Windows cp1252 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import configuration constants and backend pipeline functions
from src.config import APP_NAME, APP_TAGLINE, LANGUAGE, CHUNK_SIZE, CHUNK_OVERLAP
from src.pdf_reader import extract_text_from_pdf
from src.chunker import split_documents
from src.vector_store import build_vector_store
from src.rag_chain import answer_question
from src.analyzer import detect_document_type, summarize_document, find_risky_clauses

def main():
    pdf_path = os.path.join(PROJECT_ROOT, "data", "sample_docs", "rental_agreement.pdf")
    question = "What is the security deposit and when is it refunded?"
    
    print("=" * 60)
    print(f"Testing {APP_NAME} Pipeline - \"{APP_TAGLINE}\"")
    print("=" * 60)
    
    # Step 1: Extract text from the PDF
    print(f"\n[Step 1] Reading PDF: {pdf_path}")
    documents = extract_text_from_pdf(pdf_path)
    print(f" -> Successfully extracted {len(documents)} page(s).")
    
    # Step 2: Auto-detect document type and category
    print("\n[Step 2] Detecting Document Type and Category...")
    doc_info = detect_document_type(documents)
    print(f" -> Detected Type: {doc_info.get('doc_type')}")
    print(f" -> Category: {doc_info.get('category')}")
    print(f" -> Parties Identified: {', '.join(doc_info.get('parties', [])) or 'None'}")
    
    # Step 3: Summarize the entire document using detected doc_type
    doc_type = doc_info.get("doc_type", "Legal Document")
    category = doc_info.get("category", "other")
    
    print(f"\n[Step 3] Generating Structured Summary for {doc_type}...")
    summary = summarize_document(documents, language=LANGUAGE, doc_type=doc_type)
    print("\n" + "-" * 40)
    print("DOCUMENT SUMMARY:")
    print("-" * 40)
    print(summary)
    
    # Step 4: Scan for risky clauses using category-specific checklist
    print(f"\n[Step 4] Scanning for Risky Clauses (Category: {category})...")
    risky_clauses = find_risky_clauses(
        documents,
        language=LANGUAGE,
        doc_type=doc_type,
        category=category
    )
    print("\n" + "-" * 40)
    print(f"RISKY CLAUSES FOUND: {len(risky_clauses)}")
    print("-" * 40)
    for idx, item in enumerate(risky_clauses, start=1):
        print(f"\n{idx}. Clause: \"{item.get('clause')}\" [Page {item.get('page')}]")
        print(f"   Risk Level: {item.get('risk_level')}")
        print(f"   Why it matters: {item.get('why_it_matters')}")
        print(f"   What to ask: {item.get('what_to_ask')}")
        
    # Step 5: Split the document pages into smaller chunks for RAG
    print("\n[Step 5] Chunking text into passages for Q&A...")
    chunks = split_documents(documents)
    print(f" -> Created {len(chunks)} chunk(s) (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    
    # Step 6: Embed chunks and build the in-memory FAISS vector index
    print("\n[Step 6] Generating embeddings and building in-memory FAISS index...")
    vector_store = build_vector_store(chunks)
    print(" -> FAISS vector store ready in memory.")
    
    # Step 7: Run the RAG question answering chain
    print(f"\n[Step 7] Asking question: \"{question}\"")
    print(" -> Retrieving top 4 chunks and generating answer with Gemini...")
    answer = answer_question(
        vectorstore=vector_store,
        question=question,
        language=LANGUAGE,
        chat_history=[]
    )
    
    print("\n" + "=" * 60)
    print("RAG ANSWER:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()
