import pdfplumber
from langchain_core.documents import Document
from src.config import APP_NAME

def extract_text_from_pdf(file) -> list[Document]:
    """
    Extracts text from each page of a PDF using pdfplumber.
    
    Why pdfplumber?
    Unlike basic PDF parsers, pdfplumber accurately extracts text layout
    and character positioning, which is critical for complex Indian legal documents.
    
    Parameters:
        file: Either a file path (str or Path) or a file-like object (e.g. BytesIO from Streamlit upload).
        
    Returns:
        List of LangChain Document objects, one per page with metadata {"page": n}.
        
    Raises:
        ValueError: If no text could be extracted at all (indicating a scanned/image PDF).
    """
    documents = []
    
    # Open the PDF using pdfplumber (works with both file paths and in-memory byte streams)
    with pdfplumber.open(file) as pdf:
        # Loop through each page, using 1-based indexing so page numbers match human reading
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            
            # If the page contains readable text, store it as a LangChain Document
            # Why: Packaging text into Document objects allows LangChain's chunkers and vector stores
            # to track metadata (like page numbers) alongside the text content.
            if text and text.strip():
                doc = Document(
                    page_content=text.strip(),
                    metadata={"page": page_index}
                )
                documents.append(doc)
                
    # Calculate total extracted characters across all pages
    total_chars = sum(len(doc.page_content) for doc in documents)
    
    # If no text or fewer than 200 characters were extracted, alert the user
    # Why: Scanned contracts and image-only PDFs yield zero or only tiny garbage snippets without full OCR.
    if total_chars < 200:
        raise ValueError(
            "The uploaded PDF appears to be a scanned image or contains non-selectable text "
            f"(only {total_chars} readable character{'s' if total_chars != 1 else ''} found). "
            f"{APP_NAME} currently requires a digital PDF with selectable text."
        )
        
    return documents
