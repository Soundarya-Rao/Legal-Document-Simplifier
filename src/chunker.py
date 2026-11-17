from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import configurable chunk settings from config
# Why: Rule requires keeping chunk size configurable in config.py (~1500 with overlap 200)
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Splits long Document objects into smaller chunks while preserving page metadata.
    
    Why chunking is necessary:
    LLMs have context limits and perform much better when fed concise, relevant passages
    rather than entire multi-page documents at once.
    
    Why chunk_size=1500:
    Reduces total chunk count for multi-page documents (~25 pages produces ~30-40 chunks
    instead of ~80-100 chunks), ensuring embedding requests stay safely below the
    100 requests per minute quota.
    
    Parameters:
        documents: A list of LangChain Document objects (typically one per page).
        chunk_size: Maximum character count per chunk (defaults to CHUNK_SIZE=1500).
        chunk_overlap: Overlapping character count (defaults to CHUNK_OVERLAP=200).
        
    Returns:
        A list of smaller LangChain Document chunks, each retaining its original page metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # split_documents automatically copies the metadata (such as {"page": n}) to every child chunk
    chunks = text_splitter.split_documents(documents)
    return chunks
