"""
RAG tools for the LangGraph chatbot.
- ingest_file(): processes uploaded files into the vector store
- rag_search: a tool the LLM calls to retrieve relevant document context
"""

import hashlib
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── Vector Store Setup ──────────────────────────────────────────────
# GoogleGenerativeAIEmbeddings converts text into numerical vectors
# ChromaDB stores these vectors locally so we can search by similarity
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="my_rag_docs",
    persist_directory="./chroma_rag_db",
)

# ── Text Splitter ───────────────────────────────────────────────────
# chunk_size=1000  → each chunk is ~1000 characters
# chunk_overlap=200 → chunks share 200 chars at boundaries
#   so we don't lose context at split points
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def ingest_file(file_path: str) -> dict:
    """
    Takes a file path, loads it, chunks it, embeds it, stores in ChromaDB.

    How it works:
    1. Detect file type (PDF or TXT) → use appropriate loader
    2. Split the loaded text into chunks
    3. Give each chunk a unique ID (hash of content + source)
       so we never store duplicates
    4. Only embed & store chunks that aren't already in the DB
    5. Return a summary of what happened

    Args:
        file_path: absolute path to the file to ingest

    Returns:
        dict with counts of total/existing/new chunks
    """
    # Step 1: Load the file
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return {"error": f"Unsupported file type: {ext}. Use PDF or TXT."}

    docs = loader.load()

    # Step 2: Chunk the documents
    chunks = text_splitter.split_documents(docs)

    # Step 3: Generate unique IDs to avoid duplicates
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["id"] = f"{source}_p{page}_{content_hash}"

    # Step 4: Filter out already-stored chunks
    existing_ids = set(vector_store.get()["ids"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    # Step 5: Store new chunks in batches
    if new_chunks:
        batch_size = 50
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            batch_ids = [doc.metadata["id"] for doc in batch]
            vector_store.add_documents(documents=batch, ids=batch_ids)

    return {
        "total_chunks": len(chunks),
        "already_stored": len(existing_ids),
        "new_chunks_added": len(new_chunks),
        "file": os.path.basename(file_path),
    }


# ── RAG Search Tool ────────────────────────────────────────────────
@tool
def rag_search(query: str) -> dict:
    """
    Search through uploaded documents to find relevant information.
    Use this tool when the user asks questions about documents they uploaded,
    or when you need context from their files.

    Args:
        query: the search query / question to look up in the documents

    Returns:
        dict with the query, matching context chunks, and their metadata
              (source file, page number)
    """
    # k=4 means retrieve top 4 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    results = retriever.invoke(query)

    if not results:
        return {
            "query": query,
            "context": "No relevant documents found. The user may not have uploaded any files yet.",
            "sources": [],
        }

    # Build context string from all retrieved chunks
    context_parts = []
    sources = []
    for i, doc in enumerate(results, 1):
        context_parts.append(f"[Chunk {i}]: {doc.page_content}")
        sources.append(
            {
                "chunk": i,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
            }
        )

    return {
        "query": query,
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }
