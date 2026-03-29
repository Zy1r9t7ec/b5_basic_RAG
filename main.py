import os
import hashlib
import ollama
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
EMBED_MODEL = os.getenv("EMBED_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
PDF_NAME = os.getenv("PDF_NAME")

def get_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    # Check if the reader sees all 46 pages
    print(f"DEBUG: Found {len(reader.pages)} pages in {path}") 
    
    for i, page in enumerate(reader.pages):
        try:
            content = page.extract_text()
            if content:
                text += content + "\n"
            else:
                # Log pages that return no text (common for images)
                print(f"Warning: No text found on page {i+1}")
        except Exception as e:
            print(f"Error reading page {i+1}: {e}")
    return text

    
def split_text(text, chunk_size=1000, overlap=250):
    """Fixed-size chunking with overlap. Filters out empty/whitespace-only chunks."""
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk.strip():  # Skip empty or whitespace-only chunks
            chunks.append(chunk)
    return chunks

def get_config_hash(pdf_path, chunk_size, overlap):
    """Returns a hash representing the current indexing config so we can detect stale DBs."""
    file_size = os.path.getsize(pdf_path)
    config_str = f"{pdf_path}|{file_size}|{chunk_size}|{overlap}"
    return hashlib.md5(config_str.encode()).hexdigest()

CHUNK_SIZE = 1000
OVERLAP = 250

def run_rag_pipeline():
    # 1. Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="local_pdf_data")

    # 2. Ingest Data — re-index if empty OR if config/file changed
    config_hash = get_config_hash(PDF_NAME, CHUNK_SIZE, OVERLAP)
    
    # Check if the stored hash matches current config
    meta = collection.metadata or {}
    stored_hash = meta.get("config_hash", "")
    needs_reindex = (collection.count() == 0) or (stored_hash != config_hash)

    if needs_reindex:
        if collection.count() > 0:
            print(f" Chunk settings or PDF changed — clearing stale index and re-indexing...")
            client.delete_collection(name="local_pdf_data")
            collection = client.create_collection(
                name="local_pdf_data",
                metadata={"config_hash": config_hash}
            )
        else:
            print(f" Indexing {PDF_NAME} for the first time...")
            # Stamp collection with config hash
            client.delete_collection(name="local_pdf_data")
            collection = client.create_collection(
                name="local_pdf_data",
                metadata={"config_hash": config_hash}
            )

        raw_text = get_pdf_text(PDF_NAME)
        chunks = split_text(raw_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        print(f" Splitting into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})...")
        
        for i, chunk in enumerate(chunks):
            # Generate embedding via local Ollama
            embed = ollama.embed(model=EMBED_MODEL, input=chunk)['embeddings'][0]
            collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[embed],
                documents=[chunk]
            )
            if (i + 1) % 10 == 0:
                print(f"   ... embedded {i+1}/{len(chunks)} chunks")
        print(f" Done! Indexed {len(chunks)} chunks.")
    else:
        print(f" Found existing vector database with {collection.count()} chunks (config unchanged).")

    # 3. The Query Loop
    while True:
        query = input("\n Ask a question (or type 'exit'): ")
        if query.lower() == 'exit': break

        # A. Embed the query
        query_embed = ollama.embed(model=EMBED_MODEL, input=query)['embeddings'][0]
        
        # B. Retrieve top 2 matches
        results = collection.query(query_embeddings=[query_embed], n_results=6)
        context = "\n\n".join(results['documents'][0])

        # C. Generate Final Answer
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer using the context above:"
        print("\n thinking...")
        
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        print(f"\nResponse:\n{response['response']}")

if __name__ == "__main__":
    run_rag_pipeline()