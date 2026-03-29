import os
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
    """Extracts text from a local PDF file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Please check your .env and folder.")
    
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def split_text(text, chunk_size=1000, overlap=150):
    """Simple chunking logic to stay within context limits."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

def run_rag_pipeline():
    # 1. Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="local_pdf_data")

    # 2. Ingest Data (Only if the collection is empty)
    if collection.count() == 0:
        print(f" Indexing {PDF_NAME} for the first time...")
        raw_text = get_pdf_text(PDF_NAME)
        chunks = split_text(raw_text)
        
        for i, chunk in enumerate(chunks):
            # Generate embedding via local Ollama
            embed = ollama.embed(model=EMBED_MODEL, input=chunk)['embeddings'][0]
            collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[embed],
                documents=[chunk]
            )
        print(f" Indexed {len(chunks)} chunks.")
    else:
        print(f" Found existing vector database with {collection.count()} chunks.")

    # 3. The Query Loop
    while True:
        query = input("\n Ask a question (or type 'exit'): ")
        if query.lower() == 'exit': break

        # A. Embed the query
        query_embed = ollama.embed(model=EMBED_MODEL, input=query)['embeddings'][0]
        
        # B. Retrieve top 2 matches
        results = collection.query(query_embeddings=[query_embed], n_results=2)
        context = "\n\n".join(results['documents'][0])

        # C. Generate Final Answer
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer using the context above:"
        print("\n thinking...")
        
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        print(f"\nResponse:\n{response['response']}")

if __name__ == "__main__":
    run_rag_pipeline()