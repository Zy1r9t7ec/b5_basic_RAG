
# Local PDF RAG Pipeline

A minimal, high-performance Retrieval-Augmented Generation (RAG) pipeline built to run entirely on local hardware. This project enables private, secure, and cost-free interaction with local PDF documents using ChromaDB for vector storage and Ollama for running Small Language Models (SLMs).

---

## Key Features
* **100% Local Execution**: All data processing and model inference occur on-device. This ensures that sensitive information from documents like the AI Agents Handbook remains private.
* **Smart Re-indexing**: Employs a configuration hash (MD5) to monitor changes in the source PDF or chunking parameters. This ensures the database is only rebuilt when necessary, saving computational resources.
* **Apple Silicon Optimization**: Specifically configured to utilize Metal acceleration on M-series MacBooks through the Ollama framework.
* **Persistent Vector Storage**: Document embeddings are committed to disk, allowing for immediate query capabilities upon restarting the application without re-processing the document.

---

## Tech Stack

### 1. Models (via Ollama)
* **LLM/SLM**: **llama3.2:3b** — A lightweight, high-performance 3-billion parameter model. It is used as the reasoning engine to synthesize answers based on retrieved context.
* **Embedding Model**: **nomic-embed-text** — A specialized model used to transform text into high-dimensional vectors, enabling semantic search capabilities.

### 2. Core Libraries
* **ChromaDB**: The primary vector database used to store, index, and query document embeddings.
* **PyPDF**: Utilized for extracting raw text from multi-page PDF documents.
* **Python-Dotenv**: Manages configuration and environment variables to keep the codebase clean.

---

## Pipeline Workflow

The system follows a standard RAG architecture divided into three logical phases:

### Phase 1: Ingestion and Chunking
The system extracts text from all 46 pages of the document. The text is then divided using a sliding window approach:
* **Chunk Size**: 1000 characters.
* **Overlap**: 250 characters. This overlap is critical for maintaining semantic continuity, ensuring that specific case studies are not fragmented across chunks.

### Phase 2: Vectorization and Storage
Each text fragment is passed to the local embedding model. The resulting vectors represent the "meaning" of the text. These are stored in ChromaDB, which creates a searchable index of the entire handbook.

### Phase 3: Retrieval and Augmented Generation
1. **Query Transformation**: The user's natural language question is converted into a vector.
2. **Similarity Search**: ChromaDB identifies the top **6** most relevant text segments based on vector distance.
3. **Prompt Augmentation**: The retrieved facts—such as the prediction that **15%** of work decisions will be autonomous by 2028—are inserted into a structured prompt.
4. **Grounded Generation**: The SLM generates a final response restricted strictly to the provided context to prevent hallucinations.

---

## Setup and Installation

1. **Install Ollama**: Download the application and pull the required models via terminal:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

2. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Create a `.env` file in the root directory:
   ```text
   EMBED_MODEL="nomic-embed-text"
   LLM_MODEL="llama3.2:3b"
   CHROMA_PATH="./chroma_db"
   PDF_NAME="AI Agents Handbook.pdf"
   ```

4. **Execution**:
   ```bash
   python main.py
   ```

---

## Known Failure Modes
* **Context Fragmentation**: Even with overlap, complex data tables or very long sentences may be split, causing a loss of context.
* **Small Model Bias**: A 3B model may occasionally struggle with highly complex reasoning compared to larger models, though it excels at direct fact extraction from context.
* **Parser Limitations**: Standard text extraction may fail on image-heavy pages or non-standard PDF encodings, potentially leading to missing information from certain sections of the handbook.

---

