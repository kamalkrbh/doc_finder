# Document Finder (Doc Finder)

Doc Finder is a Python-based Retrieval-Augmented Generation (RAG) system designed to help users find relevant information within a collection of PDF documents. It indexes the content of PDFs and uses a Large Language Model (LLM) to understand user queries and suggest the most relevant documents.

## Features

*   **PDF Loading:** Loads text content from PDF files located in a specified directory.
*   **Content Indexing:** Creates vector embeddings of the entire text content of each PDF using Hugging Face sentence transformers (`BAAI/bge-small-en-v1.5` by default).
*   **Vector Storage:** Uses FAISS (Facebook AI Similarity Search) to store and efficiently search document embeddings.
*   **LLM Integration:** Supports different LLM providers (Groq, Ollama) via LangChain for understanding queries and generating responses.
*   **Relevant Document Suggestion:** Takes a user's natural language query, finds the most semantically similar PDFs in the index, and uses the LLM to suggest which files the user should consult.
*   **Dummy Data Generation:** Includes a script (`pdf_gen.py`) to generate sample PDF documents (short stories) using an LLM, useful for testing.
*   **Configurable:** Key settings like directories, LLM provider, models, and API keys can be configured.

## Architecture Overview

The system consists of several core components:

1.  **`DataLoader` (`data_loader.py`):** Scans the specified PDF directory (`pdfs/` by default), extracts the full text content from each PDF using `PyPDFLoader`, and creates one LangChain `Document` object per PDF file.
2.  **`Embedder` (`embedder.py`):** Initializes the Hugging Face embedding model used to convert document text into numerical vectors.
3.  **`Indexer` (`indexer.py`):**
    *   Manages the FAISS vector store.
    *   Creates a new index from the loaded `Document` objects if one doesn't exist.
    *   Saves the index to disk (`index/` directory by default).
    *   Loads an existing index from disk.
    *   Performs similarity searches on the index using query embeddings.
4.  **`LLMHandler` (`llm_handler.py`):**
    *   Provides a consistent interface for interacting with the chosen LLM (Groq or Ollama).
    *   Handles the configuration and initialization of the LLM client based on environment variables.
5.  **`QueryEngine` (`query_engine.py`):**
    *   Orchestrates the query process.
    *   Takes a user query.
    *   Uses the `Indexer` to find potentially relevant PDF documents based on semantic similarity.
    *   Constructs a prompt for the `LLMHandler` containing the user query and the filenames of the relevant PDFs.
    *   Returns the LLM's suggestion on which files are most likely to contain the answer.
6.  **`Config` (`config.py`):** Centralizes configuration settings, pulling from environment variables where appropriate (e.g., API keys, LLM provider).
7.  **`main.py`:** The main entry point for the application. It initializes components, sets up the index (creating it if necessary), and runs the interactive query loop.
8.  **`pdf_gen.py`:** An auxiliary script to generate dummy PDF files using an LLM for populating the `pdfs/` directory.

## Setup and Installation

**Prerequisites:**

*   Python 3.8+
*   `pip` (Python package installer)
*   Access to an LLM (either a local Ollama instance or a Groq API key)

**Steps:**

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd doc_finder
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the necessary libraries:
    ```txt
    # requirements.txt
    langchain
    langchain-community
    langchain-huggingface
    langchain-groq # If using Groq
    langchain-ollama # If using Ollama
    faiss-cpu # Or faiss-gpu if you have CUDA setup
    pypdf
    fpdf2 # For pdf_gen.py
    python-dotenv # To load .env file
    # Add any other specific dependencies if needed
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project's root directory (`doc_finder/`) to store sensitive information and configuration overrides:

    ```dotenv
    # .env file

    # --- LLM Configuration ---
    # Set ONE of these (groq or ollama)
    LLM_PROVIDER="groq"
    # LLM_PROVIDER="ollama"

    # --- Model Selection ---
    # For Groq (see Groq Cloud for available models)
    MODEL="llama-3.1-8b-instant"
    # For Ollama (ensure the model is pulled, e.g., `ollama pull llama3`)
    # MODEL="llama3"

    # --- API Keys ---
    # Required only if LLM_PROVIDER is "groq"
    GROQ_API_KEY="your_groq_api_key_here"

    # --- Optional: Override config.py defaults ---
    # PDF_DIRECTORY="my_pdfs"
    # INDEX_DIRECTORY="my_index"
    # EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" # Example alternative
    ```

    *   Replace `"your_groq_api_key_here"` with your actual Groq API key if using Groq.
    *   Make sure the `MODEL` specified is compatible with the chosen `LLM_PROVIDER`. If using Ollama, ensure you have pulled the model locally (e.g., `ollama run llama3`).

## Usage

1.  **Place PDFs:** Put the PDF documents you want to index into the directory specified by `PDF_DIRECTORY` in `config.py` (defaults to `pdfs/`). Create the directory if it doesn't exist.

2.  **Generate Dummy PDFs (Optional):**
    If you don't have PDFs, you can generate some dummy story PDFs using the included script.
    *   Review the configuration constants at the top of `ai_app/pdf_gen.py` (e.g., `NUM_PDFS_TO_CREATE`).
    *   Ensure your LLM environment variables (`.env`) are set correctly, as this script also uses the LLM.
    *   Run the script from the project's root directory (`doc_finder/`):
        ```bash
        python ai_app/pdf_gen.py
        ```
    *   This will create PDF files in the `PDF_DIRECTORY`.

3.  **Run the Application:**
    Execute the main script from the project's root directory (`doc_finder/`):
    ```bash
    python ai_app/main.py
    ```

4.  **Indexing:**
    *   On the first run, if the index directory (`index/` by default) is empty or doesn't contain valid FAISS index files, the application will:
        *   Load all PDFs from the `PDF_DIRECTORY`.
        *   Generate embeddings for each PDF's content.
        *   Create a FAISS index.
        *   Save the index files (`faiss_index.faiss` and `faiss_index.pkl`) to the `INDEX_DIRECTORY`.
    *   On subsequent runs, it will detect the existing index files and load them directly, which is much faster.

5.  **Querying:**
    *   Once the index is ready, you'll see a prompt:
        ```
        Enter your query (or 'exit' to quit):
        ```
    *   Type your question or topic related to the content of your PDFs and press Enter.
    *   The system will:
        *   Search the FAISS index for PDFs with content similar to your query.
        *   Send the query and the list of potentially relevant PDF filenames to the configured LLM.
        *   Display the list of potentially relevant filenames found by the indexer.
        *   Display the LLM's suggestion about which of those files you should check and why.
    *   Type `exit` to quit the application.

## Dependencies

*   **LangChain:** Core framework for building RAG applications.
    *   `langchain-community`: Community integrations (loaders, vector stores).
    *   `langchain-huggingface`: Hugging Face embeddings integration.
    *   `langchain-groq` / `langchain-ollama`: Specific LLM integrations.
*   **FAISS (`faiss-cpu` or `faiss-gpu`):** Library for efficient similarity search and vector storage.
*   **Sentence Transformers (via `langchain-huggingface`):** Used for generating text embeddings (specifically `BAAI/bge-small-en-v1.5` by default).
*   **PyPDF (`pypdf`):** Used by `PyPDFLoader` to extract text from PDF files.
*   **FPDF2 (`fpdf2`):** Used by `pdf_gen.py` to create PDF files.
*   **python-dotenv:** For loading environment variables from a `.env` file.

## Future Improvements

*   **OCR Integration:** Add support for extracting text from image-based PDFs using OCR (like Tesseract via `UnstructuredFileLoader` or similar).
*   **Chunking Strategy:** Implement text chunking *before* indexing if dealing with very large documents or if finer-grained retrieval is needed. Currently, it embeds the entire PDF content as one vector.
*   **More Vector Stores:** Add support for other vector databases (e.g., ChromaDB, Pinecone, Weaviate).
*   **Web UI:** Create a simple web interface (e.g., using Flask or Streamlit) instead of the command-line interface.
*   **Advanced Retrieval:** Explore more sophisticated retrieval techniques (e.g., HyDE, re-ranking).
*   **Evaluation:** Implement metrics to evaluate the quality of the retrieval and generation.
*   **Error Handling:** Add more robust error handling, especially around file loading and LLM API calls.
