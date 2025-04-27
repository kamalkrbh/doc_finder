import os

class Config:
    def __init__(self):
        
        self.LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # ollama or groq
        self.GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")  # llama3.1:latest , llama-3.3-70b-versatile
        self.GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        self.PDF_DIRECTORY = "pdfs"  # Directory where PDFs are stored
        self.INDEX_DIRECTORY = "index" # Directory where index is stored
        self.EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # Embedding model

config = Config()
