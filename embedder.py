from langchain_huggingface import HuggingFaceEmbeddings
from config import config

class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
