from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from config import config

class LLMHandler:
    def __init__(self):
        self.llm = self._get_llm()

    def _get_llm(self):
        """Configures and returns the appropriate LLM."""
        if config.LLM_PROVIDER == "ollama":
            return OllamaLLM(model=config.OLLAMA_MODEL)
        elif config.LLM_PROVIDER == "groq":
            if not config.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY environment variable is not set.")
            return ChatGroq(model=config.GROQ_MODEL)
        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {config.LLM_PROVIDER}")

    def generate_response(self, prompt: str):
        """Generates a response from the LLM."""
        return str(self.llm.invoke(prompt))
