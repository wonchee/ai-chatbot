from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingService():
    
    def __init__(self, model: str) -> None:
        self.embedding_function = None
        
        if model == "openai":
            self.embedding_function = OpenAIEmbeddings()
        elif model == "ollama":
            self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        else:
            raise ValueError("Invalid argument: Only 'openai' and 'ollama' are supported for now.")
        