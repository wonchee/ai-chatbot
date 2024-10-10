__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List
from langchain.schema.document import Document
from langchain_chroma import Chroma
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from utils import logger

CHROMA_PATH = "chroma"

class ChromaService(VectorStoreService):
    
    def __init__(self) -> None:
        """Initialize vector database"""
        super().__init__()
        
        embedding_service = EmbeddingService("openai")
        embedding_function = embedding_service.embedding_function
        
        # Initialize Chroma database.
        self.vector_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )
        

    def add_documents(self, chunks: List[Document]) -> None:
        """Add chunks of documents to vector database

        Args:
            chunks (List[Document]): Chunks of documents to be added to vector database
        """
        if not chunks:
            logger.warning("No documents to be added.")
            return
        
        # Form unique chunk ids.
        chunks_with_ids = self.form_unique_chunk_ids(chunks)
        
        # Get the source for each chunk
        sources = [chunk.metadata["source"] for chunk in chunks_with_ids]
        unique_sources = list(set(sources))
        
        # Delete any existing chunks with the same source
        ids_for_delete = []
        for source in unique_sources:
            existing_ids = self.vector_db.get(where = {'source': source})['ids']
            ids_for_delete.extend(existing_ids)
        self.delete_documents(ids_for_delete)
       
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        self.vector_db.add_documents(chunks_with_ids, ids=chunk_ids)
    
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete chunks of documents in vector database

        Args:
            ids (List[str]): List of ids for deletion in vector database
        """
        if ids is None or len(ids) == 0:
            logger.debug("No documents to be deleted.")
            return
        
        logger.info(f"Number of existing documents to be deleted in DB: {len(ids)}")
        self.vector_db.delete(ids=ids)
        
    def has_no_documents(self) -> bool:
        """Check if the vector database has any documents

        Returns:
            bool: A flag that indicates if the vector database has any documents
        """
        existing_items = self.vector_db.get(include=[])
        if len(existing_items["ids"]) == 0:
            return True
        
        return False
        