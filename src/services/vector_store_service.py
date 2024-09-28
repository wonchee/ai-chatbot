from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


class VectorStoreService():
     
    def load_documents(self, file_paths: List[str] = None) -> List[Document]:
        """Load the list of documents given their file paths

        Args:
            file_paths (List[str], optional): List of file paths to load the documents from. Defaults to None.

        Raises:
            ValueError: Missing file paths

        Returns:
            List[Document]: Chunks of documents
        """
        if file_paths is None:
            raise ValueError("Invalid argument: Must provide file paths.")
            
        all_documents = []
        document_loaders = [PDFPlumberLoader(file_path=path) for path in file_paths]
        
        for loader in document_loaders:
            documents = loader.load()
            chunks = self.chunk_documents(documents)
            all_documents.extend(chunks)
            
        return all_documents
    
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents

        Args:
            documents (List[Document]): List of documents to be split into chunks

        Returns:
            List[Document]: Chunks of documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)
    
    
    def form_unique_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """Form unique id for each chunk of documents

        Args:
            documents (List[Document]): Chunks of documents

        Returns:
            List[Document]: Chunks of documents with unique id added
        """
        last_page = None
        current_chunk_index = 0

        for chunk in chunks:
            chunk_metadata = chunk.metadata
            source = chunk_metadata.get("source")
            page = chunk_metadata.get("page")
            current_page = f"{source}:page_{page}"

            if current_page == last_page:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Form the chunk ID.
            chunk_id = f"{current_page}:chunk_{current_chunk_index}"
            last_page = current_page

            # Add the unique chunk id to metadata
            # Sample id: data/demo.pdf:page_0:chunk_0
            chunk.metadata["id"] = chunk_id

        return chunks
    
    
    def add_documents(self, chunks: List[Document]):
        raise NotImplementedError
    
    
    def delete_documents(self, ids: List[str]):
        raise NotImplementedError
    
    
    def has_no_documents(self):
        raise NotImplementedError
   