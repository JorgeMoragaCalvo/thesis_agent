from typing import List, Dict, Any
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from apps.api.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing and chunking documents."""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_document(self, file_path: str) -> List[LangChainDocument]:
        """
        Load a document from the file.

        Args:
            file_path: Path to the document file

        Returns:
            List of LangChain Document objects

        Raises:
            ValueError: If the file type is not supported
        """
        try:
            file_extension = Path(file_path).suffix.lower()

            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type '{file_extension}'")

            documents = loader.load()
            logger.info(f"Loaded document from {file_path}: {len(documents)} pages/sections")
            return documents
        except Exception as e:
            logger.error(f"Error loading document from {file_path}: {e}")
            raise

    def chunk_documents(self, documents: List[LangChainDocument]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)

            formatted_chunks = []
            for idx, chunk in enumerate(chunks):
                formatted_chunk = {
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_index": idx,
                        "chunk_size": len(chunk.page_content)
                    }
                }
                formatted_chunks.append(formatted_chunk)

            logger.info(f"Created {len(formatted_chunks)} chunks from {len(documents)} documents")
            return formatted_chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise

    def process_file(self, file_path: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Process a file: load and chunk it.

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of (full_content, chunks)
        """
        try:
            documents = self.load_document(file_path) # Load document
            full_content = "\n\n".join([doc.page_content for doc in documents]) # Extract full content
            chunks = self.chunk_documents(documents) # Create chunks
            logger.info(f"Processed file {file_path}: {len(full_content)} chars, {len(chunks)} chunks")
            return full_content, chunks
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        try:
            path = Path(file_path)
            stat = path.stat()

            metadata = {
                "filename": path.name,
                "file_size": stat.st_size,
                "file_extension": path.suffix.lower(),
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime,
            }
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            raise

document_processor = DocumentProcessor() # Global document processor instance