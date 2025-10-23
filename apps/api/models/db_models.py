from sqlalchemy import Column, Integer, String, DateTime, Float, Text, func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# This creates a base class (Base) that all subsequent model classes will inherit from.
Base = declarative_base()

class Document(Base):
    """Database model for storing documents and their metadata."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False) # pdf, txt, etc.
    content = Column(Text, nullable=False)
    extra_metadata = Column(Text, nullable=True) # JSON string for additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}')>"

class DocumentChunk(Base):

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False) # Order of the chunk in the document
    embedding = Column(Vector(1536), nullable=False) # OpenAI embedding dimension
    extra_metadata = Column(Text, nullable=True) # JSON string for chunk-specific metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"

class QueryLog(Base):
    """Database model for logging user queries and responses."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    retrieved_chunks = Column(Text, nullable=True) # JSON string of chunk IDs
    similarity_score = Column(Text, nullable=True) # JSON string of scores
    response_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<QueryLog(id={self.id}, query='{self.query[:50]}...')>"

class DocumentMetadata(BaseModel):
    """Metadata for a document stored in Chroma."""
    filename: str
    file_path: str
    file_type: str
    chunk_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None

class ChunkMetadata(BaseModel):
    """Metadata for a document chunk stored in Chroma."""
    document_id: str
    filename: str
    chunk_index: int
    chunk_size: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class QueryLogEntry(BaseModel):
    """Entry for logging queries (stored as JSON)."""
    id: str
    query: str
    response: str
    retrieved_chunks: List[int] = Field(default_factory=list)
    similarity_score: List[float] = Field(default_factory=list)
    response_time: float
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

# Chroma collection names
COLLECTION_DOCUMENTS = "documents"
COLLECTION_CHUNKS = "document_chunks"
COLLECTION_QUERY_LOG = "query_logs"

def create_document_metadata(filename: str, file_path: str, file_type: str,
                             chunk_count: int) -> Dict[str, Any]:
    """
    Create the metadata dictionary for a document.

    Args:
        filename: Name of the file
        file_path: Path to the file
        file_type: Type of file (pdf, txt, etc.)
        chunk_count: Number of chunks created from document

    Returns:
        Dictionary containing document metadata
    """
    metadata = DocumentMetadata(
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        chunk_count=chunk_count
    )
    return metadata.model_dump()

def create_chunk_metadata(document_id: str, filename: str,
                          chunk_index: int, chunk_size: int) -> Dict[str, Any]:
    """
    Create the metadata dictionary for a document chunk.

    Args:
        document_id: ID of the parent document
        filename: Name of the source file
        chunk_index: Index of chunk in document
        chunk_size: Size of chunk in characters

    Returns:
        Dictionary containing chunk metadata
    """
    metadata = ChunkMetadata(
        document_id=document_id,
        filename=filename,
        chunk_index=chunk_index,
        chunk_size=chunk_size
    )
    return metadata.model_dump()