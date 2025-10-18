from sqlalchemy import Column, Integer, String, DateTime, Float, Text, func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

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