from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""
    document_id: int
    filename: str
    chunk_created: int
    message: str

class DocumentInfo(BaseModel):
    """Schema for document information."""
    id: int
    filename: str
    file_type: str
    created_at: datetime
    chunk_count: Optional[int] = None

    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    """Request schema for RAG queries."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=5, ge=1, le=10)
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)

class RetrievedChunk(BaseModel):
    """Schema for a retrieved document chunk."""
    chunk_id: int
    document_id: int
    filename: str
    chunk_text: str
    similarity_score: float
    chunk_index: int

class QueryResponse(BaseModel):
    """Response schema for RAG queries."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    response_time: float

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    app_name: str
    version: str
    database_connected: bool
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime