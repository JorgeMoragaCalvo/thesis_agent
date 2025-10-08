from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from apps.api.models.schemas import QueryRequest, QueryResponse
from apps.api.core.database import get_db
from apps.api.core.rag_pipeline import rag_pipeline

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)

@router.post("/", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    """
    Query the knowledge base using RAG.

    Args:
        request: Query request with query text and optional parameters
        db: Database session

    Returns:
        QueryResponse with answer and retrieved chunks
    """
    try:
        logger.info(f"Processing query: {request.query[:50]}...")

        response = rag_pipeline.query(
            query=request.query,
            db=db,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing query: {str(e)}"
                            )