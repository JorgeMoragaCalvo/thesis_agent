from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
import shutil
from pathlib import Path
import logging

from apps.api.core.rag_pipeline import rag_pipeline
from apps.api.models.schemas import DocumentUploadResponse, DocumentInfo
from apps.api.models.db_models import Document, DocumentChunk
from apps.api.core.database import get_db

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"txt", "pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10 MB

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
) -> DocumentUploadResponse:
    """
    Upload and process a document.

    Args:
        file: The document file to upload
        db: Database session

    Returns:
        DocumentUploadResponse with upload details
    """
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File extension '{file_extension}' not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Check file size
        file.file.seek(0, 2) # Seek to end
        file_size = file.file.tell()
        file.file.seek(0) # Reset to beginning

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size {file_size} exceeds maximum. Max size: {MAX_FILE_SIZE / 1024 / 1024} MB"
            )

        # Create the data directory if it doesn't exist
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save the file
        file_path = data_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved to {file_path}")

        document_id, chunks_created = rag_pipeline.ingest_document(
            file_path=str(file_path),
            filename=file.filename,
            file_type=file_extension[1:], # remove the dot
            db=db
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunk_created=chunks_created,
            message=f"Document uploaded and processed successfully."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/", response_model=List[DocumentInfo])
def list_documents(db: Session = Depends(get_db)) -> List[DocumentInfo]:
    """
    List all documents in the knowledge base.

    Args:
        db: Database session

    Returns:
        List of DocumentInfo objects
    """
    try:
        # Query documents with chunk counts
        documents = db.query(
            Document,
            func.count(DocumentChunk.id).label("chunk_count")
        ).outerjoin(
            DocumentChunk, Document.id == DocumentChunk.document_id
        ).group_by(Document.id).all()
        # Format response
        document_list = []
        for doc, chunk_count in documents:
            doc_info = DocumentInfo(
                id=doc.id,
                filename=doc.filename,
                file_type=doc.file_type,
                created_at=doc.created_at,
                chunk_count=chunk_count
            )
            document_list.append(doc_info)
        return document_list
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentInfo)
def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentInfo:
    """
    Get details of a specific document.

    Args:
        document_id: ID of the document
        db: Database session

    Returns:
        DocumentInfo object
    """
    try:
        result = db.query(
            Document,
            func.count(DocumentChunk.id).label("chunk_count")
        ).outerjoin(
            DocumentChunk, Document.id == DocumentChunk.document_id
        ).filter(Document.id == document_id).group_by(Document.id).first()

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found."
            )

        doc, chunk_count = result

        return DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            created_at=doc.created_at,
            chunk_count=chunk_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving document: {str(e)}"
        )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and its chunks.

    Args:
        document_id: ID of the document to delete
        db: Database session
    """
    try:
        # Find document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_400_NOT_FOUND,
                detail=f"Document with ID {document_id} not found."
            )
        # Delete associated chunks
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

        # Delete file from filesystem
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
        # Delete document record
        db.delete(document)
        db.commit()

        logger.info(f"Document {document_id} deleted successfully.")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
e