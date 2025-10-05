from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from multipart import file_path
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
import shutil
import os
from pathlib import Path
import logging

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

        document_id, chunks_created =
