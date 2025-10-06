from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
import json
import time

from apps.api.models.db_models import Document, DocumentChunk, QueryLog
from apps.api.models.schemas import RetrievedChunk, QueryResponse
from apps.api.core.embeddings import embedding_service
from apps.api.core.document_processor import document_processor
from apps.api.config import settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) pipeline for document Q&A."""
    def __init__(self):
        self.embedding_service = embedding_service
        self.document_processor = document_processor
        self.top_k = settings.top_k_results
        self.similarity_threshold = settings.similarity_threshold

    def ingest_document(self, file_path: str, filename: str,
                        file_type: str, db: Session) -> Tuple[int, int]:
        """
        Ingest a document: process, chunk, embed, and store in database.

        Args:
            file_path: Path to the document file
            filename: Name of the file
            file_type: Type of file (pdf, txt, etc.)
            db: Database session

        Returns:
            Tuple of (document_id, number_of_chunks)
        """
        try:
            logger.info(f"Starting ingestion of document {filename}...")
            #Process document
            full_content, chunks = self.document_processor.process_file(file_path)
            # Create the document record
            document = Document(
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                content=full_content,
                metadata=json.dumps({"chunk_count": len(chunks)})
            )
            db.add(document)
            db.flush() # Get document ID without committing
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.generate_embedding_batch(chunk_texts)

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_text=chunk["text"],
                    chunk_index=idx,
                    embedding=embedding,
                    metadata=json.dumps(chunk["metadata"])
                )
                db.add(chunk_record)
            db.commit()
            logger.info(f"Successfully ingested document {filename}: "
                        f"ID={document.id}, chunks={len(chunks)}")
            return document.id, len(chunks)
        except Exception as e:
            db.rollback()
            logger.error(f"Error ingesting document {filename}: {e}")
            raise