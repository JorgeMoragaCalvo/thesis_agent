from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import OpenAI
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

    def retrieval_relevant_chunks(self, query:str, db: Session, top_k: int = None,
                                  similarity_threshold: float = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant document chunks for a query using vector similarity.

        Args:
            query: User query
            db: Database session
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
        List of retrieved chunks with metadata
        """
        try:
            # Use default values if not provided
            top_k = top_k or self.top_k
            similarity_threshold = similarity_threshold or self.similarity_threshold

            logger.info(f"Retrieving relevant chunks for query: '{query[:50]}...'")
            #Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)

            # Use pgvector for similarity search
            # Note: Using cosine distance (1 - cosine similarity)
            sql_query = text("""
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.chunk_text,
                    dc.chunk_index,
                    d.filename,
                    1 - (dc.embedding <=> :query_embedding) AS similarity
                FROM document_chunk dc
                JOIN documents d ON dc.document_id = d.id
                WHERE 1 - (dc.embedding <=> :query_embedding) >= :threshold
                ORDER BY dc.embedding <=> :query_embedding
                LIMIT :top_k
            """)

            result = db.execute(
                sql_query,
                {
                    "query_embedding": str(query_embedding),
                    "threshold": similarity_threshold,
                    "top_k": top_k
                }
            )

            retrieved_chunks = []
            for row in result:
                chunk = RetrievedChunk(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    filename=row.filename,
                    chunk_text=row.chunk_text,
                    similarity_score=float(row.similarity),
                    chunk_index=row.chunk_index
                )
                retrieved_chunks.append(chunk)
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")

            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks for query: {e}")
            raise

    def generate_answer(self, query: str, retrieved_chunks: List[RetrievedChunk]):
        """
        Generate an answer based on retrieved chunks.

        For now, this is a simple context-based response.
        This will be enhanced with LLM integration later.

        Args:
            query: User query
            retrieved_chunks: Retrieved relevant chunks

        Returns:
            Generated answer
        """
        try:
            if not retrieved_chunks:
                return ("I couldn't find any relevant information in the knowledge base "
                    "to answer your question. Please try rephrasing or ask about a different topic.")

            # Build context from retrieved chunks
            context = "\n\n".join([
                f"Source: {chunk.filename} (Chunk {chunk.chunk_index})\n{chunk.chunk_text}"
                for chunk in retrieved_chunks
            ])

            # Create prompt for LLM
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Use only the information from the context to answer the question. "
                "If the context doesn't contain enough information, say so. "
                "Always cite which source document your answer comes from."
            )

            user_prompt = f"""Context: {context}
            Question: {query}
            Please provide a clear and detailed answer based on the context.
            """

            client = OpenAI(api_key=settings.openai_api_key)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise