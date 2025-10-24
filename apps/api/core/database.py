from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator, Optional
import logging

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path

from apps.api.config import settings
from apps.api.models.db_models import Base
from apps.api.models.db_models import (COLLECTION_DOCUMENTS, COLLECTION_CHUNKS)

logger = logging.getLogger(__name__)

# This is the core class, responsible for all database-related operations.
# It follows a singleton-like pattern via the global db_manager instance
class DatabaseManager:
    """Manager class for database operations."""
    def __init__(self):
        self.engine = create_engine(
            settings.sync_database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self):
        """Create all database tables"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension enabled")

            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Tables created successfully")

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")

    # creates a session, yields it, and handles cleanup.
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Usage:
            with db_manager.get_session() as session:
            # Use session here
            pass
        """
        session = self.SessionLocal() # creates a new session
        try:
            yield session
            session.commit() # saves changes if no errors
        except Exception as e:
            session.rollback() # Reverts changes on exceptions.
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def check_connection(self) -> bool:
        """Check if the database connection is working."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1")) # Attempts a simple query
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

# Creates a single, global instance of DatabaseManager for app-wide use (singleton)
db_manager = DatabaseManager()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database sessions.

    Usage in FastAPI:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db here
            pass
    """
    session = db_manager.SessionLocal()
    try:
        yield session # Gives the session to FastAPI routes
    finally:
        session.close()

class ChromaDBManager:
    """Manager class for ChromaDB operations."""
    def __init__(self):
        self._client: Optional[chromadb.Client] = None
        self._documents_collection = None
        self._chunks_collection = None
        self.persist_directory = settings.chroma_persist_directory

    @property
    def client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            # Ensure persist directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized with persist directory: {self.persist_directory}")

        return self._client

    def get_or_create_collection(self, collection_name: str, metadata: dict = None):
        """
        Get or create a ChromaDB collection.

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection

        Returns:
            ChromaDB collection object
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            logger.debug(f"Collection '{collection_name}' ready")
            return collection
        except Exception as e:
            logger.error(f"Error getting/creating collection '{collection_name}': {e}")
            raise

    @property
    def documents_collection(self):
        """Get documents collection."""
        if self._documents_collection is None:
            self._documents_collection = self.get_or_create_collection(
                COLLECTION_DOCUMENTS,
                metadata={"description": "Document storage with full content"}
            )
        return self._documents_collection

    @property
    def chunks_collection(self):
        """Get chunks collection."""
        if self._chunks_collection is None:
            self._chunks_collection = self.get_or_create_collection(
                COLLECTION_CHUNKS,
                metadata={"description": "Document chunks with embeddings for retrieval"}
            )
        return self._chunks_collection

    def create_collections(self):
        """Initialize all required collections."""
        try:
            # Access properties to trigger creation
            _ = self.documents_collection
            _ = self.chunks_collection
            logger.info("All ChromaDB collections created successfully")
        except Exception as e:
            logger.error(f"Error creating ChromaDB collections: {e}")
            raise

    def delete_collection(self, collection_name: str):
        """
        Delete a ChromaDB collection.

        Args:
            collection_name: Name of the collection to delete
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.warning(f"Collection '{collection_name}' deleted")

            # Reset cached collection references
            if collection_name == COLLECTION_DOCUMENTS:
                self._documents_collection = None
            elif collection_name == COLLECTION_CHUNKS:
                self._chunks_collection = None

        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            raise

    def reset_database(self):
        """Reset all collections (use with caution!)."""
        try:
            self.client.reset()
            self._documents_collection = None
            self._chunks_collection = None
            logger.warning("ChromaDB reset - all collections deleted")
        except Exception as e:
            logger.error(f"Error resetting ChromaDB: {e}")
            raise

    def check_connection(self) -> bool:
        """Check if ChromaDB is working."""
        try:
            _ = self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB connection check failed: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> dict:
        """
        Get statistics for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()

            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting stats for collection '{collection_name}': {e}")
            return {
                "name": collection_name,
                "count": 0,
                "error": str(e)
            }

chroma_manager = ChromaDBManager()

def get_chroma_client() -> chromadb.Client:
    """Get ChromaDB client instance."""
    return chroma_manager.client

def get_chunks_collection():
    """Get chunks collection for dependency injection."""
    return chroma_manager.chunks_collection