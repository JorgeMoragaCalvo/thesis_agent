from typing import List
from openai import OpenAI
import logging
import numpy as np

from apps.api.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_EMBEDDING_MODEL
        self.dimension = 1536 # text-embedding-3-small dimension

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        Args:
            text: The text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Replace newlines with spaces for better embedding quality
            text = text.replace("\n", " ").strip()

            if not text:
                logger.warning("Empty text provided for embedding.")
                return [0.0] * self.dimension

            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text of length: {len(text)}")

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embedding_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        try:
            all_embeddings = []

            for i in range(0, len(texts), batch_size):



embedding_service = EmbeddingService()
