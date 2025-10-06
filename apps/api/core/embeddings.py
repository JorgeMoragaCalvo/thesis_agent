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

            # This pattern is commonly used for batch processing -
            # dividing a large list into smaller chunks to process incrementally.
            # Process texts in batches of batch_size
            for i in range(0, len(texts), batch_size): # 0 (start), len(texts) (end), batch_size step (increment)
                batch = texts[i:i + batch_size]
                # Clean texts
                cleaned_batch = [text.replace("\n", " ").strip() for text in batch]

                response = self.client.embeddings.create(
                    input=cleaned_batch,
                    model=self.model
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.info(f"Generated embeddings for batch {i // batch_size + 1} "
                            f"({len(batch)} texts)")

            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            arr1 = np.array(vec1)
            arr2 = np.array(vec2)

            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            raise

embedding_service = EmbeddingService()
