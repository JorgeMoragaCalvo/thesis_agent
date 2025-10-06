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