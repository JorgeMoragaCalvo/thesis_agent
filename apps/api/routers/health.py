from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

from apps.api.models.schemas import HealthResponse
from apps.api.core.database import get_db
from apps.api.config import settings

router = APIRouter(prefix="health", tags=["health"])

@router.get("/", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Health check endpoint to verify service status.

    Returns:
        HealthResponse with system status
    """
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    return HealthResponse(
        status="healthy" if db_connected else "unhealthy",
        app_name=settings.app_name,
        version=settings.app_version,
        database_connected=db_connected,
        timestamp=datetime.now()
    )now
