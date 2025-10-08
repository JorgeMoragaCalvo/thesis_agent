"""
Dependency injection for FastAPI endpoints.

This module contains common dependencies used across routers.
"""

from apps.api.core.database import get_db

__all__ = ["get_db"]