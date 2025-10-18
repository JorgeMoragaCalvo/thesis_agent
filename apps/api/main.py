from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys

from apps.api.config import settings
from apps.api.routers import documents, query, health
from apps.api.core.database import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    logger.info(f"Environment: {settings.environment}")

    # Check database connection
    if db_manager.check_connection():
        logger.info("Database connection established")
    else:
        logger.error("Database connection failed")
    yield

    logger.info("Shutting down application...")

app = FastAPI(
    title=settings.app_name,
    description="AI-driven agent for personalized student support in Optimization Methods course",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Configure cors
# This is essential for a web API that might be called from a frontend (e.g., a React app), enabling secure cross-origin interactions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router) # Likely handles CRUD operations for documents
app.include_router(query.router) # Probably processes user queries
app.include_router(health.router)

@app.get("/") # Root endpoint. Purpose: Provides basic app metadata and links to useful endpoints.
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs", # Swagger UI
        "health": "/health" # Links to a health check endpoint from the health router.
    }

# Main execution block
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )