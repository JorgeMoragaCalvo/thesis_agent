from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(
    title=settings.app_name,
    description="AI-driven agent for personalized student support in Optimization Methods course",
    version=settings.app_version,
    debug=settings.debug
)

# Configure cors
app.middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    logger.info(f"Environment: {settings.environment}")

    # Check database connection
    if db_manager.check_connection():
        logger.info("Database connection established")
    else:
        logger.error("Database connection failed")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down application...")

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(health.router)

@app.get("/") # Root endpoint
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )