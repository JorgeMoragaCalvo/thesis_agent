"""
Database initialization script.
This script initializes the database by:
1. Creating the pgvector extension
2. Creating all necessary tables
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.api.core.database import db_manager
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    """Initialize database."""
    try:
        logger.info("Initializing database...")
        # Check database connection
        if not db_manager.check_connection():
            logger.error("Cannot connect to database. Please check your configuration.")
            return False

        logger.info("Database connection successful")
        # Create tables
        logger.info("Creating database tables...")
        db_manager.create_tables()

        logger.info("✅ Database initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"An error occurred while initializing database: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)