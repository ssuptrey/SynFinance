"""
Initialize Database Schema

Creates all database tables for SynFinance.
Week 8 Day 1: Database setup for GraphQL integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_manager import get_db_manager
from src.observability import get_logger

logger = get_logger(__name__)


def main():
    """Initialize database schema"""
    try:
        logger.info("Starting database initialization")
        
        # Get database manager
        db_manager = get_db_manager()
        
        # Create all tables
        logger.info("Creating database tables...")
        db_manager.create_all_tables()
        
        logger.info("Database initialization complete!")
        logger.info("Tables created successfully in the 'synfinance' database")
        
        # Health check
        if db_manager.health_check():
            logger.info("Database health check: PASSED")
        else:
            logger.error("Database health check: FAILED")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
