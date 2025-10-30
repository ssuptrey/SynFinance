"""
Database Manager for SynFinance

Manages database connections, sessions, and provides utilities
for database operations.

Week 7 Day 5: Database Integration
"""

import os
from typing import Optional, Dict, Any, List, Generator
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import Base
from src.observability import get_logger, LogCategory

logger = get_logger(__name__)


class DatabaseConfig:
    """Database configuration"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "synfinance",
        username: str = "synfinance_user",
        password: str = "synfinance_password",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables"""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "synfinance"),
            username=os.getenv("DB_USER", "synfinance_user"),
            password=os.getenv("DB_PASSWORD", "synfinance_password"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )
    
    def get_connection_string(self, driver: str = "postgresql+psycopg2") -> str:
        """Get SQLAlchemy connection string"""
        password_encoded = quote_plus(self.password)
        return f"{driver}://{self.username}:{password_encoded}@{self.host}:{self.port}/{self.database}"


class DatabaseManager:
    """
    Database Manager
    
    Manages database connections, sessions, and provides utilities
    for database operations with connection pooling and monitoring.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager
        
        Args:
            config: Database configuration (uses environment if None)
        """
        self.config = config or DatabaseConfig.from_env()
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.scoped_session_factory: Optional[scoped_session] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize database connection and session factory"""
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        try:
            logger.info(
                "Initializing database connection",
                category=LogCategory.DATABASE,
                extra={
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database,
                    "pool_size": self.config.pool_size
                }
            )
            
            # Create engine with connection pooling
            self.engine = create_engine(
                self.config.get_connection_string(),
                poolclass=pool.QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Verify connections before using
                echo=self.config.echo
            )
            
            # Add event listeners for monitoring
            self._add_event_listeners()
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            # Create scoped session for thread safety
            self.scoped_session_factory = scoped_session(self.session_factory)
            
            self._initialized = True
            
            logger.info(
                "Database initialized successfully",
                category=LogCategory.DATABASE
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize database",
                category=LogCategory.DATABASE,
                extra={"error": str(e)}
            )
            raise
    
    def _add_event_listeners(self) -> None:
        """Add SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new connection"""
            logger.debug(
                "New database connection established",
                category=LogCategory.DATABASE
            )
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log connection checkout from pool"""
            logger.debug(
                "Connection checked out from pool",
                category=LogCategory.DATABASE
            )
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Log connection checkin to pool"""
            logger.debug(
                "Connection returned to pool",
                category=LogCategory.DATABASE
            )
    
    def create_all_tables(self) -> None:
        """Create all database tables"""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.info(
                "Creating database tables",
                category=LogCategory.DATABASE
            )
            
            Base.metadata.create_all(self.engine)
            
            logger.info(
                "Database tables created successfully",
                category=LogCategory.DATABASE
            )
            
        except Exception as e:
            logger.error(
                "Failed to create database tables",
                category=LogCategory.DATABASE,
                extra={"error": str(e)}
            )
            raise
    
    def drop_all_tables(self) -> None:
        """Drop all database tables (use with caution)"""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.warning(
                "Dropping all database tables",
                category=LogCategory.DATABASE
            )
            
            Base.metadata.drop_all(self.engine)
            
            logger.info(
                "Database tables dropped successfully",
                category=LogCategory.DATABASE
            )
            
        except Exception as e:
            logger.error(
                "Failed to drop database tables",
                category=LogCategory.DATABASE,
                extra={"error": str(e)}
            )
            raise
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations
        
        Yields:
            Database session
        
        Example:
            >>> with db_manager.session_scope() as session:
            ...     transaction = Transaction(...)
            ...     session.add(transaction)
        """
        if not self._initialized:
            self.initialize()
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(
                "Database transaction failed, rolling back",
                category=LogCategory.DATABASE,
                extra={"error": str(e)}
            )
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """
        Get a new database session
        
        Returns:
            Database session
        
        Note:
            Caller is responsible for closing the session
        """
        if not self._initialized:
            self.initialize()
        
        return self.session_factory()
    
    def get_scoped_session(self) -> scoped_session:
        """
        Get a thread-local scoped session
        
        Returns:
            Scoped session
        """
        if not self._initialized:
            self.initialize()
        
        return self.scoped_session_factory
    
    def bulk_insert(self, objects: List[Any], batch_size: int = 1000) -> int:
        """
        Bulk insert objects with batching
        
        Args:
            objects: List of ORM objects to insert
            batch_size: Number of objects per batch
        
        Returns:
            Number of objects inserted
        """
        if not objects:
            return 0
        
        total_inserted = 0
        
        try:
            with self.session_scope() as session:
                for i in range(0, len(objects), batch_size):
                    batch = objects[i:i + batch_size]
                    session.bulk_save_objects(batch)
                    total_inserted += len(batch)
                    
                    logger.debug(
                        f"Inserted batch of {len(batch)} objects",
                        category=LogCategory.DATABASE,
                        extra={"total": total_inserted}
                    )
            
            logger.info(
                f"Bulk insert complete",
                category=LogCategory.DATABASE,
                extra={"total_inserted": total_inserted}
            )
            
            return total_inserted
            
        except Exception as e:
            logger.error(
                "Bulk insert failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e), "objects_attempted": len(objects)}
            )
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            List of result rows as dictionaries
        """
        try:
            with self.session_scope() as session:
                result = session.execute(text(query), params or {})
                
                # Convert to list of dictionaries
                rows = []
                if result.returns_rows:
                    columns = result.keys()
                    for row in result:
                        rows.append(dict(zip(columns, row)))
                
                return rows
                
        except Exception as e:
            logger.error(
                "Query execution failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e), "query": query[:100]}
            )
            raise
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status
        
        Returns:
            Dictionary with pool statistics
        """
        if not self._initialized or not self.engine:
            return {"status": "not_initialized"}
        
        pool = self.engine.pool
        
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "max_overflow": self.config.max_overflow,
            "timeout": self.config.pool_timeout
        }
    
    def health_check(self) -> bool:
        """
        Check database connection health
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(
                "Database health check failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e)}
            )
            return False
    
    def close(self) -> None:
        """Close database connections and dispose engine"""
        if not self._initialized:
            return
        
        try:
            logger.info(
                "Closing database connections",
                category=LogCategory.DATABASE
            )
            
            if self.scoped_session_factory:
                self.scoped_session_factory.remove()
            
            if self.engine:
                self.engine.dispose()
            
            self._initialized = False
            
            logger.info(
                "Database connections closed",
                category=LogCategory.DATABASE
            )
            
        except Exception as e:
            logger.error(
                "Error closing database connections",
                category=LogCategory.DATABASE,
                extra={"error": str(e)}
            )


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """
    Get global database manager instance
    
    Args:
        config: Database configuration (uses environment if None)
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(config)
        _db_manager.initialize()
    
    return _db_manager


def get_db_session() -> Session:
    """
    Get database session from global manager
    
    Returns:
        Database session
    """
    return get_db_manager().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Get database session with automatic transaction management
    
    Yields:
        Database session
    """
    with get_db_manager().session_scope() as session:
        yield session
