"""
SynFinance Database Package

Database integration with PostgreSQL, SQLAlchemy ORM,
and repository pattern for data access.

Week 7 Day 5: Database Integration
"""

from src.database.models import (
    Base,
    Transaction,
    Customer,
    Merchant,
    MLFeatures,
    ModelPrediction,
    FraudPattern,
    AnomalyPattern
)

from src.database.db_manager import (
    DatabaseConfig,
    DatabaseManager,
    get_db_manager,
    get_db_session,
    session_scope
)

from src.database.repositories import (
    BaseRepository,
    TransactionRepository,
    CustomerRepository,
    MerchantRepository,
    MLFeaturesRepository,
    ModelPredictionRepository
)

__all__ = [
    # Models
    'Base',
    'Transaction',
    'Customer',
    'Merchant',
    'MLFeatures',
    'ModelPrediction',
    'FraudPattern',
    'AnomalyPattern',
    
    # Database Manager
    'DatabaseConfig',
    'DatabaseManager',
    'get_db_manager',
    'get_db_session',
    'session_scope',
    
    # Repositories
    'BaseRepository',
    'TransactionRepository',
    'CustomerRepository',
    'MerchantRepository',
    'MLFeaturesRepository',
    'ModelPredictionRepository',
]

__version__ = "1.0.0"
