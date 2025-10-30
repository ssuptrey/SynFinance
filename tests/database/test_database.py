"""
Tests for Database Layer

Tests for models, db_manager, and repositories.

Week 7 Day 5: Database Integration
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.database import (
    Transaction, Customer, Merchant, MLFeatures, ModelPrediction,
    DatabaseConfig, DatabaseManager,
    TransactionRepository, CustomerRepository, MerchantRepository
)


class TestDatabaseConfig:
    """Test database configuration"""
    
    def test_config_creation(self):
        """Test creating database config"""
        config = DatabaseConfig(
            host="testhost",
            port=5433,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        
        assert config.host == "testhost"
        assert config.port == 5433
        assert config.database == "testdb"
    
    def test_connection_string(self):
        """Test connection string generation"""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="synfinance",
            username="user",
            password="pass"
        )
        
        conn_str = config.get_connection_string()
        assert "postgresql+psycopg2://user:" in conn_str
        assert "@localhost:5432/synfinance" in conn_str
    
    def test_from_env(self):
        """Test creating config from environment"""
        with patch.dict('os.environ', {
            'DB_HOST': 'envhost',
            'DB_PORT': '5433',
            'DB_NAME': 'envdb'
        }):
            config = DatabaseConfig.from_env()
            assert config.host == 'envhost'
            assert config.port == 5433
            assert config.database == 'envdb'


class TestModels:
    """Test database models"""
    
    def test_transaction_creation(self):
        """Test creating a transaction"""
        tx = Transaction(
            transaction_id="TX123",
            customer_id="CUST001",
            merchant_id="MERCH001",
            amount=Decimal("100.50"),
            timestamp=datetime.now(),
            transaction_type="purchase",
            payment_method="credit_card",
            channel="online",
            hour_of_day=14,
            day_of_week=2,
            day_of_month=15,
            month=10,
            is_fraud=False
        )
        
        assert tx.transaction_id == "TX123"
        assert tx.amount == Decimal("100.50")
        assert tx.is_fraud == False
    
    def test_transaction_to_dict(self):
        """Test converting transaction to dict"""
        tx = Transaction(
            transaction_id="TX123",
            customer_id="CUST001",
            merchant_id="MERCH001",
            amount=Decimal("100.50"),
            timestamp=datetime.now(),
            transaction_type="purchase",
            payment_method="credit_card",
            channel="online",
            hour_of_day=14,
            day_of_week=2,
            day_of_month=15,
            month=10,
            is_fraud=False
        )
        
        tx_dict = tx.to_dict()
        assert tx_dict['transaction_id'] == "TX123"
        assert tx_dict['amount'] == 100.50
        assert 'customer_id' in tx_dict
    
    def test_customer_creation(self):
        """Test creating a customer"""
        customer = Customer(
            customer_id="CUST001",
            first_name="John",
            last_name="Doe",
            email="john.doe@example.com",
            account_created=datetime.now(),
            age=35,
            city="New York",
            state="NY"
        )
        
        assert customer.customer_id == "CUST001"
        assert customer.first_name == "John"
        assert customer.email == "john.doe@example.com"
    
    def test_merchant_creation(self):
        """Test creating a merchant"""
        merchant = Merchant(
            merchant_id="MERCH001",
            name="Test Store",
            category="retail",
            city="Los Angeles",
            state="CA"
        )
        
        assert merchant.merchant_id == "MERCH001"
        assert merchant.name == "Test Store"
        assert merchant.category == "retail"
    
    def test_ml_features_creation(self):
        """Test creating ML features"""
        features = MLFeatures(
            transaction_id="TX123",
            amount_normalized=0.5,
            tx_velocity_1h=3,
            unusual_time=False,
            customer_transaction_count=50
        )
        
        assert features.transaction_id == "TX123"
        assert features.amount_normalized == 0.5
    
    def test_model_prediction_creation(self):
        """Test creating model prediction"""
        prediction = ModelPrediction(
            transaction_id="TX123",
            model_name="RandomForest",
            model_version="1.0.0",
            prediction=False,
            confidence_score=0.85,
            fraud_probability=0.15
        )
        
        assert prediction.transaction_id == "TX123"
        assert prediction.model_name == "RandomForest"
        assert prediction.prediction == False


class TestDatabaseManager:
    """Test database manager (mocked)"""
    
    def test_manager_creation(self):
        """Test creating database manager"""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="testdb"
        )
        
        manager = DatabaseManager(config)
        assert manager.config == config
        assert manager._initialized == False
    
    def test_pool_status_not_initialized(self):
        """Test pool status when not initialized"""
        manager = DatabaseManager()
        status = manager.get_pool_status()
        
        assert status['status'] == 'not_initialized'


class TestRepositories:
    """Test repository pattern (mocked sessions)"""
    
    def test_transaction_repository_create(self):
        """Test creating transaction via repository"""
        mock_session = MagicMock()
        repo = TransactionRepository(mock_session)
        
        # Mock flush to prevent errors
        mock_session.flush = MagicMock()
        
        tx = repo.create(
            transaction_id="TX123",
            customer_id="CUST001",
            merchant_id="MERCH001",
            amount=Decimal("100.00"),
            timestamp=datetime.now(),
            transaction_type="purchase",
            payment_method="credit_card",
            channel="online",
            hour_of_day=14,
            day_of_week=2,
            day_of_month=15,
            month=10
        )
        
        # Verify session.add was called
        mock_session.add.assert_called_once()
    
    def test_customer_repository_create(self):
        """Test creating customer via repository"""
        mock_session = MagicMock()
        repo = CustomerRepository(mock_session)
        
        mock_session.flush = MagicMock()
        
        customer = repo.create(
            customer_id="CUST001",
            first_name="John",
            last_name="Doe",
            email="john@example.com",
            account_created=datetime.now()
        )
        
        mock_session.add.assert_called_once()
    
    def test_merchant_repository_create(self):
        """Test creating merchant via repository"""
        mock_session = MagicMock()
        repo = MerchantRepository(mock_session)
        
        mock_session.flush = MagicMock()
        
        merchant = repo.create(
            merchant_id="MERCH001",
            name="Test Store",
            category="retail"
        )
        
        mock_session.add.assert_called_once()


class TestIntegration:
    """Integration tests (require database)"""
    
    @pytest.mark.skip(reason="Requires PostgreSQL database")
    def test_full_workflow(self):
        """Test full database workflow"""
        # This would test:
        # 1. Initialize database
        # 2. Create tables
        # 3. Insert data
        # 4. Query data
        # 5. Update data
        # 6. Delete data
        pass
