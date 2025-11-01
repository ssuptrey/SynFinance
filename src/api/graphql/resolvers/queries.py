"""
GraphQL Query Resolvers

Implements all GraphQL queries for data retrieval.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import strawberry

from ..types import (
    TransactionType,
    CustomerType,
    MerchantType,
    MLFeaturesType,
    FraudPatternType,
    SystemHealthType,
    GenerationStatsType,
    TransactionFilterInput,
    CustomerFilterInput,
    MerchantFilterInput,
)
from src.database.db_manager import session_scope
from src.database.repositories import (
    TransactionRepository,
    CustomerRepository,
    MerchantRepository,
    MLFeaturesRepository,
)
from src.observability import get_logger

logger = get_logger(__name__)


@strawberry.type
class Query:
    """Root Query type with all available queries"""
    
    @strawberry.field
    async def transactions(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[TransactionFilterInput] = None
    ) -> List[TransactionType]:
        """
        Query transactions with optional filters.
        
        Args:
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip
            filters: Optional filters for transactions
            
        Returns:
            List of transactions matching the criteria
        """
        with session_scope() as session:
            repo = TransactionRepository(session)
            
            # Apply filters if provided
            if filters:
                if filters.is_fraud is not None:
                    transactions = repo.get_fraud_transactions(
                        start_date=filters.start_date,
                        end_date=filters.end_date,
                        limit=limit,
                        offset=offset
                    ) if filters.is_fraud else repo.get_by_date_range(
                        start_date=filters.start_date or datetime(2020, 1, 1),
                        end_date=filters.end_date or datetime.now(),
                        limit=limit,
                        offset=offset
                    )
                elif filters.start_date and filters.end_date:
                    transactions = repo.get_by_date_range(
                        start_date=filters.start_date,
                        end_date=filters.end_date,
                        limit=limit,
                        offset=offset
                    )
                elif filters.customer_id:
                    transactions = repo.get_by_customer(
                        customer_id=filters.customer_id,
                        limit=limit,
                        offset=offset
                    )
                elif filters.merchant_id:
                    transactions = repo.get_by_merchant(
                        merchant_id=filters.merchant_id,
                        limit=limit,
                        offset=offset
                    )
                elif filters.min_amount and filters.max_amount:
                    transactions = repo.get_by_amount_range(
                        min_amount=Decimal(str(filters.min_amount)),
                        max_amount=Decimal(str(filters.max_amount)),
                        limit=limit
                    )
                else:
                    transactions = repo.get_all(limit=limit, offset=offset)
            else:
                transactions = repo.get_all(limit=limit, offset=offset)
            
            # Convert to GraphQL types
            return [
                TransactionType(
                    transaction_id=tx.transaction_id,
                    customer_id=tx.customer_id,
                    merchant_id=tx.merchant_id,
                    amount=float(tx.amount),
                    timestamp=tx.timestamp,
                    is_fraud=tx.is_fraud,
                    fraud_type=tx.fraud_type,
                    category=tx.category or "",
                    payment_mode=tx.payment_mode or "",
                    city=tx.city or "",
                    state=tx.state,
                    latitude=float(tx.latitude) if tx.latitude else None,
                    longitude=float(tx.longitude) if tx.longitude else None,
                    fraud_confidence=float(tx.fraud_confidence) if tx.fraud_confidence else None,
                    is_anomaly=tx.is_anomaly,
                    anomaly_score=float(tx.anomaly_score) if tx.anomaly_score else None,
                    anomaly_type=tx.anomaly_type,
                    risk_score=float(tx.risk_score) if tx.risk_score else None,
                    velocity_score=float(tx.velocity_score) if tx.velocity_score else None
                )
                for tx in transactions
            ]
    
    @strawberry.field
    async def transaction(
        self,
        transaction_id: str
    ) -> Optional[TransactionType]:
        """
        Get a single transaction by ID.
        
        Args:
            transaction_id: Unique transaction identifier
            
        Returns:
            Transaction if found, None otherwise
        """
        with session_scope() as session:
            repo = TransactionRepository(session)
            tx = repo.get_by_transaction_id(transaction_id)
            
            if not tx:
                return None
            
            return TransactionType(
                transaction_id=tx.transaction_id,
                customer_id=tx.customer_id,
                merchant_id=tx.merchant_id,
                amount=float(tx.amount),
                timestamp=tx.timestamp,
                is_fraud=tx.is_fraud,
                fraud_type=tx.fraud_type,
                category=tx.category or "",
                payment_mode=tx.payment_mode or "",
                city=tx.city or "",
                state=tx.state,
                latitude=float(tx.latitude) if tx.latitude else None,
                longitude=float(tx.longitude) if tx.longitude else None,
                fraud_confidence=float(tx.fraud_confidence) if tx.fraud_confidence else None,
                is_anomaly=tx.is_anomaly,
                anomaly_score=float(tx.anomaly_score) if tx.anomaly_score else None,
                anomaly_type=tx.anomaly_type,
                risk_score=float(tx.risk_score) if tx.risk_score else None,
                velocity_score=float(tx.velocity_score) if tx.velocity_score else None
            )
    
    @strawberry.field
    async def customers(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[CustomerFilterInput] = None
    ) -> List[CustomerType]:
        """
        Query customers with optional filters.
        
        Args:
            limit: Maximum number of customers to return
            offset: Number of customers to skip
            filters: Optional filters for customers
            
        Returns:
            List of customers matching the criteria
        """
        with session_scope() as session:
            repo = CustomerRepository(session)
            
            # For now, just get all customers - filtering can be enhanced later
            customers = repo.get_all(limit=limit, offset=offset)
            
            # Convert to GraphQL types
            return [
                CustomerType(
                    customer_id=c.customer_id,
                    name=c.name,
                    age=c.age or 0,
                    gender=c.gender or "unknown",
                    occupation=c.occupation or "unknown",
                    income_bracket=c.income_bracket or "unknown",
                    customer_segment=c.customer_segment or "regular",
                    city=c.city,
                    state=c.state,
                    digital_savviness=float(c.digital_savviness) if c.digital_savviness else None,
                    loyalty_score=float(c.loyalty_score) if c.loyalty_score else None,
                    risk_level=c.risk_category or "low",
                    total_transactions=c.transaction_count or 0,
                    total_spent=float(c.total_spent) if c.total_spent else 0.0,
                    avg_transaction_amount=float(c.avg_transaction_amount) if c.avg_transaction_amount else 0.0
                )
                for c in customers
            ]
    
    @strawberry.field
    async def customer(
        self,
        customer_id: str
    ) -> Optional[CustomerType]:
        """
        Get a single customer by ID.
        
        Args:
            customer_id: Unique customer identifier
            
        Returns:
            Customer if found, None otherwise
        """
        with session_scope() as session:
            repo = CustomerRepository(session)
            c = repo.get_by_customer_id(customer_id)
            
            if not c:
                return None
            
            return CustomerType(
                customer_id=c.customer_id,
                name=c.name,
                age=c.age or 0,
                gender=c.gender or "unknown",
                occupation=c.occupation or "unknown",
                income_bracket=c.income_bracket or "unknown",
                customer_segment=c.customer_segment or "regular",
                city=c.city,
                state=c.state,
                digital_savviness=float(c.digital_savviness) if c.digital_savviness else None,
                loyalty_score=float(c.loyalty_score) if c.loyalty_score else None,
                risk_level=c.risk_category or "low",
                total_transactions=c.transaction_count or 0,
                total_spent=float(c.total_spent) if c.total_spent else 0.0,
                avg_transaction_amount=float(c.avg_transaction_amount) if c.avg_transaction_amount else 0.0
            )
    
    @strawberry.field
    async def merchants(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[MerchantFilterInput] = None
    ) -> List[MerchantType]:
        """
        Query merchants with optional filters.
        
        Args:
            limit: Maximum number of merchants to return
            offset: Number of merchants to skip
            filters: Optional filters for merchants
            
        Returns:
            List of merchants matching the criteria
        """
        with session_scope() as session:
            repo = MerchantRepository(session)
            
            # Apply filters if provided
            if filters and filters.category:
                merchants = repo.get_by_category(category=filters.category, limit=limit)
            else:
                merchants = repo.get_all(limit=limit, offset=offset)
            
            # Convert to GraphQL types
            return [
                MerchantType(
                    merchant_id=m.merchant_id,
                    name=m.name,
                    category=m.category or "unknown",
                    subcategory=m.subcategory,
                    city=m.city,
                    state=m.state,
                    reputation_score=float(m.reputation_score) if m.reputation_score else None,
                    is_chain=m.is_chain or False,
                    risk_level=m.risk_category or "low",
                    total_transactions=m.total_transactions or 0,
                    total_revenue=float(m.total_revenue) if m.total_revenue else 0.0
                )
                for m in merchants
            ]
    
    @strawberry.field
    async def merchant(
        self,
        merchant_id: str
    ) -> Optional[MerchantType]:
        """
        Get a single merchant by ID.
        
        Args:
            merchant_id: Unique merchant identifier
            
        Returns:
            Merchant if found, None otherwise
        """
        with session_scope() as session:
            repo = MerchantRepository(session)
            m = repo.get_by_merchant_id(merchant_id)
            
            if not m:
                return None
            
            return MerchantType(
                merchant_id=m.merchant_id,
                name=m.name,
                category=m.category or "unknown",
                subcategory=m.subcategory,
                city=m.city,
                state=m.state,
                reputation_score=float(m.reputation_score) if m.reputation_score else None,
                is_chain=m.is_chain or False,
                risk_level=m.risk_category or "low",
                total_transactions=m.total_transactions or 0,
                total_revenue=float(m.total_revenue) if m.total_revenue else 0.0
            )
    
    @strawberry.field
    async def ml_features(
        self,
        transaction_id: str
    ) -> Optional[MLFeaturesType]:
        """
        Get ML features for a specific transaction.
        
        Args:
            transaction_id: Unique transaction identifier
            
        Returns:
            ML features if found, None otherwise
        """
        with session_scope() as session:
            repo = MLFeaturesRepository(session)
            features = repo.get_by_transaction_id(transaction_id)
            
            if not features:
                return None
            
            return MLFeaturesType(
                transaction_id=features.transaction_id,
                daily_transaction_count=features.daily_transaction_count or 0,
                weekly_transaction_count=features.weekly_transaction_count or 0,
                daily_transaction_amount=float(features.daily_transaction_amount) if features.daily_transaction_amount else 0.0,
                average_daily_amount=float(features.average_daily_amount) if features.average_daily_amount else 0.0,
                transaction_frequency_1h=features.transaction_frequency_1h or 0,
                transaction_frequency_24h=features.transaction_frequency_24h or 0,
                amount_velocity_1h=float(features.amount_velocity_1h) if features.amount_velocity_1h else 0.0,
                distance_from_home=float(features.distance_from_home) if features.distance_from_home else 0.0,
                unique_cities_7d=features.unique_cities_7d or 0,
                is_unusual_hour=features.is_unusual_hour or False,
                is_weekend=features.is_weekend or False,
                hour_of_day=features.hour_of_day or 0,
                day_of_week=features.day_of_week or 0,
                category_diversity_7d=features.category_diversity_7d or 0
            )
    
    @strawberry.field
    async def fraud_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[FraudPatternType]:
        """
        Query detected fraud patterns.
        
        Args:
            pattern_type: Filter by specific pattern type
            min_confidence: Minimum confidence threshold
            limit: Maximum number of patterns to return
            
        Returns:
            List of detected fraud patterns
        """
        # TODO: Implement fraud pattern retrieval
        return []
    
    @strawberry.field
    async def system_health(self) -> SystemHealthType:
        """
        Get current system health status.
        
        Returns:
            System health information including component status
        """
        # TODO: Implement actual health checks
        return SystemHealthType(
            status="healthy",
            timestamp=datetime.now(),
            database="healthy",
            cache="healthy",
            api="healthy",
            ml_model="healthy",
            uptime_seconds=0,
            cpu_usage_percent=0.0,
            memory_usage_percent=0.0,
            active_connections=0,
        )
    
    @strawberry.field
    async def generation_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> GenerationStatsType:
        """
        Get statistics about data generation.
        
        Args:
            start_date: Start of time window (default: 24 hours ago)
            end_date: End of time window (default: now)
            
        Returns:
            Generation statistics for the specified time window
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=1)
        
        # TODO: Implement actual statistics calculation
        return GenerationStatsType(
            total_transactions_generated=0,
            total_customers=0,
            total_merchants=0,
            fraud_count=0,
            fraud_rate=0.0,
            anomaly_count=0,
            anomaly_rate=0.0,
            generation_rate_per_second=0.0,
            average_processing_time_ms=0.0,
            period_start=start_date,
            period_end=end_date,
        )
    
    @strawberry.field
    async def search_transactions(
        self,
        query: str,
        limit: int = 50
    ) -> List[TransactionType]:
        """
        Full-text search across transactions.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching transactions
        """
        # TODO: Implement full-text search
        return []
