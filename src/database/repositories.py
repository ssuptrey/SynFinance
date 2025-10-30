"""
Repository Pattern for SynFinance Database

Provides CRUD operations and business logic for database entities.

Week 7 Day 5: Database Integration
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func, select
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import Transaction, Customer, Merchant, MLFeatures, ModelPrediction
from src.observability import get_logger, LogCategory

logger = get_logger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations"""
    
    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> Any:
        """Create a new entity"""
        try:
            obj = self.model_class(**kwargs)
            self.session.add(obj)
            self.session.flush()
            return obj
        except Exception as e:
            logger.error(
                f"Failed to create {self.model_class.__name__}",
                category=LogCategory.DATABASE,
                extra={"error": str(e)}
            )
            raise
    
    def get_by_id(self, id: int) -> Optional[Any]:
        """Get entity by ID"""
        return self.session.query(self.model_class).filter(
            self.model_class.id == id
        ).first()
    
    def get_all(self, limit: int = 1000, offset: int = 0) -> List[Any]:
        """Get all entities with pagination"""
        return self.session.query(self.model_class).limit(limit).offset(offset).all()
    
    def update(self, id: int, **kwargs) -> Optional[Any]:
        """Update entity by ID"""
        obj = self.get_by_id(id)
        if obj:
            for key, value in kwargs.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            self.session.flush()
        return obj
    
    def delete(self, id: int) -> bool:
        """Delete entity by ID"""
        obj = self.get_by_id(id)
        if obj:
            self.session.delete(obj)
            self.session.flush()
            return True
        return False
    
    def count(self) -> int:
        """Count total entities"""
        return self.session.query(func.count(self.model_class.id)).scalar()


class TransactionRepository(BaseRepository):
    """Repository for Transaction operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, Transaction)
    
    def get_by_transaction_id(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by transaction_id"""
        return self.session.query(Transaction).filter(
            Transaction.transaction_id == transaction_id
        ).first()
    
    def get_by_customer(
        self,
        customer_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Transaction]:
        """Get transactions by customer ID"""
        return self.session.query(Transaction).filter(
            Transaction.customer_id == customer_id
        ).order_by(desc(Transaction.timestamp)).limit(limit).offset(offset).all()
    
    def get_by_merchant(
        self,
        merchant_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Transaction]:
        """Get transactions by merchant ID"""
        return self.session.query(Transaction).filter(
            Transaction.merchant_id == merchant_id
        ).order_by(desc(Transaction.timestamp)).limit(limit).offset(offset).all()
    
    def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Transaction]:
        """Get transactions within date range"""
        return self.session.query(Transaction).filter(
            and_(
                Transaction.timestamp >= start_date,
                Transaction.timestamp <= end_date
            )
        ).order_by(desc(Transaction.timestamp)).limit(limit).offset(offset).all()
    
    def get_fraud_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Transaction]:
        """Get fraud transactions"""
        query = self.session.query(Transaction).filter(
            Transaction.is_fraud == True
        )
        
        if start_date:
            query = query.filter(Transaction.timestamp >= start_date)
        if end_date:
            query = query.filter(Transaction.timestamp <= end_date)
        
        return query.order_by(desc(Transaction.timestamp)).limit(limit).offset(offset).all()
    
    def get_anomaly_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Transaction]:
        """Get anomaly transactions"""
        query = self.session.query(Transaction).filter(
            Transaction.is_anomaly == True
        )
        
        if start_date:
            query = query.filter(Transaction.timestamp >= start_date)
        if end_date:
            query = query.filter(Transaction.timestamp <= end_date)
        
        return query.order_by(desc(Transaction.timestamp)).limit(limit).offset(offset).all()
    
    def get_by_amount_range(
        self,
        min_amount: Decimal,
        max_amount: Decimal,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions within amount range"""
        return self.session.query(Transaction).filter(
            and_(
                Transaction.amount >= min_amount,
                Transaction.amount <= max_amount
            )
        ).limit(limit).all()
    
    def get_high_value_transactions(
        self,
        threshold: Decimal,
        limit: int = 100
    ) -> List[Transaction]:
        """Get high-value transactions"""
        return self.session.query(Transaction).filter(
            Transaction.amount >= threshold
        ).order_by(desc(Transaction.amount)).limit(limit).all()
    
    def bulk_create(self, transactions: List[Dict[str, Any]]) -> int:
        """Bulk create transactions"""
        try:
            tx_objects = [Transaction(**tx_data) for tx_data in transactions]
            self.session.bulk_save_objects(tx_objects)
            self.session.flush()
            
            logger.info(
                f"Bulk created {len(transactions)} transactions",
                category=LogCategory.DATABASE
            )
            
            return len(transactions)
            
        except Exception as e:
            logger.error(
                "Bulk create transactions failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e), "count": len(transactions)}
            )
            raise
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get transaction statistics"""
        query = self.session.query(
            func.count(Transaction.id).label('total_count'),
            func.sum(Transaction.amount).label('total_amount'),
            func.avg(Transaction.amount).label('avg_amount'),
            func.min(Transaction.amount).label('min_amount'),
            func.max(Transaction.amount).label('max_amount'),
            func.count(Transaction.id).filter(Transaction.is_fraud == True).label('fraud_count'),
            func.count(Transaction.id).filter(Transaction.is_anomaly == True).label('anomaly_count')
        )
        
        if start_date:
            query = query.filter(Transaction.timestamp >= start_date)
        if end_date:
            query = query.filter(Transaction.timestamp <= end_date)
        
        result = query.first()
        
        return {
            'total_count': result.total_count or 0,
            'total_amount': float(result.total_amount or 0),
            'avg_amount': float(result.avg_amount or 0),
            'min_amount': float(result.min_amount or 0),
            'max_amount': float(result.max_amount or 0),
            'fraud_count': result.fraud_count or 0,
            'anomaly_count': result.anomaly_count or 0,
            'fraud_rate': (result.fraud_count / result.total_count * 100) if result.total_count else 0
        }


class CustomerRepository(BaseRepository):
    """Repository for Customer operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, Customer)
    
    def get_by_customer_id(self, customer_id: str) -> Optional[Customer]:
        """Get customer by customer_id"""
        return self.session.query(Customer).filter(
            Customer.customer_id == customer_id
        ).first()
    
    def get_by_email(self, email: str) -> Optional[Customer]:
        """Get customer by email"""
        return self.session.query(Customer).filter(
            Customer.email == email
        ).first()
    
    def get_by_risk_category(
        self,
        risk_category: str,
        limit: int = 100
    ) -> List[Customer]:
        """Get customers by risk category"""
        return self.session.query(Customer).filter(
            Customer.risk_category == risk_category
        ).limit(limit).all()
    
    def get_high_risk_customers(
        self,
        risk_threshold: float = 0.7,
        limit: int = 100
    ) -> List[Customer]:
        """Get high-risk customers"""
        return self.session.query(Customer).filter(
            Customer.risk_score >= risk_threshold
        ).order_by(desc(Customer.risk_score)).limit(limit).all()
    
    def get_active_customers(self, limit: int = 1000) -> List[Customer]:
        """Get active customers"""
        return self.session.query(Customer).filter(
            Customer.account_status == 'active'
        ).limit(limit).all()
    
    def update_transaction_stats(self, customer_id: str) -> None:
        """Update customer transaction statistics"""
        customer = self.get_by_customer_id(customer_id)
        if not customer:
            return
        
        # Calculate statistics from transactions
        stats = self.session.query(
            func.count(Transaction.id).label('tx_count'),
            func.sum(Transaction.amount).label('total_spent'),
            func.avg(Transaction.amount).label('avg_amount'),
            func.count(Transaction.id).filter(Transaction.is_fraud == True).label('fraud_count')
        ).filter(
            Transaction.customer_id == customer_id
        ).first()
        
        customer.transaction_count = stats.tx_count or 0
        customer.total_spent = stats.total_spent or Decimal(0)
        customer.avg_transaction_amount = stats.avg_amount or Decimal(0)
        customer.fraud_history_count = stats.fraud_count or 0
        
        self.session.flush()
    
    def bulk_create(self, customers: List[Dict[str, Any]]) -> int:
        """Bulk create customers"""
        try:
            customer_objects = [Customer(**cust_data) for cust_data in customers]
            self.session.bulk_save_objects(customer_objects)
            self.session.flush()
            
            logger.info(
                f"Bulk created {len(customers)} customers",
                category=LogCategory.DATABASE
            )
            
            return len(customers)
            
        except Exception as e:
            logger.error(
                "Bulk create customers failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e), "count": len(customers)}
            )
            raise


class MerchantRepository(BaseRepository):
    """Repository for Merchant operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, Merchant)
    
    def get_by_merchant_id(self, merchant_id: str) -> Optional[Merchant]:
        """Get merchant by merchant_id"""
        return self.session.query(Merchant).filter(
            Merchant.merchant_id == merchant_id
        ).first()
    
    def get_by_category(
        self,
        category: str,
        limit: int = 100
    ) -> List[Merchant]:
        """Get merchants by category"""
        return self.session.query(Merchant).filter(
            Merchant.category == category
        ).limit(limit).all()
    
    def get_high_risk_merchants(
        self,
        risk_threshold: float = 0.7,
        limit: int = 100
    ) -> List[Merchant]:
        """Get high-risk merchants"""
        return self.session.query(Merchant).filter(
            Merchant.risk_score >= risk_threshold
        ).order_by(desc(Merchant.risk_score)).limit(limit).all()
    
    def update_transaction_stats(self, merchant_id: str) -> None:
        """Update merchant transaction statistics"""
        merchant = self.get_by_merchant_id(merchant_id)
        if not merchant:
            return
        
        # Calculate statistics from transactions
        stats = self.session.query(
            func.count(Transaction.id).label('tx_count'),
            func.sum(Transaction.amount).label('total_revenue'),
            func.avg(Transaction.amount).label('avg_amount'),
            func.count(Transaction.id).filter(Transaction.is_fraud == True).label('fraud_count')
        ).filter(
            Transaction.merchant_id == merchant_id
        ).first()
        
        merchant.total_transactions = stats.tx_count or 0
        merchant.total_revenue = stats.total_revenue or Decimal(0)
        merchant.avg_transaction_amount = stats.avg_amount or Decimal(0)
        merchant.fraud_report_count = stats.fraud_count or 0
        
        self.session.flush()
    
    def bulk_create(self, merchants: List[Dict[str, Any]]) -> int:
        """Bulk create merchants"""
        try:
            merchant_objects = [Merchant(**merch_data) for merch_data in merchants]
            self.session.bulk_save_objects(merchant_objects)
            self.session.flush()
            
            logger.info(
                f"Bulk created {len(merchants)} merchants",
                category=LogCategory.DATABASE
            )
            
            return len(merchants)
            
        except Exception as e:
            logger.error(
                "Bulk create merchants failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e), "count": len(merchants)}
            )
            raise


class MLFeaturesRepository(BaseRepository):
    """Repository for MLFeatures operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, MLFeatures)
    
    def get_by_transaction_id(self, transaction_id: str) -> Optional[MLFeatures]:
        """Get features by transaction_id"""
        return self.session.query(MLFeatures).filter(
            MLFeatures.transaction_id == transaction_id
        ).first()
    
    def bulk_create(self, features: List[Dict[str, Any]]) -> int:
        """Bulk create ML features"""
        try:
            feature_objects = [MLFeatures(**feat_data) for feat_data in features]
            self.session.bulk_save_objects(feature_objects)
            self.session.flush()
            
            logger.info(
                f"Bulk created {len(features)} feature sets",
                category=LogCategory.DATABASE
            )
            
            return len(features)
            
        except Exception as e:
            logger.error(
                "Bulk create features failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e), "count": len(features)}
            )
            raise


class ModelPredictionRepository(BaseRepository):
    """Repository for ModelPrediction operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, ModelPrediction)
    
    def get_by_transaction_id(
        self,
        transaction_id: str
    ) -> List[ModelPrediction]:
        """Get predictions for a transaction"""
        return self.session.query(ModelPrediction).filter(
            ModelPrediction.transaction_id == transaction_id
        ).all()
    
    def get_by_model(
        self,
        model_name: str,
        model_version: str,
        limit: int = 100
    ) -> List[ModelPrediction]:
        """Get predictions by model name and version"""
        return self.session.query(ModelPrediction).filter(
            and_(
                ModelPrediction.model_name == model_name,
                ModelPrediction.model_version == model_version
            )
        ).order_by(desc(ModelPrediction.created_at)).limit(limit).all()
    
    def get_model_performance(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Get model performance metrics"""
        predictions = self.session.query(ModelPrediction).filter(
            and_(
                ModelPrediction.model_name == model_name,
                ModelPrediction.model_version == model_version,
                ModelPrediction.ground_truth.isnot(None)
            )
        ).all()
        
        if not predictions:
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0
            }
        
        total = len(predictions)
        correct = sum(1 for p in predictions if p.is_correct)
        true_positives = sum(1 for p in predictions if p.prediction and p.ground_truth)
        false_positives = sum(1 for p in predictions if p.prediction and not p.ground_truth)
        false_negatives = sum(1 for p in predictions if not p.prediction and p.ground_truth)
        
        accuracy = correct / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def bulk_create(self, predictions: List[Dict[str, Any]]) -> int:
        """Bulk create predictions"""
        try:
            prediction_objects = [ModelPrediction(**pred_data) for pred_data in predictions]
            self.session.bulk_save_objects(prediction_objects)
            self.session.flush()
            
            logger.info(
                f"Bulk created {len(predictions)} predictions",
                category=LogCategory.DATABASE
            )
            
            return len(predictions)
            
        except Exception as e:
            logger.error(
                "Bulk create predictions failed",
                category=LogCategory.DATABASE,
                extra={"error": str(e), "count": len(predictions)}
            )
            raise
