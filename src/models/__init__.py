"""
Data models for SynFinance

Week 3: Transaction dataclass with 36+ fields
"""

from src.models.transaction import (
    Transaction,
    TRANSACTION_FIELD_COUNT,
    FIELD_CATEGORIES,
    TOTAL_FIELDS_WITH_LEGACY
)

__all__ = [
    'Transaction',
    'TRANSACTION_FIELD_COUNT',
    'FIELD_CATEGORIES',
    'TOTAL_FIELDS_WITH_LEGACY',
]
