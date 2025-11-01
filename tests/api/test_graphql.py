"""
Tests for GraphQL API

Tests queries, mutations, and subscriptions.
"""

import pytest
from datetime import datetime
import asyncio

from src.api.graphql.schema import schema


@pytest.fixture
def graphql_client():
    """Create a simple wrapper for executing GraphQL queries."""
    class SimpleClient:
        async def execute_async(self, query):
            return await schema.execute(query)
        
        def execute(self, query):
            return asyncio.run(self.execute_async(query))
    
    return SimpleClient()


class TestGraphQLQueries:
    """Test GraphQL queries"""
    
    def test_schema_builds(self, graphql_client):
        """Test that the GraphQL schema builds successfully."""
        assert schema is not None
        assert schema.query is not None
    
    def test_system_health_query(self, graphql_client):
        """Test system health query."""
        query = """
            query {
                systemHealth {
                    status
                    timestamp
                    database
                    cache
                    api
                    mlModel
                    uptimeSeconds
                    cpuUsagePercent
                    memoryUsagePercent
                    activeConnections
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
        assert result.data is not None
        assert "systemHealth" in result.data
        assert result.data["systemHealth"]["status"] == "healthy"
    
    def test_generation_stats_query(self, graphql_client):
        """Test generation statistics query."""
        query = """
            query {
                generationStats {
                    totalTransactionsGenerated
                    totalCustomers
                    totalMerchants
                    fraudCount
                    fraudRate
                    anomalyCount
                    anomalyRate
                    generationRatePerSecond
                    averageProcessingTimeMs
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
        assert result.data is not None
        assert "generationStats" in result.data
    
    def test_transactions_query_empty(self, graphql_client):
        """Test transactions query returns empty list initially."""
        query = """
            query {
                transactions(limit: 10) {
                    transactionId
                    customerId
                    merchantId
                    amount
                    category
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
        assert result.data is not None
        assert "transactions" in result.data
        assert isinstance(result.data["transactions"], list)
    
    def test_transactions_query_with_filters(self, graphql_client):
        """Test transactions query with filters."""
        query = """
            query {
                transactions(
                    limit: 5,
                    filters: {
                        category: "groceries",
                        minAmount: 100.0,
                        isFraud: false
                    }
                ) {
                    transactionId
                    category
                    amount
                    isFraud
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
        assert result.data is not None
    
    def test_customer_query(self, graphql_client):
        """Test single customer query."""
        query = """
            query {
                customer(customerId: "CUST001") {
                    customerId
                    name
                    age
                    occupation
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
    
    def test_merchant_query(self, graphql_client):
        """Test single merchant query."""
        query = """
            query {
                merchant(merchantId: "MERCH001") {
                    merchantId
                    name
                    category
                    isChain
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
    
    def test_ml_features_query(self, graphql_client):
        """Test ML features query."""
        query = """
            query {
                mlFeatures(transactionId: "TXN001") {
                    transactionId
                    dailyTransactionCount
                    weeklyTransactionCount
                    averageDailyAmount
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
    
    def test_fraud_patterns_query(self, graphql_client):
        """Test fraud patterns query."""
        query = """
            query {
                fraudPatterns(minConfidence: 0.8, limit: 10) {
                    patternId
                    transactionId
                    patternType
                    confidence
                    severity
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
    
    def test_search_transactions_query(self, graphql_client):
        """Test transaction search query."""
        query = """
            query {
                searchTransactions(query: "groceries", limit: 5) {
                    transactionId
                    category
                    amount
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0


class TestGraphQLMutations:
    """Test GraphQL mutations"""
    
    def test_generate_transactions_mutation(self, graphql_client):
        """Test transaction generation mutation."""
        mutation = """
            mutation {
                generateTransactions(
                    input: {
                        count: 100,
                        fraudRate: 0.05,
                        seed: 42,
                        outputFormat: "csv"
                    }
                ) {
                    success
                    message
                    transactionsGenerated
                    fraudInjected
                    executionTimeSeconds
                }
            }
        """
        
        result = graphql_client.execute(mutation)
        assert result.errors is None or len(result.errors) == 0
        assert result.data is not None
    
    def test_train_model_mutation(self, graphql_client):
        """Test model training mutation."""
        mutation = """
            mutation {
                trainModel(
                    input: {
                        algorithm: "random_forest",
                        trainSplitRatio: 0.8,
                        crossValidationFolds: 5
                    }
                ) {
                    success
                    message
                    modelId
                    algorithm
                    accuracy
                    precision
                    recall
                    f1Score
                }
            }
        """
        
        result = graphql_client.execute(mutation)
        assert result.errors is None or len(result.errors) == 0
    
    def test_detect_fraud_mutation(self, graphql_client):
        """Test fraud detection mutation."""
        mutation = """
            mutation {
                detectFraud(transactionId: "TXN001")
            }
        """
        
        result = graphql_client.execute(mutation)
        assert result.errors is None or len(result.errors) == 0
    
    def test_validate_data_mutation(self, graphql_client):
        """Test data validation mutation."""
        mutation = """
            mutation {
                validateData(datasetId: "DS001") {
                    success
                    qualityScore
                    totalChecks
                    passedChecks
                    failedChecks
                    warnings
                }
            }
        """
        
        result = graphql_client.execute(mutation)
        assert result.errors is None or len(result.errors) == 0
    
    def test_clear_cache_mutation(self, graphql_client):
        """Test cache clear mutation."""
        mutation = """
            mutation {
                clearCache
            }
        """
        
        result = graphql_client.execute(mutation)
        assert result.errors is None or len(result.errors) == 0


class TestGraphQLTypes:
    """Test GraphQL type definitions"""
    
    def test_transaction_type_fields(self):
        """Test that TransactionType has all required fields."""
        from src.api.graphql.types import TransactionType
        
        # Verify the type has expected attributes
        assert hasattr(TransactionType, "__annotations__")
        annotations = TransactionType.__annotations__
        
        assert "transaction_id" in annotations
        assert "customer_id" in annotations
        assert "merchant_id" in annotations
        assert "amount" in annotations
        assert "is_fraud" in annotations
    
    def test_customer_type_fields(self):
        """Test that CustomerType has all required fields."""
        from src.api.graphql.types import CustomerType
        
        annotations = CustomerType.__annotations__
        assert "customer_id" in annotations
        assert "name" in annotations
        assert "age" in annotations
    
    def test_merchant_type_fields(self):
        """Test that MerchantType has all required fields."""
        from src.api.graphql.types import MerchantType
        
        annotations = MerchantType.__annotations__
        assert "merchant_id" in annotations
        assert "name" in annotations
        assert "category" in annotations


class TestGraphQLSchema:
    """Test GraphQL schema structure"""
    
    def test_schema_exists(self):
        """Test that schema exists."""
        assert schema is not None
        assert hasattr(schema, 'execute')
    
    def test_schema_has_query(self):
        """Test that schema has Query configured."""
        assert schema.query is not None
    
    def test_schema_has_mutation(self):
        """Test that schema has Mutation configured."""
        assert schema.mutation is not None
    
    def test_schema_has_subscription(self):
        """Test that schema has Subscription configured."""
        assert schema.subscription is not None
    
    def test_introspection_query(self, graphql_client):
        """Test GraphQL introspection query."""
        query = """
            query {
                __schema {
                    queryType {
                        name
                    }
                    mutationType {
                        name
                    }
                    subscriptionType {
                        name
                    }
                }
            }
        """
        
        result = graphql_client.execute(query)
        assert result.errors is None or len(result.errors) == 0
        assert result.data is not None
        assert result.data["__schema"]["queryType"]["name"] == "Query"
        assert result.data["__schema"]["mutationType"]["name"] == "Mutation"
        assert result.data["__schema"]["subscriptionType"]["name"] == "Subscription"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
