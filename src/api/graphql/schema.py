"""
GraphQL Schema

Main GraphQL schema combining queries, mutations, and subscriptions.
"""

import strawberry
from strawberry.fastapi import GraphQLRouter

from .resolvers import Query, Mutation, Subscription


# Create the GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)


# Create FastAPI GraphQL router
def create_graphql_router(path: str = "/graphql") -> GraphQLRouter:
    """
    Create a GraphQL router for FastAPI.
    
    Args:
        path: URL path for GraphQL endpoint
        
    Returns:
        Configured GraphQLRouter instance
    """
    return GraphQLRouter(
        schema=schema,
        path=path,
        graphql_ide="apollo-sandbox"  # Use Apollo Sandbox for testing
    )


__all__ = ["schema", "create_graphql_router"]
