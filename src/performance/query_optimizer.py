"""
Query Optimizer

Database query optimization using EXPLAIN ANALYZE, indexing, and connection pooling.
Provides query analysis, slow query detection, and index recommendations.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import re
from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool
import pandas as pd


class QueryAnalysis:
    """Query execution analysis result."""
    
    def __init__(self, query: str):
        self.query = query
        self.execution_time_ms: float = 0.0
        self.rows_scanned: int = 0
        self.rows_returned: int = 0
        self.plan: str = ''
        self.cost: float = 0.0
        self.index_used: Optional[str] = None
        self.recommendations: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'execution_time_ms': self.execution_time_ms,
            'rows_scanned': self.rows_scanned,
            'rows_returned': self.rows_returned,
            'plan': self.plan,
            'cost': self.cost,
            'index_used': self.index_used,
            'recommendations': self.recommendations
        }


class IndexRecommendation:
    """Index recommendation."""
    
    def __init__(self, table: str, columns: List[str], reason: str, 
                 estimated_improvement: float = 0.0):
        self.table = table
        self.columns = columns
        self.reason = reason
        self.estimated_improvement = estimated_improvement
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'table': self.table,
            'columns': self.columns,
            'reason': self.reason,
            'estimated_improvement': self.estimated_improvement,
            'created_at': self.created_at.isoformat()
        }


class QueryOptimizer:
    """Database query optimization and analysis."""
    
    def __init__(self, database_url: str, cache_manager=None):
        """
        Initialize query optimizer.
        
        Args:
            database_url: Database connection URL
            cache_manager: Optional cache manager for query result caching
        """
        self.database_url = database_url
        self.cache_manager = cache_manager
        
        # Create engine with optimized connection pool
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        
        # Track slow queries
        self.slow_queries: List[QueryAnalysis] = []
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        
        # Track index recommendations
        self.index_recommendations: List[IndexRecommendation] = []
        
        # Setup query event listeners
        self._setup_query_listeners()
    
    def _setup_query_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for query tracking."""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, 
                                         context, executemany):
            conn.info.setdefault('query_start_time', []).append(time.time())
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters,
                                        context, executemany):
            total = time.time() - conn.info['query_start_time'].pop()
            
            # Track query execution time
            query_hash = hash(statement)
            if query_hash not in self.query_stats:
                self.query_stats[query_hash] = {
                    'query': statement,
                    'executions': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0
                }
            
            stats = self.query_stats[query_hash]
            stats['executions'] += 1
            stats['total_time'] += total
            stats['min_time'] = min(stats['min_time'], total)
            stats['max_time'] = max(stats['max_time'], total)
            
            # Detect slow queries
            if total * 1000 > 100:  # 100ms threshold
                analysis = QueryAnalysis(statement)
                analysis.execution_time_ms = total * 1000
                self.slow_queries.append(analysis)
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query execution plan using EXPLAIN ANALYZE.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            QueryAnalysis with execution details
        """
        analysis = QueryAnalysis(query)
        
        try:
            with self.engine.connect() as conn:
                # Get execution plan (PostgreSQL syntax)
                explain_query = f"EXPLAIN ANALYZE {query}"
                
                start_time = time.time()
                result = conn.execute(text(explain_query))
                execution_time = (time.time() - start_time) * 1000
                
                plan_lines = [row[0] for row in result]
                analysis.plan = '\n'.join(plan_lines)
                analysis.execution_time_ms = execution_time
                
                # Parse plan for details
                for line in plan_lines:
                    # Extract rows
                    if 'rows=' in line:
                        match = re.search(r'rows=(\d+)', line)
                        if match:
                            analysis.rows_scanned = int(match.group(1))
                    
                    # Extract cost
                    if 'cost=' in line:
                        match = re.search(r'cost=[\d.]+\.\.(\d+\.?\d*)', line)
                        if match:
                            analysis.cost = float(match.group(1))
                    
                    # Check for index usage
                    if 'Index Scan' in line or 'Index Only Scan' in line:
                        match = re.search(r'on (\w+)', line)
                        if match:
                            analysis.index_used = match.group(1)
                
                # Generate recommendations
                analysis.recommendations = self._generate_recommendations(
                    query, analysis
                )
        
        except Exception as e:
            analysis.recommendations.append(f"Error analyzing query: {str(e)}")
        
        return analysis
    
    def _generate_recommendations(self, query: str, 
                                 analysis: QueryAnalysis) -> List[str]:
        """Generate optimization recommendations based on query analysis."""
        recommendations = []
        
        # Check for missing indexes
        if analysis.index_used is None and analysis.execution_time_ms > 50:
            recommendations.append(
                "Query does not use an index. Consider adding an index."
            )
        
        # Check for full table scans
        if analysis.rows_scanned > 1000 and analysis.rows_returned < 100:
            recommendations.append(
                f"Scanning {analysis.rows_scanned} rows to return "
                f"{analysis.rows_returned}. Add more selective filters."
            )
        
        # Check for high cost
        if analysis.cost > 10000:
            recommendations.append(
                f"High query cost ({analysis.cost:.0f}). "
                "Consider optimizing joins or adding indexes."
            )
        
        # Check for SELECT *
        if 'SELECT *' in query.upper():
            recommendations.append(
                "Avoid SELECT *. Specify only needed columns."
            )
        
        return recommendations
    
    def create_index(self, table: str, columns: List[str],
                    index_name: Optional[str] = None,
                    index_type: str = 'btree',
                    unique: bool = False) -> bool:
        """
        Create database index.
        
        Args:
            table: Table name
            columns: List of column names
            index_name: Optional custom index name
            index_type: Index type (btree, hash, gin, gist)
            unique: Whether index should be unique
            
        Returns:
            True if index created successfully
        """
        if not index_name:
            col_str = '_'.join(columns)
            index_name = f"idx_{table}_{col_str}"
        
        columns_str = ', '.join(columns)
        unique_str = 'UNIQUE ' if unique else ''
        
        create_sql = (
            f"CREATE {unique_str}INDEX {index_name} "
            f"ON {table} USING {index_type} ({columns_str})"
        )
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.commit()
            return True
        except Exception as e:
            print(f"Failed to create index: {e}")
            return False
    
    def detect_slow_queries(self, threshold_ms: float = 100) -> List[QueryAnalysis]:
        """
        Get slow queries that exceeded threshold.
        
        Args:
            threshold_ms: Threshold in milliseconds
            
        Returns:
            List of slow query analyses
        """
        return [
            q for q in self.slow_queries 
            if q.execution_time_ms > threshold_ms
        ]
    
    def recommend_indexes(self, min_executions: int = 10) -> List[IndexRecommendation]:
        """
        Recommend indexes based on query patterns.
        
        Args:
            min_executions: Minimum query executions to consider
            
        Returns:
            List of index recommendations
        """
        recommendations = []
        
        for query_hash, stats in self.query_stats.items():
            if stats['executions'] < min_executions:
                continue
            
            query = stats['query']
            avg_time = stats['total_time'] / stats['executions']
            
            # Only recommend for slow queries
            if avg_time * 1000 < 50:
                continue
            
            # Extract table and WHERE columns
            tables = self._extract_tables(query)
            where_columns = self._extract_where_columns(query)
            
            for table in tables:
                if where_columns:
                    recommendation = IndexRecommendation(
                        table=table,
                        columns=where_columns[:3],  # Max 3 columns
                        reason=f"Slow query ({avg_time*1000:.1f}ms avg) "
                               f"executed {stats['executions']} times",
                        estimated_improvement=min(avg_time * 0.5, 1.0)
                    )
                    recommendations.append(recommendation)
        
        self.index_recommendations = recommendations
        return recommendations
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = []
        
        # Simple regex to find FROM and JOIN table names
        from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))
        
        join_matches = re.findall(r'JOIN\s+(\w+)', query, re.IGNORECASE)
        tables.extend(join_matches)
        
        return tables
    
    def _extract_where_columns(self, query: str) -> List[str]:
        """Extract column names from WHERE clause."""
        columns = []
        
        # Find WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', 
                               query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return columns
        
        where_clause = where_match.group(1)
        
        # Extract column names (simple pattern)
        col_matches = re.findall(r'(\w+)\s*[=<>]', where_clause)
        columns.extend(col_matches)
        
        return list(set(columns))  # Remove duplicates
    
    def batch_execute(self, queries: List[str]) -> List[Any]:
        """
        Execute multiple queries in batch.
        
        Args:
            queries: List of SQL queries
            
        Returns:
            List of query results
        """
        results = []
        
        with self.engine.connect() as conn:
            for query in queries:
                result = conn.execute(text(query))
                results.append(result.fetchall() if result.returns_rows else None)
        
        return results
    
    def optimize_connection_pool(self, pool_size: int = 20,
                                 max_overflow: int = 10,
                                 timeout: int = 30) -> None:
        """
        Reconfigure connection pool for optimal performance.
        
        Args:
            pool_size: Base number of connections
            max_overflow: Additional connections under load
            timeout: Wait time for connection
        """
        # Dispose existing pool
        self.engine.dispose()
        
        # Create new engine with optimized pool
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=timeout,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        
        # Re-setup event listeners
        self._setup_query_listeners()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        pool = self.engine.pool
        
        return {
            'size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'total_connections': pool.size() + pool.overflow()
        }
    
    def get_query_stats_summary(self) -> pd.DataFrame:
        """Get summary of query statistics as DataFrame."""
        data = []
        
        for query_hash, stats in self.query_stats.items():
            avg_time = stats['total_time'] / stats['executions']
            data.append({
                'query': stats['query'][:100] + '...' if len(stats['query']) > 100 else stats['query'],
                'executions': stats['executions'],
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': stats['min_time'] * 1000,
                'max_time_ms': stats['max_time'] * 1000,
                'total_time_s': stats['total_time']
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('avg_time_ms', ascending=False) if not df.empty else df
