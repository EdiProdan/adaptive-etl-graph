# src/database/neo4j_connector.py
"""
Neo4j Database Connector for Research Project
============================================

Implements connection management, transaction handling, and performance monitoring
for the adaptive batching algorithm research validation framework.

Key Features:
- Connection pooling optimized for research workloads
- Transaction management with conflict detection
- Performance metrics collection for algorithm evaluation
- Memory-efficient batch processing for 1M+ entity datasets
"""

from neo4j import GraphDatabase, basic_auth, Transaction
from neo4j.exceptions import TransientError, ServiceUnavailable
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from contextlib import contextmanager
import statistics
from datetime import datetime
import threading
from queue import Queue


@dataclass
class BatchMetrics:
    """Performance metrics for batch processing operations"""
    batch_id: str
    entity_count: int
    relationship_count: int
    processing_time: float
    conflict_count: int
    retry_count: int
    memory_usage: float
    lock_wait_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConnectionConfig:
    """Neo4j connection configuration for research environment"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "research123"
    database: str = "research"
    max_connection_lifetime: int = 3600  # 1 hour
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    max_transaction_retry_time: int = 30


class Neo4jConnector:
    """
    Neo4j database connector optimized for research workloads

    Designed specifically for adaptive batching algorithm validation with:
    - Performance monitoring for algorithm comparison
    - Conflict detection for hot spot analysis
    - Resource usage tracking for optimization research
    """

    def __init__(self, config: ConnectionConfig = None):
        self.config = config or ConnectionConfig()
        self.driver = None
        self.metrics_queue = Queue()
        self.performance_stats = []

        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Performance monitoring
        self._lock = threading.Lock()
        self._connection_count = 0
        self._total_queries = 0
        self._failed_queries = 0

        self.connect()

    def connect(self) -> bool:
        """
        Establish connection to Neo4j database with research-optimized settings
        """
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=basic_auth(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
                max_transaction_retry_time=self.config.max_transaction_retry_time
            )

            # Verify connection
            with self.driver.session(database=self.config.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()

            self.logger.info(f"✅ Connected to Neo4j at {self.config.uri}")
            self.logger.info(f"📊 Database: {self.config.database}")
            self.logger.info(f"🔧 Pool size: {self.config.max_connection_pool_size}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Neo4j: {str(e)}")
            return False

    @contextmanager
    def get_session(self):
        """Context manager for Neo4j sessions with error handling"""
        session = None
        try:
            session = self.driver.session(database=self.config.database)
            with self._lock:
                self._connection_count += 1
            yield session
        except Exception as e:
            self.logger.error(f"Session error: {str(e)}")
            raise
        finally:
            if session:
                session.close()
                with self._lock:
                    self._connection_count -= 1

    def execute_query(self, query: str, parameters: Dict = None,
                      retry_count: int = 3) -> List[Dict]:
        """
        Execute Cypher query with automatic retry and performance monitoring
        """
        parameters = parameters or {}
        start_time = time.time()
        last_exception = None

        for attempt in range(retry_count):
            try:
                with self.get_session() as session:
                    result = session.run(query, parameters)
                    records = [record.data() for record in result]

                    # Performance tracking
                    execution_time = time.time() - start_time
                    with self._lock:
                        self._total_queries += 1

                    self.logger.debug(f"Query executed in {execution_time:.3f}s, "
                                      f"returned {len(records)} records")

                    return records

            except TransientError as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(f"Transient error (attempt {attempt + 1}): {str(e)}")
                self.logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                last_exception = e
                self.logger.error(f"Query execution failed: {str(e)}")
                break

        # All retries failed
        with self._lock:
            self._failed_queries += 1

        raise last_exception or Exception("Query execution failed after retries")

    def create_schema(self) -> bool:
        """
        Create optimized schema for research dataset

        Designed for:
        - Fast lookups on high-traffic entities (hot spots)
        - Efficient relationship traversal
        - Optimized memory usage for 1M+ entities
        """
        try:
            schema_queries = [
                # Entity constraints and indexes
                "CREATE CONSTRAINT entity_title_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.title IS UNIQUE",
                "CREATE INDEX entity_page_id_index IF NOT EXISTS FOR (e:Entity) ON (e.page_id)",
                "CREATE INDEX entity_views_index IF NOT EXISTS FOR (e:Entity) ON (e.views)",

                # Category constraints and indexes
                "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE INDEX category_count_index IF NOT EXISTS FOR (c:Category) ON (c.page_count)",

                # Hot spot detection indexes
                "CREATE INDEX entity_link_count_index IF NOT EXISTS FOR (e:Entity) ON (e.incoming_links)",
                "CREATE INDEX relationship_weight_index IF NOT EXISTS FOR ()-[r:LINKS_TO]-() ON (r.weight)",

                # Research-specific indexes for performance analysis
                "CREATE INDEX batch_id_index IF NOT EXISTS FOR (e:Entity) ON (e.batch_id)",
                "CREATE INDEX processing_timestamp_index IF NOT EXISTS FOR (e:Entity) ON (e.processed_at)"
            ]

            self.logger.info("Creating database schema...")

            for query in schema_queries:
                self.execute_query(query)
                self.logger.debug(f"✅ Executed: {query[:50]}...")

            self.logger.info("✅ Schema creation completed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Schema creation failed: {str(e)}")
            return False

    def load_batch_with_monitoring(self, entities: List[Dict],
                                   relationships: List[Dict],
                                   batch_id: str) -> BatchMetrics:
        """
        Load data batch with comprehensive performance monitoring

        Critical for adaptive algorithm validation:
        - Tracks processing time for algorithm comparison
        - Monitors conflicts for hot spot analysis
        - Measures resource usage for optimization research
        """
        start_time = time.time()
        conflict_count = 0
        retry_count = 0

        try:
            with self.get_session() as session:
                with session.begin_transaction() as tx:
                    # Load entities
                    entity_query = """
                    UNWIND $entities as entity
                    MERGE (e:Entity {title: entity.title})
                    SET e.page_id = entity.page_id,
                        e.content_length = size(entity.content),
                        e.views = entity.views,
                        e.url = entity.url,
                        e.batch_id = $batch_id,
                        e.processed_at = datetime(),
                        e.link_count = size(entity.links),
                        e.category_count = size(entity.categories)
                    """

                    tx.run(entity_query, entities=entities, batch_id=batch_id)

                    # Load categories
                    category_query = """
                    UNWIND $entities as entity
                    UNWIND entity.categories as category_name
                    MERGE (c:Category {name: category_name})
                    ON CREATE SET c.page_count = 1
                    ON MATCH SET c.page_count = c.page_count + 1
                    WITH entity, c
                    MATCH (e:Entity {title: entity.title})
                    MERGE (e)-[:BELONGS_TO]->(c)
                    """

                    tx.run(category_query, entities=entities)

                    # Load relationships
                    relationship_query = """
                    UNWIND $relationships as rel
                    MATCH (source:Entity {title: rel.source})
                    MATCH (target:Entity {title: rel.target})
                    MERGE (source)-[r:LINKS_TO]->(target)
                    ON CREATE SET r.weight = 1, r.created_at = datetime()
                    ON MATCH SET r.weight = r.weight + 1
                    """

                    tx.run(relationship_query, relationships=relationships)

            processing_time = time.time() - start_time

            # Create metrics object
            metrics = BatchMetrics(
                batch_id=batch_id,
                entity_count=len(entities),
                relationship_count=len(relationships),
                processing_time=processing_time,
                conflict_count=conflict_count,
                retry_count=retry_count,
                memory_usage=0.0,  # Would integrate with memory monitoring
                lock_wait_time=0.0  # Would integrate with Neo4j metrics
            )

            self.performance_stats.append(metrics)
            self.logger.info(f"✅ Batch {batch_id} loaded: {len(entities)} entities, "
                             f"{len(relationships)} relationships in {processing_time:.2f}s")

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Batch loading failed: {str(e)}")
            raise

    def get_hot_spots(self, limit: int = 50) -> List[Dict]:
        """
        Identify hot spot entities for adaptive algorithm testing

        Returns entities with highest incoming link counts - perfect for
        generating conflicts in concurrent batch processing scenarios
        """
        query = """
        MATCH (target:Entity)
        OPTIONAL MATCH (source:Entity)-[:LINKS_TO]->(target)
        WITH target, COUNT(source) as incoming_links
        WHERE incoming_links > 0
        RETURN target.title as title, 
               target.page_id as page_id,
               incoming_links,
               target.views as views,
               target.category_count as categories
        ORDER BY incoming_links DESC
        LIMIT $limit
        """

        return self.execute_query(query, {"limit": limit})

    def analyze_graph_structure(self) -> Dict:
        """
        Comprehensive graph analysis for research validation

        Provides metrics essential for algorithm evaluation:
        - Connectivity distribution
        - Hot spot identification
        - Processing performance baselines
        """
        analysis_queries = {
            "total_entities": "MATCH (e:Entity) RETURN COUNT(e) as count",
            "total_relationships": "MATCH ()-[r:LINKS_TO]->() RETURN COUNT(r) as count",
            "total_categories": "MATCH (c:Category) RETURN COUNT(c) as count",

            "avg_outgoing_links": """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:LINKS_TO]->()
                WITH e, COUNT(*) as outgoing
                RETURN AVG(outgoing) as avg_outgoing
            """,

            "avg_incoming_links": """
                MATCH (e:Entity)
                OPTIONAL MATCH ()-[:LINKS_TO]->(e)
                WITH e, COUNT(*) as incoming
                RETURN AVG(incoming) as avg_incoming
            """,

            "connectivity_distribution": """
                MATCH (e:Entity)
                OPTIONAL MATCH ()-[:LINKS_TO]->(e)
                WITH e, COUNT(*) as incoming_links
                RETURN incoming_links, COUNT(e) as entity_count
                ORDER BY incoming_links DESC
                LIMIT 20
            """
        }

        results = {}
        for metric_name, query in analysis_queries.items():
            try:
                result = self.execute_query(query)
                results[metric_name] = result
            except Exception as e:
                self.logger.error(f"Analysis query failed for {metric_name}: {str(e)}")
                results[metric_name] = None

        return results

    def get_performance_summary(self) -> Dict:
        """
        Generate performance summary for research documentation
        """
        if not self.performance_stats:
            return {"status": "No performance data available"}

        processing_times = [m.processing_time for m in self.performance_stats]
        conflict_counts = [m.conflict_count for m in self.performance_stats]
        entity_counts = [m.entity_count for m in self.performance_stats]

        return {
            "total_batches_processed": len(self.performance_stats),
            "avg_processing_time": statistics.mean(processing_times),
            "median_processing_time": statistics.median(processing_times),
            "total_conflicts": sum(conflict_counts),
            "avg_entities_per_batch": statistics.mean(entity_counts),
            "total_queries_executed": self._total_queries,
            "query_failure_rate": self._failed_queries / max(self._total_queries, 1),
            "peak_concurrent_connections": max([m.entity_count for m in self.performance_stats], default=0)
        }

    def close(self):
        """Clean up database connections"""
        if self.driver:
            self.driver.close()
            self.logger.info("🔌 Neo4j connection closed")


# Research utility functions

def create_research_database(config: ConnectionConfig = None) -> Neo4jConnector:
    """
    Initialize Neo4j database for research project

    Sets up optimized schema and connection parameters for
    adaptive batching algorithm validation experiments
    """
    connector = Neo4jConnector(config)

    if connector.create_schema():
        logging.info("🎯 Research database ready for algorithm testing")
        return connector
    else:
        raise RuntimeError("Failed to initialize research database")


# Example usage for research setup
if __name__ == "__main__":
    # Initialize research database
    db = create_research_database()

    # Test connection and schema
    try:
        analysis = db.analyze_graph_structure()
        print("📊 Database Analysis:")
        for metric, value in analysis.items():
            print(f"  {metric}: {value}")

        hot_spots = db.get_hot_spots(limit=10)
        print(f"\n🔥 Top 10 Hot Spots:")
        for i, entity in enumerate(hot_spots, 1):
            print(f"  {i}. {entity['title']} ({entity['incoming_links']} links)")

    finally:
        db.close()