# scripts/diagnose_database_state.py
"""
Neo4j Database State Diagnostic Framework
========================================

Comprehensive analysis tool for validating database state post-loading.
Addresses the critical research question: Why does the database contain
only one node after processing 1M+ entities?

Research Hypothesis Testing:
1. Transaction Commitment Issues
2. Schema Constraint Conflicts
3. Memory Management Failures
4. Batch Processing Atomicity Problems
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))


from src.database.neo4j_connector import Neo4jConnector


class DatabaseDiagnosticFramework:
    """
    Comprehensive database state analysis for research validation

    Diagnostic Capabilities:
    - Node/Relationship Count Verification
    - Schema Constraint Analysis
    - Transaction State Investigation
    - Memory Usage Assessment
    - Data Integrity Validation
    """

    def __init__(self):
        self.db = Neo4jConnector()
        self.diagnostic_results = {}

        # Enhanced logging for diagnostic analysis
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/database_diagnostic.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def execute_comprehensive_diagnostic(self) -> Dict[str, Any]:
        """
        Execute complete database diagnostic suite

        Returns comprehensive analysis for research troubleshooting
        """
        self.logger.info("=" * 70)
        self.logger.info("🔍 NEO4J DATABASE STATE DIAGNOSTIC ANALYSIS")
        self.logger.info("=" * 70)

        diagnostic_suite = [
            ("basic_connectivity", self._test_basic_connectivity),
            ("node_relationship_counts", self._analyze_node_relationship_counts),
            ("schema_analysis", self._analyze_schema_state),
            ("constraint_violations", self._check_constraint_violations),
            ("transaction_analysis", self._analyze_transaction_state),
            ("memory_usage", self._analyze_memory_usage),
            ("data_integrity", self._validate_data_integrity),
            ("batch_tracking", self._analyze_batch_tracking),
            ("performance_metrics", self._collect_performance_metrics)
        ]

        for diagnostic_name, diagnostic_function in diagnostic_suite:
            self.logger.info(f"\n🧪 Executing diagnostic: {diagnostic_name}")
            try:
                start_time = time.time()
                result = diagnostic_function()
                execution_time = time.time() - start_time

                self.diagnostic_results[diagnostic_name] = {
                    "status": "completed",
                    "execution_time": execution_time,
                    "results": result
                }

                self.logger.info(f"✅ {diagnostic_name} completed in {execution_time:.3f}s")

            except Exception as e:
                self.logger.error(f"❌ {diagnostic_name} failed: {str(e)}")
                self.diagnostic_results[diagnostic_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        # Generate comprehensive report
        final_report = self._generate_diagnostic_report()
        self._save_diagnostic_results(final_report)

        return final_report

    def _test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic database connectivity and responsiveness"""
        connectivity_tests = {
            "connection_status": False,
            "response_time": 0,
            "database_version": None,
            "active_database": None
        }

        try:
            start_time = time.time()

            # Basic connectivity test
            result = self.db.execute_query("RETURN 'Connected' as status, datetime() as timestamp")
            connectivity_tests["response_time"] = time.time() - start_time
            connectivity_tests["connection_status"] = True

            # Database information
            db_info = self.db.execute_query("CALL dbms.components() YIELD name, versions, edition")
            if db_info:
                connectivity_tests["database_version"] = db_info[0].get("versions", ["Unknown"])[0]
                connectivity_tests["database_edition"] = db_info[0].get("edition", "Unknown")

            # Active database
            active_db = self.db.execute_query("CALL db.info()")
            if active_db:
                connectivity_tests["active_database"] = active_db[0].get("name", "Unknown")

        except Exception as e:
            connectivity_tests["error"] = str(e)

        return connectivity_tests

    def _analyze_node_relationship_counts(self) -> Dict[str, Any]:
        """
        Critical diagnostic: Analyze actual node and relationship counts
        This addresses the core research question about data persistence
        """
        count_analysis = {
            "total_nodes": 0,
            "total_relationships": 0,
            "node_labels": {},
            "relationship_types": {},
            "detailed_breakdown": {}
        }

        try:
            # Total node count
            total_nodes_result = self.db.execute_query("MATCH (n) RETURN count(n) as total_nodes")
            count_analysis["total_nodes"] = total_nodes_result[0]["total_nodes"] if total_nodes_result else 0

            # Total relationship count
            total_rels_result = self.db.execute_query("MATCH ()-[r]->() RETURN count(r) as total_relationships")
            count_analysis["total_relationships"] = total_rels_result[0][
                "total_relationships"] if total_rels_result else 0

            # Node labels breakdown
            node_labels_result = self.db.execute_query("""
                MATCH (n) 
                RETURN labels(n) as labels, count(n) as count
            """)

            for record in node_labels_result:
                labels = tuple(sorted(record["labels"]))  # Convert to tuple for JSON serialization
                count_analysis["node_labels"][str(labels)] = record["count"]

            # Relationship types breakdown
            rel_types_result = self.db.execute_query("""
                MATCH ()-[r]->() 
                RETURN type(r) as relationship_type, count(r) as count
            """)

            for record in rel_types_result:
                rel_type = record["relationship_type"]
                count_analysis["relationship_types"][rel_type] = record["count"]

            # Detailed entity analysis if entities exist
            if count_analysis["total_nodes"] > 0:
                entity_analysis = self.db.execute_query("""
                    MATCH (e:Entity) 
                    RETURN count(e) as entity_count, 
                           count(e.batch_id) as entities_with_batch_id,
                           count(e.title) as entities_with_title
                    LIMIT 1
                """)

                if entity_analysis:
                    count_analysis["detailed_breakdown"]["entities"] = entity_analysis[0]

                # Category analysis
                category_analysis = self.db.execute_query("""
                    MATCH (c:Category) 
                    RETURN count(c) as category_count
                """)

                if category_analysis:
                    count_analysis["detailed_breakdown"]["categories"] = category_analysis[0]

            # Sample data inspection (first 5 nodes)
            sample_nodes = self.db.execute_query("""
                MATCH (n) 
                RETURN labels(n) as labels, 
                       n.title as title, 
                       n.batch_id as batch_id,
                       id(n) as node_id
                LIMIT 5
            """)

            count_analysis["sample_nodes"] = sample_nodes

        except Exception as e:
            count_analysis["error"] = str(e)

        return count_analysis

    def _analyze_schema_state(self) -> Dict[str, Any]:
        """Analyze database schema and constraints"""
        schema_analysis = {
            "constraints": [],
            "indexes": [],
            "schema_errors": []
        }

        try:
            # Get constraints
            constraints_result = self.db.execute_query("SHOW CONSTRAINTS")
            schema_analysis["constraints"] = constraints_result

            # Get indexes
            indexes_result = self.db.execute_query("SHOW INDEXES")
            schema_analysis["indexes"] = indexes_result

        except Exception as e:
            schema_analysis["error"] = str(e)

        return schema_analysis

    def _check_constraint_violations(self) -> Dict[str, Any]:
        """Check for constraint violations that might prevent data loading"""
        violation_analysis = {
            "duplicate_titles": 0,
            "missing_required_fields": {},
            "constraint_conflicts": []
        }

        try:
            # Check for duplicate entity titles
            duplicate_check = self.db.execute_query("""
                MATCH (e:Entity)
                WITH e.title as title, count(*) as title_count
                WHERE title_count > 1
                RETURN title, title_count
                ORDER BY title_count DESC
                LIMIT 10
            """)

            violation_analysis["duplicate_titles"] = len(duplicate_check)
            violation_analysis["duplicate_examples"] = duplicate_check

            # Check for entities without required fields
            missing_fields_check = self.db.execute_query("""
                MATCH (e:Entity)
                RETURN 
                    count(CASE WHEN e.title IS NULL OR e.title = '' THEN 1 END) as missing_title,
                    count(CASE WHEN e.page_id IS NULL THEN 1 END) as missing_page_id,
                    count(*) as total_entities
            """)

            if missing_fields_check:
                violation_analysis["missing_required_fields"] = missing_fields_check[0]

        except Exception as e:
            violation_analysis["error"] = str(e)

        return violation_analysis

    def _analyze_transaction_state(self) -> Dict[str, Any]:
        """Analyze transaction state and potential isolation issues"""
        transaction_analysis = {
            "active_transactions": [],
            "transaction_history": [],
            "isolation_level": None
        }

        try:
            # Check active transactions
            active_tx = self.db.execute_query("SHOW TRANSACTIONS")
            transaction_analysis["active_transactions"] = active_tx

            # Database configuration related to transactions
            tx_config = self.db.execute_query("""
                CALL dbms.listConfig() 
                YIELD name, value 
                WHERE name CONTAINS 'transaction' OR name CONTAINS 'lock'
                RETURN name, value
            """)

            transaction_analysis["transaction_config"] = tx_config

        except Exception as e:
            transaction_analysis["error"] = str(e)

        return transaction_analysis

    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze database memory usage patterns"""
        memory_analysis = {
            "heap_usage": {},
            "page_cache": {},
            "gc_stats": {}
        }

        try:
            # Memory metrics
            memory_metrics = self.db.execute_query("""
                CALL dbms.queryJmx('java.lang:type=Memory') 
                YIELD attributes 
                RETURN attributes
            """)

            if memory_metrics:
                memory_analysis["heap_usage"] = memory_metrics[0].get("attributes", {})

            # Page cache metrics
            page_cache_metrics = self.db.execute_query("""
                CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Page cache') 
                YIELD attributes 
                RETURN attributes
            """)

            if page_cache_metrics:
                memory_analysis["page_cache"] = page_cache_metrics[0].get("attributes", {})

        except Exception as e:
            memory_analysis["error"] = str(e)

        return memory_analysis

    def _validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and consistency"""
        integrity_analysis = {
            "orphaned_relationships": 0,
            "missing_entities": [],
            "data_consistency": True,
            "sample_validation": {}
        }

        try:
            # Check for orphaned relationships
            orphaned_check = self.db.execute_query("""
                MATCH ()-[r:LINKS_TO]->()
                OPTIONAL MATCH (source:Entity)-[r]->(target:Entity)
                WITH r, source, target
                WHERE source IS NULL OR target IS NULL
                RETURN count(r) as orphaned_count
            """)

            if orphaned_check:
                integrity_analysis["orphaned_relationships"] = orphaned_check[0]["orphaned_count"]

            # Sample relationship validation
            sample_relationships = self.db.execute_query("""
                MATCH (source:Entity)-[r:LINKS_TO]->(target:Entity)
                RETURN source.title as source_title, 
                       target.title as target_title,
                       type(r) as relationship_type
                LIMIT 5
            """)

            integrity_analysis["sample_relationships"] = sample_relationships

            # Check entity-category relationships
            entity_category_check = self.db.execute_query("""
                MATCH (e:Entity)-[r:BELONGS_TO]->(c:Category)
                RETURN count(r) as entity_category_links,
                       count(DISTINCT e) as entities_with_categories,
                       count(DISTINCT c) as categories_with_entities
            """)

            if entity_category_check:
                integrity_analysis["entity_category_stats"] = entity_category_check[0]

        except Exception as e:
            integrity_analysis["error"] = str(e)

        return integrity_analysis

    def _analyze_batch_tracking(self) -> Dict[str, Any]:
        """Analyze batch processing tracking and completeness"""
        batch_analysis = {
            "unique_batch_ids": [],
            "batch_completeness": {},
            "processing_timeline": []
        }

        try:
            # Get unique batch IDs
            batch_ids = self.db.execute_query("""
                MATCH (e:Entity)
                WHERE e.batch_id IS NOT NULL
                RETURN DISTINCT e.batch_id as batch_id, count(e) as entity_count
                ORDER BY e.batch_id
            """)

            batch_analysis["unique_batch_ids"] = batch_ids

            # Batch processing timeline
            timeline = self.db.execute_query("""
                MATCH (e:Entity)
                WHERE e.processed_at IS NOT NULL
                RETURN e.batch_id as batch_id, 
                       min(e.processed_at) as earliest_processing,
                       max(e.processed_at) as latest_processing,
                       count(e) as entities_in_batch
                ORDER BY earliest_processing
            """)

            batch_analysis["processing_timeline"] = timeline

        except Exception as e:
            batch_analysis["error"] = str(e)

        return batch_analysis

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect database performance metrics"""
        performance_metrics = {
            "query_performance": {},
            "database_size": {},
            "connection_stats": {}
        }

        try:
            # Database size information
            size_info = self.db.execute_query("""
                CALL apoc.monitor.store()
            """)

            if size_info:
                performance_metrics["database_size"] = size_info[0]

        except Exception as e:
            # APOC may not be available, try alternative
            try:
                # Alternative size estimation
                node_count = self.db.execute_query("MATCH (n) RETURN count(n) as nodes")[0]["nodes"]
                rel_count = self.db.execute_query("MATCH ()-[r]->() RETURN count(r) as relationships")[0][
                    "relationships"]

                performance_metrics["database_size"] = {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "estimated_size": f"{(node_count + rel_count) * 0.001:.2f} MB"
                }
            except Exception as inner_e:
                performance_metrics["error"] = str(inner_e)

        return performance_metrics

    def _generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        report = {
            "diagnostic_metadata": {
                "execution_timestamp": datetime.now().isoformat(),
                "total_diagnostics_run": len(self.diagnostic_results),
                "successful_diagnostics": len(
                    [r for r in self.diagnostic_results.values() if r["status"] == "completed"]),
                "failed_diagnostics": len([r for r in self.diagnostic_results.values() if r["status"] == "failed"])
            },
            "critical_findings": [],
            "research_implications": {},
            "recommended_actions": [],
            "detailed_results": self.diagnostic_results
        }

        # Analyze critical findings
        node_count = self.diagnostic_results.get("node_relationship_counts", {}).get("results", {}).get("total_nodes",
                                                                                                        0)
        rel_count = self.diagnostic_results.get("node_relationship_counts", {}).get("results", {}).get(
            "total_relationships", 0)

        # Critical finding: Low node count
        if node_count < 1000:
            report["critical_findings"].append({
                "severity": "CRITICAL",
                "issue": "Extremely low node count",
                "description": f"Database contains only {node_count} nodes, expected 1M+",
                "likely_causes": [
                    "Transaction rollback during batch processing",
                    "Memory constraints causing data loss",
                    "Schema constraint violations preventing data insertion",
                    "Incomplete transaction commitment"
                ]
            })

        # Critical finding: No relationships
        if rel_count == 0:
            report["critical_findings"].append({
                "severity": "CRITICAL",
                "issue": "No relationships found",
                "description": f"Database contains {rel_count} relationships, expected 3.5M+",
                "likely_causes": [
                    "Relationship loading phase failure",
                    "Foreign key constraint violations",
                    "Transaction isolation preventing relationship creation"
                ]
            })

        # Research implications
        report["research_implications"] = {
            "baseline_validity": node_count > 100000 and rel_count > 100000,
            "algorithm_testing_feasibility": node_count > 10000,
            "performance_comparison_possible": node_count > 1000,
            "data_recovery_required": node_count < 1000
        }

        # Recommended actions
        if node_count < 1000:
            report["recommended_actions"].extend([
                "IMMEDIATE: Investigate transaction commitment issues",
                "URGENT: Check memory allocation and swap usage",
                "CRITICAL: Validate schema constraints and data format",
                "RECOVERY: Re-run data loading with enhanced monitoring"
            ])

        return report

    def _save_diagnostic_results(self, report: Dict[str, Any]):
        """Save diagnostic results for research documentation"""
        try:
            # Save detailed diagnostic report
            output_file = "data/processed/phase3/database_diagnostic_report.json"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"📊 Diagnostic report saved to: {output_file}")

            # Generate human-readable summary
            self._generate_diagnostic_summary(report)

        except Exception as e:
            self.logger.error(f"❌ Failed to save diagnostic results: {str(e)}")

    def _generate_diagnostic_summary(self, report: Dict[str, Any]):
        """Generate human-readable diagnostic summary"""

        summary_lines = [
            "=" * 70,
            "🔍 NEO4J DATABASE DIAGNOSTIC SUMMARY",
            "=" * 70,
            "",
            "📊 DIAGNOSTIC EXECUTION OVERVIEW:",
            f"   Total Diagnostics: {report['diagnostic_metadata']['total_diagnostics_run']}",
            f"   Successful: {report['diagnostic_metadata']['successful_diagnostics']}",
            f"   Failed: {report['diagnostic_metadata']['failed_diagnostics']}",
            f"   Timestamp: {report['diagnostic_metadata']['execution_timestamp']}",
            ""
        ]

        # Database state summary
        node_count = report.get("detailed_results", {}).get("node_relationship_counts", {}).get("results", {}).get(
            "total_nodes", 0)
        rel_count = report.get("detailed_results", {}).get("node_relationship_counts", {}).get("results", {}).get(
            "total_relationships", 0)

        summary_lines.extend([
            "🗄️  DATABASE STATE ANALYSIS:",
            f"   Total Nodes: {node_count:,}",
            f"   Total Relationships: {rel_count:,}",
            f"   Expected Nodes: ~1,000,000",
            f"   Expected Relationships: ~3,500,000",
            ""
        ])

        # Critical findings
        if report["critical_findings"]:
            summary_lines.append("🚨 CRITICAL FINDINGS:")
            for finding in report["critical_findings"]:
                summary_lines.append(f"   ❌ {finding['issue']}: {finding['description']}")
            summary_lines.append("")

        # Research implications
        implications = report["research_implications"]
        summary_lines.extend([
            "🎓 RESEARCH IMPLICATIONS:",
            f"   Baseline Validity: {'✅ Valid' if implications['baseline_validity'] else '❌ Invalid'}",
            f"   Algorithm Testing: {'✅ Feasible' if implications['algorithm_testing_feasibility'] else '❌ Not Feasible'}",
            f"   Performance Comparison: {'✅ Possible' if implications['performance_comparison_possible'] else '❌ Not Possible'}",
            f"   Data Recovery Required: {'❌ Yes' if implications['data_recovery_required'] else '✅ No'}",
            ""
        ])

        # Recommended actions
        if report["recommended_actions"]:
            summary_lines.append("🔧 RECOMMENDED ACTIONS:")
            for action in report["recommended_actions"]:
                summary_lines.append(f"   • {action}")
            summary_lines.append("")

        summary_lines.extend([
            "=" * 70,
            "📋 DIAGNOSTIC COMPLETE - ANALYSIS READY FOR RESEARCH REVIEW",
            "=" * 70
        ])

        # Log summary
        for line in summary_lines:
            self.logger.info(line)

        # Save summary
        summary_file = "data/processed/phase3/database_diagnostic_summary.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            self.logger.info(f"📄 Diagnostic summary saved to: {summary_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save summary: {str(e)}")


def main():
    """Execute comprehensive database diagnostic analysis"""

    print("🔍 Starting Neo4j Database Diagnostic Analysis")
    print("=" * 50)

    try:
        # Initialize diagnostic framework
        diagnostic_framework = DatabaseDiagnosticFramework()

        # Execute comprehensive diagnostic
        diagnostic_report = diagnostic_framework.execute_comprehensive_diagnostic()

        # Analyze results
        node_count = diagnostic_report.get("detailed_results", {}).get("node_relationship_counts", {}).get("results",
                                                                                                           {}).get(
            "total_nodes", 0)

        if node_count < 1000:
            print(f"\n🚨 CRITICAL ISSUE DETECTED!")
            print(f"   Database contains only {node_count} nodes (expected 1M+)")
            print(f"   This explains the baseline loading failure.")
            print(f"   Immediate data recovery required.")
        else:
            print(f"\n✅ Database state analysis complete")
            print(f"   {node_count:,} nodes found in database")

        # Cleanup
        diagnostic_framework.db.close()

        print(f"\n📊 Comprehensive diagnostic report available:")
        print(f"   JSON Report: data/processed/phase3/database_diagnostic_report.json")
        print(f"   Summary: data/processed/phase3/database_diagnostic_summary.txt")

        return True

    except Exception as e:
        logging.error(f"❌ Diagnostic analysis failed: {str(e)}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)