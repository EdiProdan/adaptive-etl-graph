# scripts/validate_entity_schema.py
"""
Entity Schema Validation Framework for Research Database
======================================================

Comprehensive analysis of actual vs. expected entity data structure
to diagnose the missing 'categories' property issue and validate
data loading pipeline integrity for research validation.

Research Objectives:
1. Analyze actual entity property structure in Neo4j
2. Compare against expected data model specifications
3. Identify data loading pipeline failures or transformations
4. Provide actionable remediation strategies
"""

import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Set
from collections import Counter, defaultdict



project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

from src.database.neo4j_connector import Neo4jConnector


class EntitySchemaValidator:
    """
    Comprehensive entity schema validation for research data integrity

    Validation Framework:
    - Property existence analysis across entity population
    - Data type validation and consistency checking
    - Comparison against expected schema specifications
    - Statistical analysis of property distribution patterns
    """

    def __init__(self):
        self.db = Neo4jConnector()
        self.validation_results = {}

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/schema_validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def execute_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Execute complete entity schema validation suite

        Returns comprehensive analysis for data integrity assessment
        """
        self.logger.info("=" * 70)
        self.logger.info("🔍 ENTITY SCHEMA VALIDATION ANALYSIS")
        self.logger.info("=" * 70)

        validation_suite = [
            ("entity_count_verification", self._verify_entity_counts),
            ("property_existence_analysis", self._analyze_property_existence),
            ("sample_entity_inspection", self._inspect_sample_entities),
            ("property_distribution_analysis", self._analyze_property_distributions),
            ("data_type_validation", self._validate_data_types),
            ("expected_vs_actual_schema", self._compare_schema_expectations),
            ("data_loading_integrity_check", self._check_loading_integrity)
        ]

        for validation_name, validation_function in validation_suite:
            self.logger.info(f"\n🧪 Executing validation: {validation_name}")
            try:
                result = validation_function()
                self.validation_results[validation_name] = {
                    "status": "success",
                    "data": result,
                    "timestamp": self._get_timestamp()
                }
                self.logger.info(f"✅ {validation_name} completed successfully")
            except Exception as e:
                self.logger.error(f"❌ {validation_name} failed: {str(e)}")
                self.validation_results[validation_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": self._get_timestamp()
                }

        # Generate comprehensive analysis
        analysis = self._generate_validation_analysis()
        self._save_validation_results(analysis)

        return analysis

    def _verify_entity_counts(self) -> Dict[str, Any]:
        """Verify basic entity counts and label distribution"""

        # Total entity count
        total_entities = self.db.execute_query("MATCH (n) RETURN count(n) as total")[0]['total']

        # Entity label distribution
        label_distribution = self.db.execute_query("""
            MATCH (n)
            RETURN labels(n) as labels, count(n) as count
            ORDER BY count DESC
        """)

        # Entity nodes specifically
        entity_count = self.db.execute_query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']

        return {
            "total_nodes": total_entities,
            "entity_nodes": entity_count,
            "label_distribution": label_distribution,
            "entity_percentage": (entity_count / total_entities * 100) if total_entities > 0 else 0
        }

    def _analyze_property_existence(self) -> Dict[str, Any]:
        """Analyze which properties actually exist on Entity nodes"""

        # Get all unique property keys across all Entity nodes
        property_analysis = self.db.execute_query("""
            MATCH (e:Entity)
            WITH e, keys(e) as entity_keys
            UNWIND entity_keys as property_key
            RETURN property_key, count(*) as entity_count
            ORDER BY entity_count DESC
        """)

        # Total entity count for percentage calculation
        total_entities = self.db.execute_query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']

        # Enhance with percentage information
        property_stats = []
        for prop in property_analysis:
            property_stats.append({
                "property": prop['property_key'],
                "entity_count": prop['entity_count'],
                "percentage": (prop['entity_count'] / total_entities * 100) if total_entities > 0 else 0
            })

        return {
            "total_entities": total_entities,
            "unique_properties": len(property_analysis),
            "property_statistics": property_stats
        }

    def _inspect_sample_entities(self) -> Dict[str, Any]:
        """Detailed inspection of sample entities to understand actual structure"""

        # Get sample entities with all their properties
        samples = self.db.execute_query("""
            MATCH (e:Entity)
            RETURN e
            LIMIT 10
        """)

        sample_analysis = []
        property_patterns = defaultdict(int)

        for sample in samples:
            entity = sample['e']
            entity_properties = dict(entity.items()) if hasattr(entity, 'items') else {}

            sample_analysis.append({
                "properties": list(entity_properties.keys()),
                "property_count": len(entity_properties),
                "sample_values": {k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                                  for k, v in list(entity_properties.items())[:5]}
            })

            # Track property patterns
            property_pattern = tuple(sorted(entity_properties.keys()))
            property_patterns[property_pattern] += 1

        return {
            "sample_count": len(samples),
            "sample_entities": sample_analysis,
            "property_patterns": dict(property_patterns),
            "most_common_pattern": max(property_patterns.items(), key=lambda x: x[1]) if property_patterns else None
        }

    def _analyze_property_distributions(self) -> Dict[str, Any]:
        """Analyze distribution patterns of key properties"""

        # Check for expected properties individually
        expected_properties = ['title', 'page_id', 'content', 'views', 'url', 'categories', 'links']

        property_analysis = {}

        for prop in expected_properties:
            try:
                # Check existence and sample values
                query = f"""
                    MATCH (e:Entity)
                    WHERE e.{prop} IS NOT NULL
                    RETURN count(e) as with_property,
                           size(collect(DISTINCT e.{prop})[0..5]) as sample_size
                """

                result = self.db.execute_query(query)

                if result:
                    property_analysis[prop] = {
                        "entities_with_property": result[0]['with_property'],
                        "sample_analysis": "available"
                    }
                else:
                    property_analysis[prop] = {
                        "entities_with_property": 0,
                        "sample_analysis": "query_failed"
                    }

            except Exception as e:
                property_analysis[prop] = {
                    "entities_with_property": 0,
                    "error": str(e),
                    "sample_analysis": "property_not_found"
                }

        return property_analysis

    def _validate_data_types(self) -> Dict[str, Any]:
        """Validate data types of existing properties"""

        # Analyze data types for properties that exist
        type_analysis = self.db.execute_query("""
            MATCH (e:Entity)
            WHERE e.title IS NOT NULL
            WITH e.title as title, 
                 e.page_id as page_id,
                 e.views as views
            RETURN 
                apoc.meta.cypher.type(title) as title_type,
                apoc.meta.cypher.type(page_id) as page_id_type,
                apoc.meta.cypher.type(views) as views_type
            LIMIT 1
        """)

        # If APOC not available, use basic validation
        if not type_analysis:
            basic_validation = self.db.execute_query("""
                MATCH (e:Entity)
                RETURN 
                    e.title as title_sample,
                    e.page_id as page_id_sample,
                    e.views as views_sample
                LIMIT 5
            """)

            return {
                "validation_method": "basic_sampling",
                "samples": basic_validation,
                "apoc_available": False
            }

        return {
            "validation_method": "apoc_meta",
            "type_analysis": type_analysis,
            "apoc_available": True
        }

    def _compare_schema_expectations(self) -> Dict[str, Any]:
        """Compare actual schema against expected research data model"""

        # Expected schema based on research requirements
        expected_schema = {
            "required_properties": [
                "title", "page_id", "content", "views", "url"
            ],
            "optional_properties": [
                "categories", "links", "batch_id", "processed_at"
            ],
            "relationship_types": [
                "LINKS_TO", "BELONGS_TO"
            ]
        }

        # Get actual property existence
        actual_properties = self.validation_results.get(
            "property_existence_analysis", {}
        ).get("data", {}).get("property_statistics", [])

        actual_property_names = {prop["property"] for prop in actual_properties}

        # Compare expected vs actual
        missing_required = set(expected_schema["required_properties"]) - actual_property_names
        missing_optional = set(expected_schema["optional_properties"]) - actual_property_names
        unexpected_properties = actual_property_names - set(expected_schema["required_properties"]) - set(
            expected_schema["optional_properties"])

        return {
            "expected_schema": expected_schema,
            "actual_properties": list(actual_property_names),
            "missing_required": list(missing_required),
            "missing_optional": list(missing_optional),
            "unexpected_properties": list(unexpected_properties),
            "schema_compliance": len(missing_required) == 0
        }

    def _check_loading_integrity(self) -> Dict[str, Any]:
        """Check data loading integrity by examining source data format"""

        # Load source data sample for comparison
        source_files = [
            "data/processed/phase1/consolidated_entities.json"
        ]

        source_analysis = {}

        for file_path in source_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        sample_data = json.load(f)

                    if sample_data:
                        sample_entity = sample_data[0]
                        source_analysis[file_path] = {
                            "sample_properties": list(sample_entity.keys()),
                            "total_entities": len(sample_data),
                            "categories_present": "categories" in sample_entity,
                            "categories_sample": sample_entity.get("categories", [])[
                                                 :5] if "categories" in sample_entity else []
                        }

                except Exception as e:
                    source_analysis[file_path] = {"error": str(e)}
            else:
                source_analysis[file_path] = {"status": "file_not_found"}

        return source_analysis

    def _generate_validation_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive validation analysis and recommendations"""

        # Extract key findings
        property_stats = self.validation_results.get("property_existence_analysis", {}).get("data", {})
        schema_comparison = self.validation_results.get("expected_vs_actual_schema", {}).get("data", {})
        loading_integrity = self.validation_results.get("data_loading_integrity_check", {}).get("data", {})

        # Determine root cause
        root_cause_analysis = self._analyze_root_cause(property_stats, schema_comparison, loading_integrity)

        return {
            "validation_summary": {
                "timestamp": self._get_timestamp(),
                "total_validations": len(self.validation_results),
                "successful_validations": sum(
                    1 for v in self.validation_results.values() if v.get("status") == "success")
            },
            "critical_findings": {
                "categories_property_missing": "categories" in schema_comparison.get("missing_optional", []),
                "schema_compliance": schema_comparison.get("schema_compliance", False),
                "data_loading_integrity": self._assess_loading_integrity(loading_integrity)
            },
            "root_cause_analysis": root_cause_analysis,
            "detailed_results": self.validation_results,
            "remediation_recommendations": self._generate_remediation_plan(root_cause_analysis)
        }

    def _analyze_root_cause(self, property_stats: Dict, schema_comparison: Dict, loading_integrity: Dict) -> Dict[
        str, Any]:
        """Analyze root cause of schema validation failures"""

        potential_causes = []

        # Check if categories exist in source data
        source_has_categories = any(
            analysis.get("categories_present", False)
            for analysis in loading_integrity.values()
            if isinstance(analysis, dict) and "categories_present" in analysis
        )

        if source_has_categories:
            potential_causes.append({
                "cause": "data_transformation_failure",
                "description": "Categories exist in source data but missing in database",
                "likelihood": "high",
                "evidence": "Source files contain categories property"
            })
        else:
            potential_causes.append({
                "cause": "source_data_incomplete",
                "description": "Categories missing from source data files",
                "likelihood": "medium",
                "evidence": "Source files lack categories property"
            })

        # Check for data loading pipeline issues
        missing_optional = schema_comparison.get("missing_optional", [])
        if len(missing_optional) > 1:
            potential_causes.append({
                "cause": "systematic_loading_failure",
                "description": "Multiple optional properties missing suggests pipeline issue",
                "likelihood": "medium",
                "evidence": f"Missing properties: {missing_optional}"
            })

        return {
            "primary_hypothesis": potential_causes[0] if potential_causes else None,
            "alternative_hypotheses": potential_causes[1:],
            "confidence_level": self._calculate_confidence(potential_causes)
        }

    def _assess_loading_integrity(self, loading_integrity: Dict) -> str:
        """Assess overall data loading integrity"""

        if not loading_integrity:
            return "unknown"

        successful_analyses = sum(
            1 for analysis in loading_integrity.values()
            if isinstance(analysis, dict) and "error" not in analysis
        )

        total_analyses = len(loading_integrity)

        if successful_analyses == total_analyses:
            return "good"
        elif successful_analyses > total_analyses / 2:
            return "partial"
        else:
            return "poor"

    def _generate_remediation_plan(self, root_cause: Dict) -> List[Dict[str, str]]:
        """Generate actionable remediation recommendations"""

        recommendations = []

        primary_cause = root_cause.get("primary_hypothesis", {})

        if primary_cause.get("cause") == "data_transformation_failure":
            recommendations.append({
                "priority": "high",
                "action": "investigate_loading_pipeline",
                "description": "Examine entity conversion logic in _convert_entities_for_db method",
                "estimated_effort": "30 minutes"
            })

            recommendations.append({
                "priority": "high",
                "action": "verify_batch_loading",
                "description": "Check if categories are being passed to load_batch_with_monitoring",
                "estimated_effort": "15 minutes"
            })

        recommendations.append({
            "priority": "medium",
            "action": "implement_incremental_fix",
            "description": "Add missing properties to existing entities if source data available",
            "estimated_effort": "45 minutes"
        })

        return recommendations

    def _calculate_confidence(self, potential_causes: List) -> str:
        """Calculate confidence level in root cause analysis"""

        if not potential_causes:
            return "low"

        high_likelihood_causes = sum(1 for cause in potential_causes if cause.get("likelihood") == "high")

        if high_likelihood_causes > 0:
            return "high"
        elif len(potential_causes) > 1:
            return "medium"
        else:
            return "low"

    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis tracking"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _save_validation_results(self, analysis: Dict):
        """Save validation results for research documentation"""

        output_file = "data/processed/phase3/entity_schema_validation.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 Schema validation results saved to: {output_file}")

        # Generate summary report
        self._generate_summary_report(analysis)

    def _generate_summary_report(self, analysis: Dict):
        """Generate human-readable summary report"""

        critical_findings = analysis.get("critical_findings", {})
        root_cause = analysis.get("root_cause_analysis", {})
        recommendations = analysis.get("remediation_recommendations", [])

        summary_lines = [
            "=" * 70,
            "🔍 ENTITY SCHEMA VALIDATION SUMMARY",
            "=" * 70,
            "",
            "🎯 CRITICAL FINDINGS:",
            f"   Categories Property Missing: {'❌ Yes' if critical_findings.get('categories_property_missing') else '✅ No'}",
            f"   Schema Compliance: {'✅ Pass' if critical_findings.get('schema_compliance') else '❌ Fail'}",
            f"   Data Loading Integrity: {critical_findings.get('data_loading_integrity', 'unknown').upper()}",
            "",
            "🔍 ROOT CAUSE ANALYSIS:",
        ]

        primary_hypothesis = root_cause.get("primary_hypothesis")
        if primary_hypothesis:
            summary_lines.extend([
                f"   Primary Hypothesis: {primary_hypothesis.get('cause', 'unknown')}",
                f"   Description: {primary_hypothesis.get('description', 'No description')}",
                f"   Likelihood: {primary_hypothesis.get('likelihood', 'unknown').upper()}",
                f"   Confidence Level: {root_cause.get('confidence_level', 'unknown').upper()}",
            ])

        summary_lines.extend([
            "",
            "🛠️  REMEDIATION RECOMMENDATIONS:",
        ])

        for i, rec in enumerate(recommendations[:3], 1):
            summary_lines.extend([
                f"   {i}. {rec.get('action', 'unknown_action').replace('_', ' ').title()}",
                f"      Priority: {rec.get('priority', 'unknown').upper()}",
                f"      Effort: {rec.get('estimated_effort', 'unknown')}",
            ])

        summary_lines.extend([
            "",
            "=" * 70
        ])

        # Log summary
        for line in summary_lines:
            self.logger.info(line)

        # Save summary file
        summary_file = "data/processed/phase3/schema_validation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))


def main():
    """Execute comprehensive entity schema validation"""

    validator = EntitySchemaValidator()

    try:
        analysis = validator.execute_comprehensive_validation()

        print("\n🎯 Schema validation completed!")
        print("📊 Check logs and output files for detailed analysis.")

        # Quick status check
        if analysis.get("critical_findings", {}).get("categories_property_missing"):
            print("⚠️  Categories property missing - remediation required")
            return False
        else:
            print("✅ Schema validation successful")
            return True

    except Exception as e:
        logging.error(f"❌ Schema validation failed: {str(e)}")
        return False

    finally:
        validator.db.close()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)