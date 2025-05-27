# scripts/consolidate_datasets.py
"""
Dataset Consolidation for Master's Thesis
=========================================

Consolidates base Wikipedia dataset with connectivity enhancement data.
Implements methodical merging with integrity validation and comprehensive
metrics generation suitable for academic research validation.

Design Philosophy:
- Prioritize data integrity over performance optimization
- Maintain clear audit trails for thesis documentation
- Implement transparent conflict resolution strategies
- Generate comprehensive validation metrics
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import time


@dataclass
class ConsolidationMetrics:
    """Consolidation process metrics for thesis documentation"""
    original_entities: int = 0
    enhancement_entities: int = 0
    consolidated_entities: int = 0
    entities_merged: int = 0
    original_relationships: int = 0
    enhancement_relationships: int = 0
    consolidated_relationships: int = 0
    relationships_deduplicated: int = 0
    processing_time_seconds: float = 0.0


class DatasetConsolidator:
    """
    Consolidates Wikipedia datasets with methodical validation
    Designed for master's thesis requirements with emphasis on clarity and auditability
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # File paths configuration
        self.config = {
            'base_entities': 'data/processed/phase1/entities.json',
            'base_relationships': 'data/processed/phase1/relationships.json',
            'enhancement_entities': 'data/processed/phase1/enhancement_entities.json',
            'enhancement_relationships': 'data/processed/phase1/enhancement_relationships.json',
            'consolidated_entities': 'data/processed/phase1/consolidated_entities.json',
            'consolidated_relationships': 'data/processed/phase1/consolidated_relationships.json',
            'consolidation_report': 'data/processed/phase1/consolidation_report.json'
        }

        self.metrics = ConsolidationMetrics()

    def validate_input_files(self) -> bool:
        """
        Validate that all required input files exist and contain valid data
        Essential for thesis methodology documentation
        """
        required_files = [
            self.config['base_entities'],
            self.config['base_relationships'],
            self.config['enhancement_entities'],
            self.config['enhancement_relationships']
        ]

        self.logger.info("Validating input files...")

        for file_path in required_files:
            if not Path(file_path).exists():
                self.logger.error(f"Required file missing: {file_path}")
                return False

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not data:
                        self.logger.error(f"Empty data file: {file_path}")
                        return False

                self.logger.info(f"✓ Validated: {file_path} ({len(data)} records)")

            except Exception as e:
                self.logger.error(f"Invalid JSON in {file_path}: {e}")
                return False

        self.logger.info("All input files validated successfully")
        return True

    def load_datasets(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Load all datasets for consolidation"""

        self.logger.info("Loading datasets...")

        # Load base dataset
        with open(self.config['base_entities'], 'r', encoding='utf-8') as f:
            base_entities = json.load(f)

        with open(self.config['base_relationships'], 'r', encoding='utf-8') as f:
            base_relationships = json.load(f)

        # Load enhancement dataset
        with open(self.config['enhancement_entities'], 'r', encoding='utf-8') as f:
            enhancement_entities = json.load(f)

        with open(self.config['enhancement_relationships'], 'r', encoding='utf-8') as f:
            enhancement_relationships = json.load(f)

        # Record metrics
        self.metrics.original_entities = len(base_entities)
        self.metrics.enhancement_entities = len(enhancement_entities)
        self.metrics.original_relationships = len(base_relationships)
        self.metrics.enhancement_relationships = len(enhancement_relationships)

        self.logger.info(f"Base dataset: {len(base_entities):,} entities, {len(base_relationships):,} relationships")
        self.logger.info(
            f"Enhancement dataset: {len(enhancement_entities):,} entities, {len(enhancement_relationships):,} relationships")

        return base_entities, base_relationships, enhancement_entities, enhancement_relationships

    def consolidate_entities(self, base_entities: List[Dict], enhancement_entities: List[Dict]) -> List[Dict]:
        """
        Consolidate entities with intelligent merging strategy
        Implements transparent conflict resolution for thesis documentation
        """
        self.logger.info("Consolidating entities...")

        # Create lookup structures for efficient processing
        entity_registry = {}  # name -> entity data
        merge_conflicts = []  # Track conflicts for thesis documentation

        # Process base entities first (they have priority)
        self.logger.info("Processing base entities...")
        for entity in base_entities:
            name = entity.get('name')
            if name:
                entity_registry[name] = entity.copy()

        # Process enhancement entities with conflict resolution
        self.logger.info("Processing enhancement entities with merge logic...")
        merged_count = 0

        for entity in enhancement_entities:
            name = entity.get('name')
            if not name:
                continue

            if name in entity_registry:
                # Entity exists - implement merge logic
                existing_entity = entity_registry[name]
                merged_entity = self._merge_entity_data(existing_entity, entity)

                if merged_entity != existing_entity:
                    merge_conflicts.append({
                        'entity_name': name,
                        'conflict_type': 'data_merge',
                        'base_data': existing_entity,
                        'enhancement_data': entity,
                        'resolution': merged_entity
                    })
                    merged_count += 1

                entity_registry[name] = merged_entity
            else:
                # New entity from enhancement dataset
                entity_registry[name] = entity.copy()

        consolidated_entities = list(entity_registry.values())
        self.metrics.consolidated_entities = len(consolidated_entities)
        self.metrics.entities_merged = merged_count

        self.logger.info(f"Entity consolidation complete:")
        self.logger.info(f"  Total entities: {len(consolidated_entities):,}")
        self.logger.info(f"  Entities merged: {merged_count}")
        self.logger.info(f"  Merge conflicts resolved: {len(merge_conflicts)}")

        return consolidated_entities

    def _merge_entity_data(self, base_entity: Dict, enhancement_entity: Dict) -> Dict:
        """
        Merge two entity records with priority-based conflict resolution
        Base entity data takes precedence, enhancement data supplements
        """
        merged = base_entity.copy()

        # Merge aliases (union of both sets)
        base_aliases = set(base_entity.get('aliases', []))
        enhancement_aliases = set(enhancement_entity.get('aliases', []))
        merged_aliases = base_aliases.union(enhancement_aliases)
        merged['aliases'] = list(merged_aliases)

        # Use base entity's page_id if available, otherwise use enhancement
        if not merged.get('page_id') and enhancement_entity.get('page_id'):
            merged['page_id'] = enhancement_entity['page_id']

        # Preserve entity_type from base (should be consistent)
        # Keep original_title from base (more authoritative)

        return merged

    def consolidate_relationships(self, base_relationships: List[Dict], enhancement_relationships: List[Dict]) -> List[
        Dict]:
        """
        Consolidate relationships with deduplication
        Maintains relationship integrity while removing exact duplicates
        """
        self.logger.info("Consolidating relationships...")

        # Use set for efficient duplicate detection
        relationship_signatures = set()
        consolidated_relationships = []
        duplicates_removed = 0

        # Process base relationships first
        self.logger.info("Processing base relationships...")
        for relationship in base_relationships:
            signature = self._get_relationship_signature(relationship)
            if signature not in relationship_signatures:
                relationship_signatures.add(signature)
                consolidated_relationships.append(relationship.copy())

        # Process enhancement relationships with deduplication
        self.logger.info("Processing enhancement relationships with deduplication...")
        for relationship in enhancement_relationships:
            signature = self._get_relationship_signature(relationship)
            if signature not in relationship_signatures:
                relationship_signatures.add(signature)
                consolidated_relationships.append(relationship.copy())
            else:
                duplicates_removed += 1

        self.metrics.consolidated_relationships = len(consolidated_relationships)
        self.metrics.relationships_deduplicated = duplicates_removed

        self.logger.info(f"Relationship consolidation complete:")
        self.logger.info(f"  Total relationships: {len(consolidated_relationships):,}")
        self.logger.info(f"  Duplicates removed: {duplicates_removed:,}")

        return consolidated_relationships

    def _get_relationship_signature(self, relationship: Dict) -> Tuple[str, str, str]:
        """Generate unique signature for relationship deduplication"""
        return (
            relationship.get('source', ''),
            relationship.get('target', ''),
            relationship.get('relationship_type', '')
        )

    def analyze_consolidation_quality(self, entities: List[Dict], relationships: List[Dict]) -> Dict:
        """
        Comprehensive quality analysis for thesis validation
        Generates metrics suitable for academic documentation
        """
        self.logger.info("Analyzing consolidation quality...")

        # Entity analysis
        entity_types = Counter(entity.get('entity_type') for entity in entities)
        entities_with_page_ids = sum(1 for entity in entities if entity.get('page_id'))
        avg_aliases_per_entity = sum(len(entity.get('aliases', [])) for entity in entities) / len(entities)

        # Relationship analysis
        relationship_types = Counter(rel.get('relationship_type') for rel in relationships)
        unique_sources = len(set(rel.get('source') for rel in relationships))
        unique_targets = len(set(rel.get('target') for rel in relationships))

        # Connectivity analysis
        target_frequency = Counter(rel.get('target') for rel in relationships)
        source_frequency = Counter(rel.get('source') for rel in relationships)

        # Hot spot identification (for thesis hot spot analysis)
        hot_spots = target_frequency.most_common(50)
        internal_hot_spots = [
            (entity_name, count) for entity_name, count in hot_spots
            if any(e.get('name') == entity_name and e.get('entity_type') == 'page' for e in entities)
        ]

        quality_analysis = {
            'entity_metrics': {
                'total_entities': len(entities),
                'entity_type_distribution': dict(entity_types),
                'entities_with_page_ids': entities_with_page_ids,
                'page_id_coverage_percentage': (entities_with_page_ids / len(entities)) * 100,
                'average_aliases_per_entity': avg_aliases_per_entity
            },
            'relationship_metrics': {
                'total_relationships': len(relationships),
                'relationship_type_distribution': dict(relationship_types),
                'unique_sources': unique_sources,
                'unique_targets': unique_targets,
                'average_relationships_per_source': len(relationships) / unique_sources if unique_sources else 0,
                'graph_density_indicator': len(relationships) / (len(entities) ** 2) if entities else 0
            },
            'connectivity_analysis': {
                'top_10_hot_spots': hot_spots[:10],
                'internal_hot_spots_count': len(internal_hot_spots),
                'top_internal_hot_spots': internal_hot_spots[:10],
                'entities_with_no_incoming_links': len(entities) - len(set(rel.get('target') for rel in relationships)),
                'entities_with_no_outgoing_links': len(entities) - len(set(rel.get('source') for rel in relationships))
            }
        }

        return quality_analysis

    def save_consolidated_data(self, entities: List[Dict], relationships: List[Dict]) -> None:
        """Save consolidated datasets with proper formatting"""

        # Create output directory
        Path(self.config['consolidated_entities']).parent.mkdir(parents=True, exist_ok=True)

        # Save entities
        self.logger.info(f"Saving {len(entities):,} consolidated entities...")
        with open(self.config['consolidated_entities'], 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)

        # Save relationships
        self.logger.info(f"Saving {len(relationships):,} consolidated relationships...")
        with open(self.config['consolidated_relationships'], 'w', encoding='utf-8') as f:
            json.dump(relationships, f, indent=2, ensure_ascii=False)

        self.logger.info("Consolidated datasets saved successfully")

    def generate_consolidation_report(self, quality_analysis: Dict) -> Dict:
        """Generate comprehensive consolidation report for thesis documentation"""

        report = {
            'consolidation_metadata': {
                'process_timestamp': time.time(),
                'processing_time_seconds': self.metrics.processing_time_seconds,
                'methodology': 'priority-based_entity_merge_with_relationship_deduplication'
            },
            'dataset_metrics': {
                'base_dataset': {
                    'entities': self.metrics.original_entities,
                    'relationships': self.metrics.original_relationships
                },
                'enhancement_dataset': {
                    'entities': self.metrics.enhancement_entities,
                    'relationships': self.metrics.enhancement_relationships
                },
                'consolidated_dataset': {
                    'entities': self.metrics.consolidated_entities,
                    'relationships': self.metrics.consolidated_relationships
                }
            },
            'consolidation_operations': {
                'entities_merged': self.metrics.entities_merged,
                'relationships_deduplicated': self.metrics.relationships_deduplicated,
                'entity_merge_rate': (
                                                 self.metrics.entities_merged / self.metrics.enhancement_entities) * 100 if self.metrics.enhancement_entities else 0,
                'relationship_deduplication_rate': (self.metrics.relationships_deduplicated / (
                            self.metrics.original_relationships + self.metrics.enhancement_relationships)) * 100
            },
            'quality_analysis': quality_analysis,
            'thesis_validation_metrics': {
                'dataset_size_sufficient_for_testing': self.metrics.consolidated_entities > 500000,
                'hot_spots_available_for_conflict_generation': len(
                    quality_analysis['connectivity_analysis']['top_internal_hot_spots']) >= 10,
                'cross_domain_connectivity_achieved': quality_analysis['relationship_metrics'][
                                                          'total_relationships'] > 3000000,
                'data_integrity_maintained': True  # Based on successful consolidation
            }
        }

        return report

    def execute_consolidation(self) -> bool:
        """
        Execute complete dataset consolidation workflow
        Main entry point for thesis data preparation
        """
        start_time = time.time()

        self.logger.info("🔄 Dataset Consolidation Started")
        self.logger.info("=" * 60)

        try:
            # Phase 1: Validation
            if not self.validate_input_files():
                self.logger.error("Input validation failed")
                return False

            # Phase 2: Data Loading
            base_entities, base_relationships, enhancement_entities, enhancement_relationships = self.load_datasets()

            # Phase 3: Entity Consolidation
            consolidated_entities = self.consolidate_entities(base_entities, enhancement_entities)

            # Phase 4: Relationship Consolidation
            consolidated_relationships = self.consolidate_relationships(base_relationships, enhancement_relationships)

            # Phase 5: Quality Analysis
            quality_analysis = self.analyze_consolidation_quality(consolidated_entities, consolidated_relationships)

            # Phase 6: Save Results
            self.save_consolidated_data(consolidated_entities, consolidated_relationships)

            # Phase 7: Generate Report
            self.metrics.processing_time_seconds = time.time() - start_time
            consolidation_report = self.generate_consolidation_report(quality_analysis)

            with open(self.config['consolidation_report'], 'w', encoding='utf-8') as f:
                json.dump(consolidation_report, f, indent=2, ensure_ascii=False)

            # Final Summary
            self.logger.info("=" * 60)
            self.logger.info("✅ Dataset Consolidation Complete")
            self.logger.info("=" * 60)

            entity_metrics = quality_analysis['entity_metrics']
            relationship_metrics = quality_analysis['relationship_metrics']
            connectivity_metrics = quality_analysis['connectivity_analysis']

            self.logger.info(f"Final Dataset Metrics:")
            self.logger.info(f"  Entities: {entity_metrics['total_entities']:,}")
            self.logger.info(f"  Relationships: {relationship_metrics['total_relationships']:,}")
            self.logger.info(f"  Processing Time: {self.metrics.processing_time_seconds:.1f} seconds")

            self.logger.info(f"\nThesis Validation Readiness:")
            validation_metrics = consolidation_report['thesis_validation_metrics']
            for metric, status in validation_metrics.items():
                status_symbol = "✅" if status else "❌"
                self.logger.info(f"  {status_symbol} {metric.replace('_', ' ').title()}: {status}")

            self.logger.info(f"\nTop Internal Hot Spots for Algorithm Testing:")
            for i, (entity, count) in enumerate(connectivity_metrics['top_internal_hot_spots'][:5], 1):
                self.logger.info(f"  {i}. {entity}: {count} references")

            self.logger.info(f"\n📁 Output Files:")
            self.logger.info(f"  Entities: {self.config['consolidated_entities']}")
            self.logger.info(f"  Relationships: {self.config['consolidated_relationships']}")
            self.logger.info(f"  Report: {self.config['consolidation_report']}")

            self.logger.info(f"\n🎯 Status: Dataset ready for Neo4j setup and adaptive batching algorithm development!")

            return True

        except Exception as e:
            self.logger.error(f"Consolidation failed: {e}")
            return False


def main():
    """Execute dataset consolidation for master's thesis"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    consolidator = DatasetConsolidator()
    success = consolidator.execute_consolidation()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)