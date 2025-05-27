# scripts/enhance_connectivity.py
"""
Incremental Connectivity Enhancement Workflow
============================================

Complete workflow for enhancing dataset connectivity without reprocessing existing data.
Implements systematic enhancement, incremental extraction, and data consolidation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from phase1_extraction.connectivity_enhancer import ConnectivityEnhancer
from phase1_extraction.entity_extractor import EntityExtractor


class IncrementalProcessor:
    """
    Orchestrates incremental dataset enhancement and processing
    Maintains data consistency while expanding connectivity
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # File paths
        self.paths = {
            'base_pages': 'data/input/pages/base_pages.json',
            'enhancement_pages': 'data/input/pages/enhancement_pages.json',
            'existing_entities': 'data/processed/phase1/entities.json',
            'existing_relationships': 'data/processed/phase1/relationships.json',
            'enhanced_entities': 'data/processed/phase1/entities_enhanced.json',
            'enhanced_relationships': 'data/processed/phase1/relationships_enhanced.json',
            'final_summary': 'data/processed/phase1/final_extraction_summary.json'
        }

    def validate_existing_data(self) -> bool:
        """Validate that existing processed data is available"""
        required_files = [
            self.paths['base_pages'],
            self.paths['existing_entities'],
            self.paths['existing_relationships']
        ]

        for file_path in required_files:
            if not Path(file_path).exists():
                self.logger.error(f"Required file not found: {file_path}")
                return False

        return True

    def enhance_connectivity(self, target_pages: int = 2000) -> bool:
        """Phase 1: Enhance connectivity through strategic page collection"""
        try:
            self.logger.info("🔗 Phase 1: Connectivity Enhancement")
            self.logger.info("=" * 50)

            # Initialize connectivity enhancer
            enhancer = ConnectivityEnhancer(self.paths['existing_entities'])

            # Identify and collect enhancement candidates
            enhancement_candidates = enhancer.enhance_connectivity(target_additional_pages=target_pages)

            if not enhancement_candidates:
                self.logger.error("No enhancement candidates identified")
                return False

            # Collect enhancement pages
            enhancement_pages = enhancer.collect_enhancement_pages(
                enhancement_candidates,
                self.paths['enhancement_pages']
            )

            self.logger.info(f"✅ Collected {len(enhancement_pages)} enhancement pages")
            return True

        except Exception as e:
            self.logger.error(f"Connectivity enhancement failed: {e}")
            return False

    def process_enhancement_data(self) -> bool:
        """Phase 2: Extract entities from enhancement pages only"""
        try:
            self.logger.info("\n📊 Phase 2: Enhancement Data Processing")
            self.logger.info("=" * 50)

            # Load enhancement pages
            if not Path(self.paths['enhancement_pages']).exists():
                self.logger.error("Enhancement pages not found. Run connectivity enhancement first.")
                return False

            with open(self.paths['enhancement_pages'], 'r', encoding='utf-8') as f:
                enhancement_pages = json.load(f)

            self.logger.info(f"Processing {len(enhancement_pages)} enhancement pages...")

            # Extract entities from enhancement pages only
            extractor = EntityExtractor()
            enhancement_entities, enhancement_relationships = extractor.process_all_pages(enhancement_pages)

            self.logger.info(f"Extracted {len(enhancement_entities)} new entities")
            self.logger.info(f"Extracted {len(enhancement_relationships)} new relationships")

            # Save enhancement-only data temporarily
            temp_entities_file = "data/processed/phase1/temp_enhancement_entities.json"
            temp_relationships_file = "data/processed/phase1/temp_enhancement_relationships.json"

            extractor.save_entities(enhancement_entities, temp_entities_file)
            extractor.save_relationships(enhancement_relationships, temp_relationships_file)

            return True

        except Exception as e:
            self.logger.error(f"Enhancement data processing failed: {e}")
            return False

    def consolidate_datasets(self) -> bool:
        """Phase 3: Merge existing and enhancement data intelligently"""
        try:
            self.logger.info("\n🔄 Phase 3: Dataset Consolidation")
            self.logger.info("=" * 50)

            # Load existing data
            with open(self.paths['existing_entities'], 'r', encoding='utf-8') as f:
                existing_entities = json.load(f)

            with open(self.paths['existing_relationships'], 'r', encoding='utf-8') as f:
                existing_relationships = json.load(f)

            # Load enhancement data
            temp_entities_file = "data/processed/phase1/temp_enhancement_entities.json"
            temp_relationships_file = "data/processed/phase1/temp_enhancement_relationships.json"

            with open(temp_entities_file, 'r', encoding='utf-8') as f:
                enhancement_entities = json.load(f)

            with open(temp_relationships_file, 'r', encoding='utf-8') as f:
                enhancement_relationships = json.load(f)

            # Consolidate entities (merge duplicates intelligently)
            consolidated_entities = self._merge_entities(existing_entities, enhancement_entities)

            # Consolidate relationships (remove duplicates)
            consolidated_relationships = self._merge_relationships(existing_relationships, enhancement_relationships)

            # Save consolidated data
            with open(self.paths['enhanced_entities'], 'w', encoding='utf-8') as f:
                json.dump(consolidated_entities, f, indent=2, ensure_ascii=False)

            with open(self.paths['enhanced_relationships'], 'w', encoding='utf-8') as f:
                json.dump(consolidated_relationships, f, indent=2, ensure_ascii=False)

            # Clean up temporary files
            Path(temp_entities_file).unlink()
            Path(temp_relationships_file).unlink()

            self.logger.info(f"✅ Consolidated entities: {len(consolidated_entities)}")
            self.logger.info(f"✅ Consolidated relationships: {len(consolidated_relationships)}")

            return True

        except Exception as e:
            self.logger.error(f"Dataset consolidation failed: {e}")
            return False

    def _merge_entities(self, existing: List[Dict], enhancement: List[Dict]) -> List[Dict]:
        """Intelligently merge entity lists, handling duplicates"""
        entity_map = {}

        # Add existing entities
        for entity in existing:
            entity_map[entity['name']] = entity

        # Merge enhancement entities
        for entity in enhancement:
            name = entity['name']
            if name in entity_map:
                # Merge aliases and preserve page_id if available
                existing_entity = entity_map[name]
                all_aliases = set(existing_entity.get('aliases', []) + entity.get('aliases', []))
                existing_entity['aliases'] = list(all_aliases)

                # Preserve page_id from either source
                if entity.get('page_id') and not existing_entity.get('page_id'):
                    existing_entity['page_id'] = entity['page_id']
            else:
                entity_map[name] = entity

        return list(entity_map.values())

    def _merge_relationships(self, existing: List[Dict], enhancement: List[Dict]) -> List[Dict]:
        """Merge relationship lists, removing duplicates"""
        relationship_set = set()
        consolidated = []

        # Add existing relationships
        for rel in existing:
            key = (rel['source'], rel['target'], rel['relationship_type'])
            if key not in relationship_set:
                relationship_set.add(key)
                consolidated.append(rel)

        # Add enhancement relationships
        for rel in enhancement:
            key = (rel['source'], rel['target'], rel['relationship_type'])
            if key not in relationship_set:
                relationship_set.add(key)
                consolidated.append(rel)

        return consolidated

    def analyze_enhancement_impact(self) -> Dict:
        """Phase 4: Analyze the impact of connectivity enhancement"""
        try:
            self.logger.info("\n📈 Phase 4: Enhancement Impact Analysis")
            self.logger.info("=" * 50)

            # Load consolidated data
            with open(self.paths['enhanced_entities'], 'r', encoding='utf-8') as f:
                enhanced_entities = json.load(f)

            with open(self.paths['enhanced_relationships'], 'r', encoding='utf-8') as f:
                enhanced_relationships = json.load(f)

            # Load original data for comparison
            with open(self.paths['existing_entities'], 'r', encoding='utf-8') as f:
                original_entities = json.load(f)

            with open(self.paths['existing_relationships'], 'r', encoding='utf-8') as f:
                original_relationships = json.load(f)

            # Calculate improvement metrics
            entity_improvement = len(enhanced_entities) - len(original_entities)
            relationship_improvement = len(enhanced_relationships) - len(original_relationships)

            # Analyze hot spot improvement
            original_hot_spots = self._analyze_hot_spots(original_relationships)
            enhanced_hot_spots = self._analyze_hot_spots(enhanced_relationships)

            # Calculate internal hot spot improvement
            enhanced_internal_hot_spots = [
                (entity, count) for entity, count in enhanced_hot_spots[:50]
                if self._is_internal_entity(entity, enhanced_entities)
            ]

            original_internal_hot_spots = [
                (entity, count) for entity, count in original_hot_spots[:50]
                if self._is_internal_entity(entity, original_entities)
            ]

            analysis = {
                'entity_metrics': {
                    'original_count': len(original_entities),
                    'enhanced_count': len(enhanced_entities),
                    'improvement': entity_improvement,
                    'improvement_percentage': (entity_improvement / len(original_entities)) * 100
                },
                'relationship_metrics': {
                    'original_count': len(original_relationships),
                    'enhanced_count': len(enhanced_relationships),
                    'improvement': relationship_improvement,
                    'improvement_percentage': (relationship_improvement / len(original_relationships)) * 100
                },
                'hot_spot_analysis': {
                    'original_internal_hot_spots': len(original_internal_hot_spots),
                    'enhanced_internal_hot_spots': len(enhanced_internal_hot_spots),
                    'top_enhanced_internal_hot_spots': enhanced_internal_hot_spots[:10]
                },
                'connectivity_assessment': {
                    'avg_references_improvement': len(enhanced_internal_hot_spots) - len(original_internal_hot_spots)
                }
            }

            # Save analysis
            with open(self.paths['final_summary'], 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            return analysis

        except Exception as e:
            self.logger.error(f"Enhancement impact analysis failed: {e}")
            return {}

    def _analyze_hot_spots(self, relationships: List[Dict]) -> List[Tuple[str, int]]:
        """Analyze hot spots from relationship data"""
        from collections import Counter

        target_counts = Counter()
        for rel in relationships:
            target_counts[rel['target']] += 1

        return target_counts.most_common(100)

    def _is_internal_entity(self, entity_name: str, entities: List[Dict]) -> bool:
        """Check if entity is an internal page (not external identifier/category)"""
        for entity in entities:
            if entity['name'] == entity_name:
                return entity.get('entity_type') == 'page'
        return False

    def run_complete_enhancement(self, target_pages: int = 2000) -> bool:
        """Execute complete enhancement workflow"""
        self.logger.info("🚀 Starting Complete Connectivity Enhancement Workflow")
        self.logger.info("=" * 60)

        # Validate prerequisites
        if not self.validate_existing_data():
            self.logger.error("Validation failed. Ensure base data collection is complete.")
            return False

        # Phase 1: Enhance connectivity
        if not self.enhance_connectivity(target_pages):
            return False

        # Phase 2: Process enhancement data
        if not self.process_enhancement_data():
            return False

        # Phase 3: Consolidate datasets
        if not self.consolidate_datasets():
            return False

        # Phase 4: Analyze impact
        analysis = self.analyze_enhancement_impact()

        # Final summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎯 CONNECTIVITY ENHANCEMENT COMPLETE")
        self.logger.info("=" * 60)

        if analysis:
            entity_metrics = analysis['entity_metrics']
            relationship_metrics = analysis['relationship_metrics']
            hot_spot_metrics = analysis['hot_spot_analysis']

            self.logger.info(
                f"Entity Enhancement: {entity_metrics['original_count']:,} → {entity_metrics['enhanced_count']:,} (+{entity_metrics['improvement_percentage']:.1f}%)")
            self.logger.info(
                f"Relationship Enhancement: {relationship_metrics['original_count']:,} → {relationship_metrics['enhanced_count']:,} (+{relationship_metrics['improvement_percentage']:.1f}%)")
            self.logger.info(
                f"Internal Hot Spots: {hot_spot_metrics['original_internal_hot_spots']} → {hot_spot_metrics['enhanced_internal_hot_spots']}")

            self.logger.info("\nTop Enhanced Internal Hot Spots:")
            for i, (entity, count) in enumerate(hot_spot_metrics['top_enhanced_internal_hot_spots'][:5], 1):
                self.logger.info(f"  {i}. {entity}: {count} references")

        self.logger.info(f"\n📁 Enhanced Data Files:")
        self.logger.info(f"  Entities: {self.paths['enhanced_entities']}")
        self.logger.info(f"  Relationships: {self.paths['enhanced_relationships']}")
        self.logger.info(f"  Analysis: {self.paths['final_summary']}")

        self.logger.info("\n🎯 Status: Ready for Neo4j setup with enhanced connectivity!")

        return True


def main():
    """Main execution workflow"""
    logging.basicConfig(level=logging.INFO)

    processor = IncrementalProcessor()
    success = processor.run_complete_enhancement(target_pages=2000)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)