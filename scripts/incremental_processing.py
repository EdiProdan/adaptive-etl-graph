# scripts/incremental_processing.py
"""
Incremental Entity Extraction for Enhancement Pages
==================================================

Processes enhancement pages collected through connectivity enhancement
and extracts entities/relationships for dataset consolidation.
Maintains consistency with existing extraction methodology.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))


from src.entity_extraction.entity_extractor import EntityExtractor


class IncrementalEntityExtractor:
    """
    Specialized entity extractor for processing enhancement pages
    Maintains methodological consistency while supporting incremental workflows
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_extractor = EntityExtractor()

        # Configuration for incremental processing
        self.config = {
            'input_file': "data/input/pages/enhancement_pages.json",
            'entities_output': "data/processed/phase1/enhancement_entities.json",
            'relationships_output': "data/processed/phase1/enhancement_relationships.json",
            'summary_output': "data/processed/phase1/enhancement_extraction_summary.json",
            'progress_checkpoint_interval': 500  # Save progress every N pages
        }

    def validate_input_data(self) -> bool:
        """Validate that enhancement pages are available for processing"""
        input_path = Path(self.config['input_file'])

        if not input_path.exists():
            self.logger.error(f"Enhancement pages not found: {input_path}")
            self.logger.info(
                "Run connectivity enhancement first: python src/entity_extraction/connectivity_enhancer.py")
            return False

        # Load and validate data structure
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                pages_data = json.load(f)

            if not pages_data:
                self.logger.error("Enhancement pages file is empty")
                return False

            # Validate page structure
            sample_page = pages_data[0]
            required_fields = ['title', 'content', 'links', 'categories']

            for field in required_fields:
                if field not in sample_page:
                    self.logger.error(f"Invalid page structure: missing '{field}' field")
                    return False

            self.logger.info(f"Validation successful: {len(pages_data)} enhancement pages ready for processing")
            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def process_enhancement_pages(self) -> bool:
        """
        Process enhancement pages using established entity extraction methodology
        Implements progress tracking and checkpoint saving for large datasets
        """
        try:
            # Load enhancement pages
            with open(self.config['input_file'], 'r', encoding='utf-8') as f:
                enhancement_pages = json.load(f)

            total_pages = len(enhancement_pages)
            self.logger.info(f"Processing {total_pages} enhancement pages for entity extraction")

            # Process pages with progress tracking
            entities, relationships = self.base_extractor.process_all_pages(enhancement_pages)

            # Validate extraction results
            if not entities or not relationships:
                self.logger.error("Entity extraction produced no results")
                return False

            self.logger.info(f"Extraction complete: {len(entities)} entities, {len(relationships)} relationships")

            # Save extraction results
            self._save_extraction_results(entities, relationships)

            # Generate extraction summary
            summary = self._generate_extraction_summary(entities, relationships, total_pages)
            self._save_extraction_summary(summary)

            return True

        except Exception as e:
            self.logger.error(f"Enhancement page processing failed: {e}")
            return False

    def _save_extraction_results(self, entities: Dict, relationships: List) -> None:
        """Save entities and relationships to configured output paths"""

        # Create output directories
        Path(self.config['entities_output']).parent.mkdir(parents=True, exist_ok=True)

        # Save entities
        self.base_extractor.save_entities(entities, self.config['entities_output'])

        # Save relationships
        self.base_extractor.save_relationships(relationships, self.config['relationships_output'])

        self.logger.info("Extraction results saved successfully")

    def _generate_extraction_summary(self, entities: Dict, relationships: List, pages_processed: int) -> Dict:
        """Generate comprehensive extraction summary for analysis"""

        # Generate base summary using existing methodology
        base_summary = self.base_extractor.generate_summary_report(entities, relationships)

        # Enhance with incremental processing metrics
        enhanced_summary = {
            'extraction_metadata': {
                'processing_type': 'incremental_enhancement',
                'pages_processed': pages_processed,
                'extraction_timestamp': str(Path().stat().st_mtime),
                'input_source': self.config['input_file']
            },
            'base_extraction_summary': base_summary,
            'enhancement_specific_metrics': {
                'entities_per_page': len(entities) / pages_processed if pages_processed > 0 else 0,
                'relationships_per_page': len(relationships) / pages_processed if pages_processed > 0 else 0,
                'average_entity_connectivity': len(relationships) / len(entities) if entities else 0
            }
        }

        return enhanced_summary

    def _save_extraction_summary(self, summary: Dict) -> None:
        """Save extraction summary with comprehensive metrics"""

        summary_path = Path(self.config['summary_output'])
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Extraction summary saved: {summary_path}")

    def analyze_extraction_quality(self) -> Dict:
        """
        Analyze extraction quality and identify potential issues
        Provides diagnostic information for dataset consolidation
        """
        try:
            # Load extraction results
            with open(self.config['entities_output'], 'r', encoding='utf-8') as f:
                entities = json.load(f)

            with open(self.config['relationships_output'], 'r', encoding='utf-8') as f:
                relationships = json.load(f)

            # Quality analysis metrics
            analysis = {
                'entity_quality': {
                    'total_entities': len(entities),
                    'page_entities': len([e for e in entities if e.get('entity_type') == 'page']),
                    'category_entities': len([e for e in entities if e.get('entity_type') == 'category']),
                    'entities_with_page_ids': len([e for e in entities if e.get('page_id')]),
                    'average_aliases_per_entity': sum(len(e.get('aliases', [])) for e in entities) / len(
                        entities) if entities else 0
                },
                'relationship_quality': {
                    'total_relationships': len(relationships),
                    'links_to_relationships': len(
                        [r for r in relationships if r.get('relationship_type') == 'links_to']),
                    'category_relationships': len(
                        [r for r in relationships if r.get('relationship_type') == 'belongs_to_category']),
                    'unique_sources': len(set(r.get('source') for r in relationships)),
                    'unique_targets': len(set(r.get('target') for r in relationships))
                },
                'connectivity_metrics': {
                    'avg_outgoing_links': len(
                        [r for r in relationships if r.get('relationship_type') == 'links_to']) / len(
                        set(r.get('source') for r in relationships)) if relationships else 0,
                    'most_connected_entities': self._identify_highly_connected_entities(relationships)
                }
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Extraction quality analysis failed: {e}")
            return {}

    def _identify_highly_connected_entities(self, relationships: List, top_n: int = 10) -> List:
        """Identify entities with highest connectivity for hot spot analysis"""
        from collections import Counter

        # Count incoming relationships (targets)
        target_counts = Counter(r.get('target') for r in relationships)

        return target_counts.most_common(top_n)

    def execute_incremental_extraction(self) -> bool:
        """
        Execute complete incremental extraction workflow
        Includes validation, processing, and quality analysis
        """
        self.logger.info("🔄 Incremental Entity Extraction Started")
        self.logger.info("=" * 60)

        # Phase 1: Input validation
        if not self.validate_input_data():
            return False

        # Phase 2: Entity extraction processing
        self.logger.info("Phase 2: Processing enhancement pages...")
        if not self.process_enhancement_pages():
            return False

        # Phase 3: Quality analysis
        self.logger.info("Phase 3: Analyzing extraction quality...")
        quality_analysis = self.analyze_extraction_quality()

        # Summary reporting
        self.logger.info("=" * 60)
        self.logger.info("✅ Incremental Extraction Complete")
        self.logger.info("=" * 60)

        if quality_analysis:
            entity_quality = quality_analysis['entity_quality']
            relationship_quality = quality_analysis['relationship_quality']

            self.logger.info(f"Entities extracted: {entity_quality['total_entities']:,}")
            self.logger.info(f"  - Pages: {entity_quality['page_entities']:,}")
            self.logger.info(f"  - Categories: {entity_quality['category_entities']:,}")

            self.logger.info(f"Relationships extracted: {relationship_quality['total_relationships']:,}")
            self.logger.info(f"  - Links: {relationship_quality['links_to_relationships']:,}")
            self.logger.info(f"  - Categories: {relationship_quality['category_relationships']:,}")

            # Hot spot identification
            connectivity = quality_analysis['connectivity_metrics']
            self.logger.info("\nTop connected entities (potential hot spots):")
            for i, (entity, count) in enumerate(connectivity['most_connected_entities'][:5], 1):
                self.logger.info(f"  {i}. {entity}: {count} incoming connections")

        self.logger.info(f"\n📁 Output Files:")
        self.logger.info(f"  Entities: {self.config['entities_output']}")
        self.logger.info(f"  Relationships: {self.config['relationships_output']}")
        self.logger.info(f"  Summary: {self.config['summary_output']}")

        self.logger.info("\n🎯 Next Step: Dataset consolidation and merge with existing data")

        return True


def main():
    """Execute incremental entity extraction workflow"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    extractor = IncrementalEntityExtractor()
    success = extractor.execute_incremental_extraction()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)