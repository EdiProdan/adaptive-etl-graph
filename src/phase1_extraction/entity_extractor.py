# src/phase1_extraction/entity_extractor.py
"""
Entity Extraction and Cleaning for Wikipedia Data
=================================================

This module processes your collected Wikipedia pages to extract clean entities
and relationships for the knowledge graph. It's designed to handle the massive
connectivity you found (318 links/page!) and prepare hot spots for conflict detection.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import unicodedata


@dataclass
class Entity:
    """Clean entity representation"""
    name: str
    original_title: str
    entity_type: str  # 'page', 'category', 'concept'
    aliases: List[str]
    page_id: Optional[int] = None


@dataclass
class Relationship:
    """Relationship between entities"""
    source: str
    target: str
    relationship_type: str  # 'links_to', 'belongs_to_category', 'mentions'
    weight: int = 1


class EntityExtractor:
    """
    Extract and clean entities from Wikipedia data
    Handles your massive link network (756K unique links!)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Keep track of all entities we've seen
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []

        # Statistics for analysis
        self.stats = {
            'pages_processed': 0,
            'entities_created': 0,
            'relationships_created': 0,
            'hot_spots_identified': 0
        }

    def clean_entity_name(self, name: str) -> str:
        """
        Clean and normalize entity names for consistent graph storage
        Handles special characters, unicode, and Wikipedia formatting
        """
        if not name:
            return ""

        # Remove Wikipedia markup and formatting
        cleaned = name.strip()

        # Handle Unicode normalization
        cleaned = unicodedata.normalize('NFKC', cleaned)

        # Remove extra whitespace and line breaks
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Remove common Wikipedia prefixes/suffixes
        prefixes_to_remove = ['File:', 'Category:', 'Template:', 'Wikipedia:', 'Help:']
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]

        # Handle disambiguation pages
        if cleaned.endswith(' (disambiguation)'):
            cleaned = cleaned[:-16]  # Remove ' (disambiguation)'

        # Remove year ranges for people (but keep the info)
        # e.g., "Einstein (1879-1955)" -> "Einstein"
        cleaned = re.sub(r'\s*\(\d{4}[-–]\d{4}\)', '', cleaned)

        return cleaned.strip()

    def extract_page_entities(self, page_data: Dict) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from a single Wikipedia page
        Returns entities found and relationships created
        """
        page_title = page_data.get('title', '')
        page_id = page_data.get('page_id', 0)

        entities = []
        relationships = []

        # 1. Create entity for the page itself
        main_entity = Entity(
            name=self.clean_entity_name(page_title),
            original_title=page_title,
            entity_type='page',
            aliases=[page_title],
            page_id=page_id
        )
        entities.append(main_entity)

        # 2. Extract linked entities (your 318 avg links per page!)
        links = page_data.get('links', [])
        for link in links:
            if not link:
                continue

            cleaned_link = self.clean_entity_name(link)
            if not cleaned_link:
                continue

            # Create entity for linked page
            linked_entity = Entity(
                name=cleaned_link,
                original_title=link,
                entity_type='page',
                aliases=[link]
            )
            entities.append(linked_entity)

            # Create relationship
            relationship = Relationship(
                source=main_entity.name,
                target=linked_entity.name,
                relationship_type='links_to'
            )
            relationships.append(relationship)

        # 3. Extract category entities (your 31.7 avg per page!)
        categories = page_data.get('categories', [])
        for category in categories:
            if not category:
                continue

            cleaned_category = self.clean_entity_name(category)
            if not cleaned_category:
                continue

            # Skip meta categories that don't add semantic value
            if self._is_meta_category(cleaned_category):
                continue

            # Create category entity
            category_entity = Entity(
                name=cleaned_category,
                original_title=category,
                entity_type='category',
                aliases=[category]
            )
            entities.append(category_entity)

            # Create relationship
            relationship = Relationship(
                source=main_entity.name,
                target=category_entity.name,
                relationship_type='belongs_to_category'
            )
            relationships.append(relationship)

        return entities, relationships

    def _is_meta_category(self, category: str) -> bool:
        """Filter out Wikipedia meta categories that don't add semantic value"""
        meta_patterns = [
            'Articles with short description',
            'Short description is different from Wikidata',
            'Articles with hCards',
            'Webarchive template',
            'All articles with unsourced statements',
            'Commons category link from Wikidata',
            'Wikipedia articles written in',
            'All articles with dead external links',
            'Use mdy dates',
            'Use dmy dates',
            'Articles with',
            'Pages with',
            'CS1 ',
            'Wikipedia ',
            'Template '
        ]

        return any(pattern in category for pattern in meta_patterns)

    def process_all_pages(self, pages_data: List[Dict]) -> Tuple[Dict[str, Entity], List[Relationship]]:
        """
        Process all pages and create unified entity/relationship collections
        Handles deduplication and entity merging
        """
        self.logger.info(f"Processing {len(pages_data)} pages for entity extraction...")

        all_entities = {}
        all_relationships = []
        entity_frequency = Counter()  # Track hot spots

        for i, page_data in enumerate(pages_data, 1):
            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(pages_data)} pages...")

            # Extract entities and relationships from this page
            page_entities, page_relationships = self.extract_page_entities(page_data)

            # Add entities to collection (with deduplication)
            for entity in page_entities:
                if entity.name in all_entities:
                    # Merge with existing entity
                    existing = all_entities[entity.name]
                    existing.aliases.extend(entity.aliases)
                    existing.aliases = list(set(existing.aliases))  # Remove duplicates

                    # Keep page_id if this entity has one
                    if entity.page_id and not existing.page_id:
                        existing.page_id = entity.page_id
                else:
                    all_entities[entity.name] = entity

                # Track frequency for hot spot analysis
                entity_frequency[entity.name] += 1

            # Add relationships
            all_relationships.extend(page_relationships)

            self.stats['pages_processed'] += 1

        # Identify hot spots (entities referenced many times)
        hot_spots = entity_frequency.most_common(50)
        self.stats['hot_spots_identified'] = len([name for name, count in hot_spots if count >= 10])

        self.logger.info(f"Hot spots identified: {self.stats['hot_spots_identified']} entities with 10+ references")
        self.logger.info("Top 10 hot spots:")
        for name, count in hot_spots[:10]:
            self.logger.info(f"  {name}: {count} references")

        self.stats['entities_created'] = len(all_entities)
        self.stats['relationships_created'] = len(all_relationships)

        return all_entities, all_relationships

    def save_entities(self, entities: Dict[str, Entity], output_file: str):
        """Save entities to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        entities_data = []
        for entity in entities.values():
            entities_data.append({
                'name': entity.name,
                'original_title': entity.original_title,
                'entity_type': entity.entity_type,
                'aliases': entity.aliases,
                'page_id': entity.page_id
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(entities_data)} entities to {output_file}")

    def save_relationships(self, relationships: List[Relationship], output_file: str):
        """Save relationships to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        relationships_data = []
        for rel in relationships:
            relationships_data.append({
                'source': rel.source,
                'target': rel.target,
                'relationship_type': rel.relationship_type,
                'weight': rel.weight
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relationships_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(relationships_data)} relationships to {output_file}")

    def generate_summary_report(self, entities: Dict[str, Entity], relationships: List[Relationship]) -> Dict:
        """Generate processing summary for analysis"""

        # Entity type distribution
        entity_types = Counter(entity.entity_type for entity in entities.values())

        # Relationship type distribution
        relationship_types = Counter(rel.relationship_type for rel in relationships)

        # Most connected entities (by outgoing links)
        outgoing_connections = Counter()
        for rel in relationships:
            outgoing_connections[rel.source] += 1

        # Most referenced entities (by incoming links)
        incoming_connections = Counter()
        for rel in relationships:
            incoming_connections[rel.target] += 1

        summary = {
            'processing_stats': self.stats,
            'entity_distribution': dict(entity_types),
            'relationship_distribution': dict(relationship_types),
            'top_outgoing_entities': outgoing_connections.most_common(20),
            'top_incoming_entities': incoming_connections.most_common(20),
            'potential_hot_spots': incoming_connections.most_common(50)
        }

        return summary


def main():
    """Process collected Wikipedia data to extract entities and relationships"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input and output paths
    # input_file = "data/input/pages/base_pages.json"
    # entities_output = "data/processed/phase1/entities.json"
    # relationships_output = "data/processed/phase1/relationships.json"
    # summary_output = "data/processed/phase1/extraction_summary.json"

    # Modified file paths for incremental processing
    input_file = "data/input/pages/enhancement_pages.json"  # Changed from base_pages.json
    entities_output = "data/processed/phase1/enhancement_entities.json"  # New incremental output
    relationships_output = "data/processed/phase1/enhancement_relationships.json"  # New incremental output
    summary_output = "data/processed/phase1/enhancement_extraction_summary.json"  # New summary file

    # Load collected pages
    logger.info(f"Loading pages from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        pages_data = json.load(f)

    logger.info(f"Loaded {len(pages_data)} pages")

    # Initialize extractor
    extractor = EntityExtractor()

    # Process all pages
    entities, relationships = extractor.process_all_pages(pages_data)

    # Save results
    extractor.save_entities(entities, entities_output)
    extractor.save_relationships(relationships, relationships_output)

    # Generate and save summary
    summary = extractor.generate_summary_report(entities, relationships)
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final summary
    logger.info("=" * 60)
    logger.info("🎯 ENTITY EXTRACTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Pages processed: {summary['processing_stats']['pages_processed']}")
    logger.info(f"Entities created: {summary['processing_stats']['entities_created']:,}")
    logger.info(f"Relationships created: {summary['processing_stats']['relationships_created']:,}")
    logger.info(f"Hot spots identified: {summary['processing_stats']['hot_spots_identified']}")

    logger.info("\nEntity types:")
    for entity_type, count in summary['entity_distribution'].items():
        logger.info(f"  {entity_type}: {count:,}")

    logger.info("\nTop 5 most referenced entities (hot spots!):")
    for i, (entity, count) in enumerate(summary['top_incoming_entities'][:5], 1):
        logger.info(f"  {i}. {entity}: {count} incoming links")

    logger.info(f"\nFiles saved:")
    logger.info(f"  Entities: {entities_output}")
    logger.info(f"  Relationships: {relationships_output}")
    logger.info(f"  Summary: {summary_output}")

    logger.info("\n🚀 Ready for Neo4j setup and base graph loading!")


if __name__ == "__main__":
    main()