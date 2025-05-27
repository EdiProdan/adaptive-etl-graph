# src/phase1_extraction/connectivity_enhancer.py
"""
Connectivity Enhancement Module
==============================

Strategically enhance internal connectivity of existing Wikipedia dataset
by identifying and collecting pages that heavily reference our current collection.
Implements incremental enhancement without reprocessing existing data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import requests
import time

from .wikipedia_client import WikipediaClient, WikiPage


class ConnectivityEnhancer:
    """
    Enhance dataset connectivity through strategic page collection
    Focus on creating semantic hot spots rather than metadata hot spots
    """

    def __init__(self, existing_entities_file: str):
        self.logger = logging.getLogger(__name__)
        self.wikipedia_client = WikipediaClient()

        # Load existing entities to identify targets for enhancement
        self.existing_entities = self._load_existing_entities(existing_entities_file)
        self.existing_page_titles = {entity['original_title'] for entity in self.existing_entities
                                     if entity['entity_type'] == 'page'}

        self.logger.info(f"Loaded {len(self.existing_entities)} existing entities")
        self.logger.info(f"Identified {len(self.existing_page_titles)} existing page titles")

    def _load_existing_entities(self, file_path: str) -> List[Dict]:
        """Load existing entities from extraction output"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def identify_semantic_hubs(self, min_references: int = 10) -> List[str]:
        """
        Identify semantic hubs from existing entities that should have high connectivity
        Focus on entities that represent cross-domain concepts
        """
        # Load relationship data to find most referenced internal entities
        relationships_file = Path("data/processed/phase1/relationships.json")
        if not relationships_file.exists():
            self.logger.warning("Relationships file not found, using predefined semantic hubs")
            return self._get_predefined_semantic_hubs()

        with open(relationships_file, 'r', encoding='utf-8') as f:
            relationships = json.load(f)

        # Count internal references (target entities that exist in our page collection)
        internal_references = Counter()
        for rel in relationships:
            target = rel['target']
            # Check if target is in our existing page collection
            if target in {entity['name'] for entity in self.existing_entities
                          if entity['entity_type'] == 'page'}:
                internal_references[target] += 1
                print(f"Found internal reference to {target} (count: {internal_references[target]})")

        # Filter for semantic hubs (high-value entities for expansion)
        semantic_hubs = []
        for entity, count in internal_references.most_common(100):
            if count >= min_references and self._is_semantic_hub(entity):
                semantic_hubs.append(entity)

        self.logger.info(f"Identified {len(semantic_hubs)} semantic hubs for enhancement")
        return semantic_hubs[:20]  # Focus on top 20 for manageable scope

    def _is_semantic_hub(self, entity_name: str) -> bool:
        """
        Determine if entity represents a valuable semantic hub for cross-domain connectivity
        """
        # Geographic entities (cities, countries)
        geographic_patterns = ['city', 'country', 'state', 'province', 'region']

        # Temporal concepts
        temporal_patterns = ['century', 'year', 'period', 'era', 'age']

        # Disciplinary concepts
        disciplinary_patterns = ['science', 'physics', 'chemistry', 'biology', 'history',
                                 'art', 'literature', 'music', 'philosophy', 'mathematics']

        # Institutional entities
        institutional_patterns = ['university', 'institute', 'academy', 'society', 'foundation']

        entity_lower = entity_name.lower()

        # Check for known high-value entities
        high_value_entities = {
            'paris', 'london', 'rome', 'new york', 'berlin', 'tokyo', 'moscow',
            'united states', 'france', 'germany', 'italy', 'united kingdom', 'japan',
            'einstein', 'newton', 'darwin', 'shakespeare', 'mozart', 'leonardo da vinci',
            'world war ii', 'renaissance', 'industrial revolution',
            'physics', 'chemistry', 'biology', 'mathematics', 'history', 'literature'
        }

        return (entity_lower in high_value_entities or
                any(pattern in entity_lower for pattern in
                    geographic_patterns + temporal_patterns + disciplinary_patterns + institutional_patterns))

    def _get_predefined_semantic_hubs(self) -> List[str]:
        """Fallback list of known semantic hubs if relationship analysis fails"""
        return [
            "Paris", "London", "Rome", "New York City", "Berlin",
            "United States", "France", "Germany", "Italy", "United Kingdom",
            "Albert Einstein", "Isaac Newton", "Charles Darwin", "William Shakespeare",
            "World War II", "Renaissance", "Industrial Revolution",
            "Physics", "Chemistry", "Biology", "Mathematics", "History"
        ]

    def find_referencing_pages(self, target_entity: str, max_pages: int = 150) -> List[str]:
        """
        Find Wikipedia pages that reference a specific entity
        Uses Wikipedia's "What links here" functionality
        """
        try:
            # Use Wikipedia API to find pages that link to target
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'backlinks',
                'bltitle': target_entity,
                'bllimit': max_pages,
                'blnamespace': 0,  # Main namespace only
                'blfilterredir': 'nonredirects'
            }

            self.wikipedia_client._rate_limit()
            response = self.wikipedia_client.session.get(api_url, params=params)

            if response.status_code != 200:
                self.logger.warning(f"Failed to get backlinks for {target_entity}")
                return []

            data = response.json()
            backlinks = data.get('query', {}).get('backlinks', [])

            # Extract page titles and filter
            referencing_pages = []
            for link in backlinks:
                title = link.get('title', '')
                if self._is_suitable_enhancement_page(title):
                    referencing_pages.append(title)

            return referencing_pages

        except Exception as e:
            self.logger.error(f"Error finding referencing pages for {target_entity}: {e}")
            return []

    def _is_suitable_enhancement_page(self, title: str) -> bool:
        """
        Determine if a page is suitable for connectivity enhancement
        Avoid duplicate collection and low-quality pages
        """
        # Skip if already collected
        if title in self.existing_page_titles:
            return False

        # Use same filtering logic as original client
        exclude_patterns = [
            'List of', 'Category:', 'Template:', 'File:', 'Wikipedia:',
            'disambiguation', 'index', 'timeline', 'chronology'
        ]

        return not any(pattern in title for pattern in exclude_patterns)

    def enhance_connectivity(self, target_additional_pages: int = 2000) -> List[str]:
        """
        Main enhancement function: collect additional pages to improve internal connectivity
        """
        self.logger.info(f"Starting connectivity enhancement targeting {target_additional_pages} additional pages")

        # 1. Identify semantic hubs that need better connectivity
        semantic_hubs = self.identify_semantic_hubs()
        self.logger.info(f"Targeting {len(semantic_hubs)} semantic hubs for enhancement")

        # 2. For each semantic hub, find pages that reference it
        enhancement_candidates = set()
        pages_per_hub = target_additional_pages // len(semantic_hubs) if semantic_hubs else 100

        for hub in semantic_hubs:
            self.logger.info(f"Finding referencing pages for semantic hub: {hub}")
            referencing_pages = self.find_referencing_pages(hub, max_pages=pages_per_hub + 50)

            # Add to candidates (set automatically handles duplicates)
            for page in referencing_pages[:pages_per_hub]:
                enhancement_candidates.add(page)

            self.logger.info(f"Found {len(referencing_pages)} candidates for {hub}")

            if len(enhancement_candidates) >= target_additional_pages:
                break

        enhancement_pages = list(enhancement_candidates)[:target_additional_pages]

        self.logger.info(f"Selected {len(enhancement_pages)} pages for connectivity enhancement")
        return enhancement_pages

    def collect_enhancement_pages(self, page_titles: List[str], output_file: str) -> List[WikiPage]:
        """
        Collect the enhancement pages using existing Wikipedia client
        """
        self.logger.info(f"Collecting {len(page_titles)} enhancement pages...")

        # Use existing Wikipedia client for consistency
        collected_pages = self.wikipedia_client.collect_pages(page_titles, output_file)

        self.logger.info(f"Successfully collected {len(collected_pages)} enhancement pages")
        return collected_pages


def main():
    """
    Main enhancement workflow
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configuration
    existing_entities_file = "data/processed/phase1/entities.json"
    enhancement_output = "data/input/pages/enhancement_pages.json"
    target_pages = 2000

    # Initialize enhancer
    enhancer = ConnectivityEnhancer(existing_entities_file)

    # 1. Identify enhancement candidates
    logger.info("Phase 1: Identifying connectivity enhancement candidates...")
    enhancement_candidates = enhancer.enhance_connectivity(target_additional_pages=target_pages)

    # 2. Collect enhancement pages
    logger.info("Phase 2: Collecting enhancement pages...")
    enhancement_pages = enhancer.collect_enhancement_pages(enhancement_candidates, enhancement_output)

    # 3. Summary
    logger.info("=" * 60)
    logger.info("🔗 CONNECTIVITY ENHANCEMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Enhancement pages collected: {len(enhancement_pages)}")
    logger.info(f"Total dataset size: {8019 + len(enhancement_pages)} pages")
    logger.info(f"Enhancement data saved to: {enhancement_output}")

    logger.info("\n🎯 Next Steps:")
    logger.info("1. Run incremental entity extraction on enhancement pages")
    logger.info("2. Merge with existing entities and relationships")
    logger.info("3. Analyze improved hot spot distribution")
    logger.info("4. Proceed to Neo4j setup with enhanced dataset")


if __name__ == "__main__":
    main()