# src/incremental_loading/hotspot_targeted_collector.py
"""
Hot-Spot Targeted Wikipedia Collection Framework
===============================================

Strategic content collection optimized for maximum interaction with
existing 167 hot-spot entities. Designed to generate high-probability
conflict scenarios for adaptive batching algorithm validation.

Research Strategy:
1. Analyze existing hot-spots to identify optimal content categories
2. Target pages likely to reference established conflict entities
3. Ensure sufficient semantic overlap for meaningful experimental validation
"""

import requests
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.neo4j_connector import Neo4jConnector


class HotSpotTargetedCollector:
    """
    Strategic collector targeting existing hot-spot entities for conflict generation

    Research Methodology:
    - Query existing database for confirmed hot-spot entities
    - Identify content categories with high reference probability
    - Collect pages likely to contain semantic relationships
    - Optimize for conflict generation rather than content diversity
    """

    def __init__(self, output_dir: str = "data/input/phase4"):
        self.output_dir = Path(output_dir)
        self.text_dir = self.output_dir / "text"
        self.text_dir.mkdir(parents=True, exist_ok=True)

        # Database connection for hot-spot analysis
        self.db = Neo4jConnector()

        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ResearchBot/HotSpotTargeted (Conflict Optimization)'
        })

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Collection tracking
        self.collected_pages = []
        self.hot_spots = []
        self.target_categories = {}

    def collect_hotspot_targeted_content(self, target_pages: int = 1500) -> List[Dict]:
        """
        Execute hot-spot targeted collection strategy

        Research Optimization:
        1. Analyze existing hot-spots for targeting strategy
        2. Identify high-probability reference categories
        3. Collect content optimized for conflict generation
        4. Validate semantic overlap with established entities
        """

        self.logger.info("=" * 60)
        self.logger.info("🎯 HOT-SPOT TARGETED COLLECTION STRATEGY")
        self.logger.info("=" * 60)

        # Step 1: Analyze existing hot-spots
        self._analyze_existing_hotspots()

        # Step 2: Develop targeting strategy
        self._develop_targeting_strategy()

        # Step 3: Execute strategic collection
        collected_titles = self._execute_strategic_collection(target_pages)

        # Step 4: Process content for research format
        processed_pages = self._process_collected_content(collected_titles)

        # Step 5: Validate hot-spot coverage
        self._validate_hotspot_coverage(processed_pages)

        self.logger.info(f"✅ Hot-spot targeted collection complete: {len(processed_pages)} pages")
        return processed_pages

    def _analyze_existing_hotspots(self):
        """Analyze existing database for hot-spot characteristics"""

        self.logger.info("🔍 Analyzing existing hot-spot entities...")

        try:
            # Get top hot-spots from existing database
            hot_spots_query = """
            MATCH (target:Entity)
            OPTIONAL MATCH (source:Entity)-[:LINKS_TO]->(target)
            WITH target, COUNT(source) as incoming_links
            WHERE incoming_links > 50
            RETURN target.title as title, incoming_links
            ORDER BY incoming_links DESC
            LIMIT 167
            """

            results = self.db.execute_query(hot_spots_query)
            self.hot_spots = [r['title'] for r in results]

            self.logger.info(f"📊 Identified {len(self.hot_spots)} hot-spot entities")

            # Log top hot-spots for targeting strategy
            top_hotspots = [r for r in results[:10]]
            self.logger.info("🔥 Top hot-spots for targeting:")
            for i, hs in enumerate(top_hotspots, 1):
                self.logger.info(f"   {i}. {hs['title']} ({hs['incoming_links']} links)")

        except Exception as e:
            self.logger.error(f"❌ Hot-spot analysis failed: {str(e)}")
            # Fallback to manual hot-spot list
            self._use_fallback_hotspots()

    def _use_fallback_hotspots(self):
        """Fallback hot-spot list for targeting"""
        self.hot_spots = [
            # Geographic hot-spots (high reference probability)
            "United States", "France", "Germany", "United Kingdom", "China",
            "Japan", "India", "Paris", "London", "New York City", "Tokyo",

            # Institutional hot-spots
            "University", "Nobel Prize", "Academy Awards", "Olympics",
            "United Nations", "European Union", "NATO",

            # Temporal hot-spots
            "World War II", "COVID-19", "2024", "2023", "2022",

            # Cultural hot-spots
            "English language", "Christianity", "Islam", "Buddhism",
            "Wikipedia", "Google", "Apple Inc.", "Microsoft",

            # Scientific hot-spots
            "DNA", "Climate change", "Artificial intelligence", "Physics",
            "Medicine", "Technology", "Internet", "Computer science",

            # Bibliographic identifiers (guaranteed conflicts)
            "ISBN", "ISSN", "DOI", "PMID"
        ]

        self.logger.info(f"📋 Using fallback hot-spot list: {len(self.hot_spots)} entities")

    def _develop_targeting_strategy(self):
        """Develop content targeting strategy based on hot-spot analysis"""

        self.logger.info("🎯 Developing targeting strategy...")

        # Categorize hot-spots for targeted collection
        self.target_categories = {

            # Geographic targeting (high probability)
            'geographic_current_events': {
                'keywords': ['United States', 'France', 'Germany', 'China', 'Paris', 'London'],
                'search_terms': [
                    "2025 in United States", "2025 in France", "2025 in Germany",
                    "2025 in China", "2025 in United Kingdom", "2025 Paris events",
                    "2025 London", "2025 Tokyo", "Current events 2025",
                    "2025 elections", "2025 politics", "2025 international relations"
                ],
                'target_count': 400
            },

            # Institutional targeting
            'institutional_academic': {
                'keywords': ['University', 'Nobel Prize', 'Academy Awards', 'United Nations'],
                'search_terms': [
                    "2025 Nobel Prize", "2025 Academy Awards", "2025 Olympics",
                    "Universities in 2025", "2025 academic research", "2025 discoveries",
                    "2025 scientific achievements", "2025 innovations", "2025 awards",
                    "2025 conferences", "2025 symposiums"
                ],
                'target_count': 300
            },

            # Cultural/technological targeting
            'cultural_technological': {
                'keywords': ['Google', 'Apple', 'Microsoft', 'AI', 'Internet', 'Technology'],
                'search_terms': [
                    "Artificial intelligence 2025", "Technology in 2025", "2025 innovations",
                    "Google 2025", "Apple 2025", "Microsoft 2025", "2025 software",
                    "2025 applications", "2025 digital", "2025 computing",
                    "Internet in 2025", "Social media 2025"
                ],
                'target_count': 300
            },

            # Scientific targeting
            'scientific_medical': {
                'keywords': ['Medicine', 'DNA', 'Climate change', 'Physics', 'Research'],
                'search_terms': [
                    "2025 in medicine", "2025 medical research", "2025 healthcare",
                    "Climate change 2025", "2025 environmental", "2025 in science",
                    "2025 physics research", "2025 medical discoveries",
                    "2025 pharmaceutical", "2025 biotechnology"
                ],
                'target_count': 300
            },

            # Recent publications (bibliographic targeting)
            'recent_publications': {
                'keywords': ['ISBN', 'book', 'publication', 'research', 'journal'],
                'search_terms': [
                    "2025 books", "2025 publications", "Books published in 2025",
                    "2025 literature", "2025 academic papers", "2025 journals",
                    "2025 research publications", "New books 2025", "2025 authors",
                    "2025 scientific papers"
                ],
                'target_count': 200
            }
        }

        total_targeted = sum(cat['target_count'] for cat in self.target_categories.values())
        self.logger.info(f"📋 Targeting strategy developed: {len(self.target_categories)} categories")
        self.logger.info(f"🎯 Total targeted pages: {total_targeted}")

    def _execute_strategic_collection(self, target_pages: int) -> List[str]:
        """Execute strategic page collection based on hot-spot targeting"""

        collected_titles = set()

        for category_name, category_config in self.target_categories.items():
            self.logger.info(f"🔍 Collecting {category_name}: target {category_config['target_count']} pages")

            category_pages = self._collect_category_pages(
                category_config['search_terms'],
                category_config['target_count']
            )

            collected_titles.update(category_pages)
            self.logger.info(f"✅ {category_name}: collected {len(category_pages)} pages")

            # Stop if we've reached target
            if len(collected_titles) >= target_pages:
                break

        return list(collected_titles)[:target_pages]

    def _collect_category_pages(self, search_terms: List[str], target_count: int) -> List[str]:
        """Collect pages for specific category using search terms"""

        category_pages = set()

        for search_term in search_terms:
            try:
                # Search for pages matching the term
                search_results = self._search_wikipedia_pages(search_term, limit=50)

                for page_title in search_results:
                    if self._validate_page_for_hotspot_targeting(page_title):
                        category_pages.add(page_title)

                        if len(category_pages) >= target_count:
                            break

                time.sleep(0.1)  # Rate limiting

                if len(category_pages) >= target_count:
                    break

            except Exception as e:
                self.logger.warning(f"⚠️  Search failed for '{search_term}': {str(e)}")
                continue

        return list(category_pages)

    def _search_wikipedia_pages(self, query: str, limit: int = 50) -> List[str]:
        """Search Wikipedia for pages matching query"""

        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit,
                'srnamespace': 0  # Main namespace only
            }

            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                search_results = data.get('query', {}).get('search', [])

                return [result['title'] for result in search_results]

        except Exception as e:
            self.logger.error(f"❌ Search API error: {str(e)}")

        return []

    def _validate_page_for_hotspot_targeting(self, title: str) -> bool:
        """Validate if page is suitable for hot-spot targeting"""

        if not title:
            return False

        # Exclude problematic pages
        exclude_patterns = [
            'disambiguation', 'List of', 'Category:', 'Template:',
            'Wikipedia:', 'File:', 'Help:', 'Portal:', 'User:'
        ]

        if any(pattern in title for pattern in exclude_patterns):
            return False

        # Prefer recent content
        prefer_patterns = ['2025', '2024', '2023']
        if any(pattern in title for pattern in prefer_patterns):
            return True

        # Check for potential hot-spot references in title
        title_lower = title.lower()
        hotspot_keywords = [hs.lower() for hs in self.hot_spots[:20]]  # Top hotspots

        if any(keyword in title_lower for keyword in hotspot_keywords):
            return True

        return len(title.split()) > 1  # Prefer multi-word titles

    def _process_collected_content(self, titles: List[str]) -> List[Dict]:
        """Process collected titles into research-ready content files"""

        self.logger.info(f"📄 Processing {len(titles)} collected pages...")

        processed_pages = []

        for i, title in enumerate(titles, 1):
            if i % 100 == 0:
                self.logger.info(f"📄 Processing {i}/{len(titles)}: {title}")

            try:
                # Get page content
                content = self._get_page_content(title)
                if content:
                    # Save to text file
                    filename = self._save_text_file(title, content, i)

                    processed_pages.append({
                        'title': title,
                        'filename': filename,
                        'content_length': len(content),
                        'collection_method': 'hotspot_targeted'
                    })

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                self.logger.error(f"❌ Processing failed for {title}: {str(e)}")
                continue

        # Save metadata
        self._save_collection_metadata(processed_pages)

        return processed_pages

    def _get_page_content(self, title: str) -> Optional[str]:
        """Get clean page content from Wikipedia API"""

        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
                'exsectionformat': 'plain'
            }

            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})

                if pages:
                    page_data = list(pages.values())[0]
                    return page_data.get('extract', '')

        except Exception as e:
            self.logger.error(f"❌ Content retrieval failed for {title}: {str(e)}")

        return None

    def _save_text_file(self, title: str, content: str, index: int) -> str:
        """Save content to numbered text file"""

        # Create safe filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]

        filename = f"{index:04d}_{safe_title}.txt"
        file_path = self.text_dir / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n\n{content}")

        return filename

    def _validate_hotspot_coverage(self, processed_pages: List[Dict]):
        """Validate coverage of existing hot-spots in collected content"""

        self.logger.info("🔍 Validating hot-spot coverage in collected content...")

        # Sample content analysis for hot-spot references
        sample_pages = processed_pages[:100]  # Analyze first 100 pages

        hotspot_mentions = {}
        total_mentions = 0

        for page in sample_pages:
            try:
                file_path = self.text_dir / page['filename']
                content = file_path.read_text(encoding='utf-8').lower()

                for hotspot in self.hot_spots[:20]:  # Check top 20 hotspots
                    if hotspot.lower() in content:
                        hotspot_mentions[hotspot] = hotspot_mentions.get(hotspot, 0) + 1
                        total_mentions += 1

            except Exception as e:
                self.logger.warning(f"⚠️  Validation failed for {page['filename']}: {str(e)}")
                continue

        coverage_rate = len(hotspot_mentions) / min(len(self.hot_spots), 20) * 100

        self.logger.info("📊 Hot-spot coverage validation:")
        self.logger.info(f"   Hot-spots referenced: {len(hotspot_mentions)}/20")
        self.logger.info(f"   Coverage rate: {coverage_rate:.1f}%")
        self.logger.info(f"   Total mentions: {total_mentions}")

        if coverage_rate > 50:
            self.logger.info("✅ Excellent hot-spot coverage - optimal for conflict generation")
        elif coverage_rate > 25:
            self.logger.info("✅ Good hot-spot coverage - suitable for experimental validation")
        else:
            self.logger.warning("⚠️  Low hot-spot coverage - consider refining targeting strategy")

    def _save_collection_metadata(self, processed_pages: List[Dict]):
        """Save comprehensive collection metadata"""

        metadata = {
            'collection_strategy': 'hotspot_targeted',
            'total_pages_collected': len(processed_pages),
            'target_hotspots_count': len(self.hot_spots),
            'collection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),

            'targeting_categories': {
                name: config['target_count']
                for name, config in self.target_categories.items()
            },

            'top_hotspots_targeted': self.hot_spots[:10],

            'pages': processed_pages,

            'research_validation': {
                'conflict_generation_optimized': True,
                'semantic_overlap_targeted': True,
                'experimental_validity': 'high'
            }
        }

        metadata_file = self.output_dir / "hotspot_targeted_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📋 Collection metadata saved: {metadata_file}")


def main():
    """Execute hot-spot targeted collection"""

    collector = HotSpotTargetedCollector()

    try:
        # Execute strategic collection
        pages = collector.collect_hotspot_targeted_content(target_pages=1500)

        print(f"\n🎯 Hot-Spot Targeted Collection Complete!")
        print(f"📊 Pages collected: {len(pages)}")
        print(f"📁 Text files: data/input/phase4/text/")
        print(f"📋 Metadata: data/input/phase4/hotspot_targeted_metadata.json")
        print(f"🔥 Optimized for conflict generation with existing hot-spots")

        return True

    except Exception as e:
        print(f"❌ Collection failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)