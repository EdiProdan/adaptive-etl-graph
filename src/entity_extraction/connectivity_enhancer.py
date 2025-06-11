# src/entity_extraction/connectivity_enhancer.py
"""
Fast-Track Connectivity Enhancement
==================================

Alternative implementation that bypasses computational bottlenecks through
strategic semantic hub identification and targeted collection optimization.
Designed for immediate execution with your existing dataset constraints.
"""

import json
import logging
from pathlib import Path
from typing import List, Set

from src.entity_extraction.wikipedia_client import WikipediaClient


class ConnectivityEnhancer:
    """
    Optimized connectivity enhancement focusing on execution efficiency
    Uses predefined semantic targets with validation rather than exhaustive analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.wikipedia_client = WikipediaClient()

        # Load existing page titles for duplicate prevention
        self.existing_page_titles = self._load_existing_page_titles()
        self.logger.info(f"Loaded {len(self.existing_page_titles)} existing page titles")

    def _load_existing_page_titles(self) -> Set[str]:
        """Load existing page titles from base collection"""
        base_pages_file = "data/input/pages/base_pages.json"

        if not Path(base_pages_file).exists():
            self.logger.error(f"Base pages file not found: {base_pages_file}")
            return set()

        with open(base_pages_file, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)

        return {page['title'] for page in pages_data}

    def get_strategic_semantic_hubs(self) -> List[str]:
        """
        Curated list of high-value semantic hubs guaranteed to create cross-domain conflicts
        Based on empirical analysis of Wikipedia connectivity patterns
        """
        # Prioritized by cross-domain connectivity potential
        strategic_hubs = [
            # Geographic hot spots (appear in history, culture, science, politics)
            "Paris", "London", "Rome", "New York City", "Berlin", "Tokyo", "Moscow",

            # Major countries (political, cultural, scientific connections)
            "United States", "France", "Germany", "Italy", "United Kingdom",
            "Japan", "China", "Russia", "India", "Canada",

            # Historical periods (cross-temporal connections)
            "World War II", "World War I", "Renaissance", "Industrial Revolution",
            "Cold War", "French Revolution", "American Revolution",

            # Scientific concepts (interdisciplinary connections)
            "Physics", "Chemistry", "Biology", "Mathematics", "Medicine",
            "Computer science", "Engineering", "Astronomy",

            # Cultural institutions (arts, literature, music connections)
            "University", "Museum", "Library", "Theater", "Opera",
            "Academy of Sciences", "Royal Society",

            # Biographical hot spots (multi-domain figures)
            "Leonardo da Vinci", "Isaac Newton", "Albert Einstein", "Charles Darwin",
            "William Shakespeare", "Wolfgang Amadeus Mozart", "Aristotle", "Plato",

            # Temporal entities (broad historical connections)
            "19th century", "20th century", "Middle Ages", "Ancient Rome",
            "Ancient Greece", "Byzantine Empire", "Ottoman Empire",

            # Conceptual bridges (philosophy, religion, ideology)
            "Christianity", "Philosophy", "Democracy", "Capitalism", "Socialism",
            "Art", "Literature", "Science", "Religion", "Politics"
        ]

        # Filter to entities that exist in our collection
        validated_hubs = []
        for hub in strategic_hubs:
            if hub in self.existing_page_titles:
                validated_hubs.append(hub)
                self.logger.info(f"Validated strategic hub: {hub}")
            else:
                # Check for close matches
                close_matches = [title for title in self.existing_page_titles
                                 if hub.lower() in title.lower() or title.lower() in hub.lower()]
                if close_matches:
                    # Use the closest match
                    best_match = min(close_matches, key=len)
                    validated_hubs.append(best_match)
                    self.logger.info(f"Strategic hub matched: {hub} → {best_match}")

        self.logger.info(f"Strategic semantic hubs identified: {len(validated_hubs)}")
        return validated_hubs

    def collect_referencing_pages_batch(self, semantic_hubs: List[str],
                                        pages_per_hub: int = 150) -> List[str]:
        """
        Efficiently collect pages referencing multiple semantic hubs
        Uses batch processing with progress tracking
        """
        all_candidates = set()
        failed_hubs = []

        total_hubs = len(semantic_hubs)
        self.logger.info(f"Collecting referencing pages for {total_hubs} semantic hubs...")

        for i, hub in enumerate(semantic_hubs, 1):
            self.logger.info(f"Processing hub {i}/{total_hubs}: {hub}")

            try:
                referencing_pages = self._get_backlinks_optimized(hub, max_pages=pages_per_hub)

                # Filter and add to candidates
                valid_pages = [page for page in referencing_pages
                               if self._is_enhancement_candidate(page)]

                before_count = len(all_candidates)
                all_candidates.update(valid_pages)
                after_count = len(all_candidates)
                new_pages = after_count - before_count

                self.logger.info(
                    f"  Found {len(referencing_pages)} total, {len(valid_pages)} valid, {new_pages} new candidates")
                self.logger.info(f"  Total unique candidates: {len(all_candidates)}")

            except Exception as e:
                self.logger.warning(f"Failed to process hub {hub}: {e}")
                failed_hubs.append(hub)
                continue

            # Progress update
            progress_pct = (i / total_hubs) * 100
            self.logger.info(f"Progress: {progress_pct:.1f}% complete")

            # Early termination if we have enough candidates
            if len(all_candidates) >= 3000:  # Collect more than needed for quality selection
                self.logger.info(f"Reached target candidate count, stopping early")
                break

        if failed_hubs:
            self.logger.warning(f"Failed to process {len(failed_hubs)} hubs: {failed_hubs}")

        candidates_list = list(all_candidates)
        self.logger.info(f"Batch collection complete: {len(candidates_list)} unique candidates identified")

        return candidates_list

    def _get_backlinks_optimized(self, target_entity: str, max_pages: int = 150) -> List[str]:
        """
        Optimized backlink collection with error handling and rate limiting
        """
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
        response = self.wikipedia_client.session.get(api_url, params=params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")

        data = response.json()
        backlinks = data.get('query', {}).get('backlinks', [])

        return [link.get('title', '') for link in backlinks if link.get('title')]

    def _is_enhancement_candidate(self, title: str) -> bool:
        """
        Determine if page is suitable for enhancement
        Optimized filtering logic
        """
        if not title or title in self.existing_page_titles:
            return False

        # Quick exclusion patterns
        exclude_patterns = [
            'List of', 'Category:', 'Template:', 'File:', 'Wikipedia:',
            'disambiguation', 'index', 'timeline', 'chronology', 'User:',
            'Talk:', 'Draft:', 'Portal:', 'Help:', 'Book:'
        ]

        title_lower = title.lower()
        return not any(pattern.lower() in title_lower for pattern in exclude_patterns)

    def select_optimal_enhancement_pages(self, candidates: List[str],
                                         target_count: int = 2000) -> List[str]:
        """
        Select optimal pages for enhancement based on diversity and quality heuristics
        """
        self.logger.info(f"Selecting {target_count} optimal pages from {len(candidates)} candidates")

        if len(candidates) <= target_count:
            return candidates

        # Prioritization scoring
        scored_candidates = []

        for candidate in candidates:
            score = self._calculate_enhancement_score(candidate)
            scored_candidates.append((candidate, score))

        # Sort by score (descending) and select top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_pages = [candidate for candidate, score in scored_candidates[:target_count]]

        # Log score distribution
        scores = [score for _, score in scored_candidates[:target_count]]
        avg_score = sum(scores) / len(scores) if scores else 0
        self.logger.info(f"Selected pages with average score: {avg_score:.2f}")

        return selected_pages

    def _calculate_enhancement_score(self, title: str) -> float:
        """
        Calculate enhancement value score for candidate page
        Higher scores indicate better connectivity potential
        """
        score = 1.0  # Base score

        title_lower = title.lower()

        # Boost biographical pages
        if any(indicator in title_lower for indicator in ['born', 'died', 'biography']):
            score += 0.5

        # Boost geographic entities
        if any(geo in title_lower for geo in ['city', 'country', 'state', 'province', 'region']):
            score += 0.3

        # Boost institutional entities
        if any(inst in title_lower for inst in ['university', 'institute', 'academy', 'society']):
            score += 0.3

        # Boost historical entities
        if any(hist in title_lower for hist in ['war', 'battle', 'revolution', 'empire', 'kingdom']):
            score += 0.2

        # Boost cultural entities
        if any(cult in title_lower for cult in ['art', 'music', 'literature', 'culture', 'museum']):
            score += 0.2

        # Penalty for overly specific or technical titles
        if len(title.split()) > 6:
            score -= 0.2

        return max(score, 0.1)  # Minimum score

    def execute_fast_enhancement(self, target_pages: int = 2000) -> bool:
        """
        Execute complete fast-track enhancement workflow
        """
        try:
            self.logger.info("🚀 Fast-Track Connectivity Enhancement Started")
            self.logger.info("=" * 60)

            # Phase 1: Get strategic semantic hubs
            self.logger.info("Phase 1: Identifying strategic semantic hubs...")
            semantic_hubs = self.get_strategic_semantic_hubs()

            if not semantic_hubs:
                self.logger.error("No semantic hubs identified")
                return False

            # Phase 2: Collect referencing pages
            self.logger.info("Phase 2: Collecting referencing pages...")
            pages_per_hub = max(100, (target_pages * 2) // len(semantic_hubs))  # Collect 2x for selection

            candidates = self.collect_referencing_pages_batch(semantic_hubs, pages_per_hub)

            if not candidates:
                self.logger.error("No enhancement candidates found")
                return False

            # Phase 3: Select optimal pages
            self.logger.info("Phase 3: Selecting optimal enhancement pages...")
            selected_pages = self.select_optimal_enhancement_pages(candidates, target_pages)

            # Phase 4: Collect page data
            self.logger.info("Phase 4: Collecting page content...")
            output_file = "data/input/pages/enhancement_pages.json"
            collected_pages = self.wikipedia_client.collect_pages(selected_pages, output_file)

            # Summary
            self.logger.info("=" * 60)
            self.logger.info("✅ Fast-Track Enhancement Complete")
            self.logger.info("=" * 60)
            self.logger.info(f"Semantic hubs processed: {len(semantic_hubs)}")
            self.logger.info(f"Candidates evaluated: {len(candidates)}")
            self.logger.info(f"Pages selected: {len(selected_pages)}")
            self.logger.info(f"Pages collected: {len(collected_pages)}")
            self.logger.info(f"Enhancement data saved to: {output_file}")

            return len(collected_pages) > 0

        except Exception as e:
            self.logger.error(f"Fast-track enhancement failed: {e}")
            return False


def main():
    """Execute fast-track connectivity enhancement"""
    logging.basicConfig(level=logging.INFO)

    enhancer = FastConnectivityEnhancer()
    success = enhancer.execute_fast_enhancement(target_pages=2000)

    if success:
        print("\n🎯 Next Steps:")
        print("1. Run entity extraction on enhancement pages")
        print("2. Merge with existing data")
        print("3. Proceed to Neo4j setup")
    else:
        print("\n❌ Enhancement failed. Check logs for details.")

    return success


if __name__ == "__main__":
    main()
