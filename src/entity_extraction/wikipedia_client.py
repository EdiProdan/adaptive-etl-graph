# src/entity_extraction/wikipedia_client.py
import sys

import requests
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WikiPage:
    """Container for Wikipedia page data"""
    title: str
    page_id: int
    content: str
    links: List[str]
    categories: List[str]
    views: int
    url: str
    extract: str  # Short summary


class WikipediaClient:
    """
    Wikipedia API client for collecting pages and metadata
    Focus on popular, well-linked pages for stable data
    """

    def __init__(self, base_url: str = "https://en.wikipedia.org/api/rest_v1"):
        self.base_url = base_url
        self.wikimedia_url = "https://wikimedia.org/api/rest_v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ResearchBot/1.0 (Educational Research Project)'
        })

        # Rate limiting
        self.last_request = 0
        self.min_delay = 0.1  # 100ms between requests

        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)

    def _rate_limit(self):
        """Ensure we don't overwhelm Wikipedia's servers"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request = time.time()

    def get_popular_pages(self, limit: int = 10000) -> List[str]:
        """
        Get list of popular Wikipedia pages based on view statistics
        Focus on stable, educational content with guaranteed hot spots
        """
        self.logger.info(f"Fetching {limit} popular pages...")

        # Strategy: Mix stable educational content with popular pages
        # This ensures we have both semantic diversity AND natural hot spots

        collected_pages = []



        # 2. Add popular pages from multiple time periods for stability
        try:
            popular_pages = self._get_popular_from_multiple_periods(limit)
            collected_pages.extend(popular_pages)
            self.logger.info(f"Added {len(popular_pages)} popular pages")
        except Exception as e:
            self.logger.warning(f"Failed to get popular pages: {e}")
            # Fill remaining with more stable pages

        # Remove duplicates while preserving order
        seen = set()
        unique_pages = []
        for page in collected_pages:
            if page not in seen:
                seen.add(page)
                unique_pages.append(page)

        self.logger.info(f"Final collection: {len(unique_pages)} unique pages")
        return unique_pages[:limit]

    def _get_popular_from_multiple_periods(self, limit: int) -> List[str]:
        """Get popular pages from multiple time periods for stability"""
        all_popular = []

        # Try different months to get stable popular content
        months = ['2025/01', '2025/02', '2025/03', '2025/04']

        for month in months:
            try:
                url = f"{self.wikimedia_url}/metrics/pageviews/top/en.wikipedia/all-access/{month}/all-days"
                self._rate_limit()
                response = self.session.get(url)

                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', [{}])[:1]:
                        for article in item.get('articles', []):
                            title = article.get('article', '').replace('_', ' ')
                            if self._is_suitable_page(title) and title not in all_popular:
                                all_popular.append(title)

                    self.logger.info(f"Got {len(all_popular)} pages from {month}")

                    if len(all_popular) >= limit:
                        break

            except Exception as e:
                self.logger.warning(f"Failed to get data from {month}: {e}")
                continue

        return all_popular[:limit]

    def _is_suitable_page(self, title: str) -> bool:
        """Filter out unsuitable pages for our research"""
        exclude_patterns = [
            'Main_Page', 'Special:', 'File:', 'Category:', 'Template:',
            'Wikipedia:', 'Help:', 'Portal:', 'List_of', 'disambiguation'
        ]

        return not any(pattern in title for pattern in exclude_patterns)

    def _get_stable_educational_pages(self, limit: int, offset: int = 0) -> List[str]:
        """
        Get stable, educational Wikipedia pages that are perfect for research
        These pages have rich semantic relationships and guaranteed hot spots
        """
        # Organized by domain - this creates natural semantic clusters
        stable_categories = {
            'science_physics': [
                "Albert Einstein", "Isaac Newton", "Marie Curie", "Stephen Hawking",
                "Quantum mechanics", "Relativity", "Electromagnetic radiation",
                "Particle physics", "Atomic theory", "Thermodynamics"
            ],
            'science_biology': [
                "Charles Darwin", "DNA", "Evolution", "Cell biology", "Genetics",
                "Photosynthesis", "Ecology", "Human anatomy", "Microbiology"
            ],
            'science_chemistry': [
                "Periodic table", "Chemical bond", "Organic chemistry",
                "Atom", "Molecule", "Chemical reaction", "Biochemistry"
            ],
            'geography_countries': [
                "United States", "China", "India", "Brazil", "Russia", "Germany",
                "Japan", "United Kingdom", "France", "Italy", "Canada", "Australia"
            ],
            'geography_cities': [
                "Paris", "London", "Tokyo", "New York City", "Rome", "Berlin",
                "Moscow", "Beijing", "Mumbai", "Los Angeles", "Chicago"
            ],
            'history_periods': [
                "World War II", "World War I", "Ancient Rome", "Renaissance",
                "French Revolution", "Industrial Revolution", "Cold War",
                "Ancient Egypt", "Medieval period", "American Civil War"
            ],
            'history_figures': [
                "Napoleon", "Julius Caesar", "George Washington", "Winston Churchill",
                "Abraham Lincoln", "Cleopatra", "Alexander the Great"
            ],
            'arts_visual': [
                "Leonardo da Vinci", "Vincent van Gogh", "Pablo Picasso",
                "Michelangelo", "Claude Monet", "Salvador Dalí", "Painting", "Sculpture"
            ],
            'arts_literature': [
                "William Shakespeare", "Literature", "Poetry", "Novel",
                "Charles Dickens", "Jane Austen", "Mark Twain"
            ],
            'arts_music': [
                "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "Music",
                "Johann Sebastian Bach", "Classical music", "Opera"
            ],
            'philosophy_religion': [
                "Philosophy", "Aristotle", "Plato", "Christianity", "Islam",
                "Buddhism", "Hinduism", "Ethics", "Logic"
            ],
            'technology_computing': [
                "Computer science", "Artificial intelligence", "Internet",
                "Programming language", "Algorithm", "Computer", "Software"
            ],
            'mathematics': [
                "Mathematics", "Calculus", "Geometry", "Statistics", "Algebra",
                "Number theory", "Pythagoras", "Euclid"
            ]
        }

        # Flatten all pages
        all_pages = []
        for category, pages in stable_categories.items():
            all_pages.extend(pages)

        # Apply offset and limit
        selected_pages = all_pages[offset:offset + limit]

        return selected_pages

    def get_page_data(self, title: str) -> Optional[WikiPage]:
        """
        Get comprehensive data for a single Wikipedia page
        """
        try:
            self._rate_limit()

            # Get page summary (includes extract, links count, etc.)
            summary_url = f"{self.base_url}/page/summary/{title.replace(' ', '_')}"
            summary_response = self.session.get(summary_url)

            if summary_response.status_code != 200:
                self.logger.warning(f"Failed to get summary for {title}")
                return None

            summary_data = summary_response.json()

            # Get full page content
            self._rate_limit()
            content_url = f"https://en.wikipedia.org/w/api.php"
            content_params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|links|categories|pageviews',
                'explaintext': True,
                'exsectionformat': 'plain',
                'pllimit': 'max',  # Get all links
                'cllimit': 'max',  # Get all categories
                'pvipdays': 30  # Get 30-day view stats
            }

            content_response = self.session.get(content_url, params=content_params)

            if content_response.status_code != 200:
                self.logger.warning(f"Failed to get content for {title}")
                return None

            content_data = content_response.json()

            # Extract page data
            pages = content_data.get('query', {}).get('pages', {})
            if not pages:
                return None

            page_data = list(pages.values())[0]

            # Extract links (filter to main namespace articles only)
            raw_links = page_data.get('links', [])
            links = [link['title'] for link in raw_links
                     if not any(prefix in link['title'] for prefix in
                                ['File:', 'Category:', 'Template:', 'Wikipedia:'])]

            # Extract categories (clean format)
            raw_categories = page_data.get('categories', [])
            categories = [cat['title'].replace('Category:', '') for cat in raw_categories]

            # Get view statistics (handle None values)
            pageviews = page_data.get('pageviews', {})
            if pageviews:
                valid_views = [v for v in pageviews.values() if v is not None and isinstance(v, (int, float))]
                avg_views = sum(valid_views) // len(valid_views) if valid_views else 0
            else:
                avg_views = 0

            return WikiPage(
                title=title,
                page_id=page_data.get('pageid', 0),
                content=page_data.get('extract', ''),
                links=links,
                categories=categories,
                views=avg_views,
                url=summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                extract=summary_data.get('extract', '')
            )

        except Exception as e:
            self.logger.error(f"Error processing {title}: {str(e)}")
            return None

    def collect_pages(self, page_titles: List[str], output_file: str) -> List[WikiPage]:
        """
        Collect data for multiple pages and save to file
        """
        collected_pages = []

        self.logger.info(f"Starting collection of {len(page_titles)} pages...")

        for i, title in enumerate(page_titles, 1):
            self.logger.info(f"Processing {i}/{len(page_titles)}: {title}")

            page_data = self.get_page_data(title)
            if page_data:
                collected_pages.append(page_data)

            # Save progress every 100 pages
            if i % 100 == 0:
                self._save_pages(collected_pages, f"{output_file}.temp")
                self.logger.info(f"Saved progress: {len(collected_pages)} pages collected")

        # Final save
        self._save_pages(collected_pages, output_file)
        self.logger.info(f"Collection complete! Saved {len(collected_pages)} pages to {output_file}")

        return collected_pages

    def _save_pages(self, pages: List[WikiPage], filename: str):
        """Save pages to JSON file"""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        pages_data = []
        for page in pages:
            pages_data.append({
                'title': page.title,
                'page_id': page.page_id,
                'content': page.content,
                'links': page.links,
                'categories': page.categories,
                'views': page.views,
                'url': page.url,
                'extract': page.extract
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, indent=2, ensure_ascii=False)
