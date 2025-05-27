# scripts/analyze_collected_data.py
"""
Analyze the collected Wikipedia data to understand what we have
This helps validate our collection and plan the next steps
"""

import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import statistics


def setup_logging():
    """Setup logging for the analysis process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/analyzed_data.log'),
            logging.StreamHandler()
        ]
    )


def load_pages(file_path: str) -> List[Dict]:
    """Load pages from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_basic_stats(pages: List[Dict]) -> Dict:
    """Basic statistics about the collected pages"""
    if not pages:
        return {}

    content_lengths = [len(page['content']) for page in pages]
    link_counts = [len(page['links']) for page in pages]
    category_counts = [len(page['categories']) for page in pages]
    view_counts = [page['views'] for page in pages if page['views'] > 0]

    stats = {
        'total_pages': len(pages),
        'content_stats': {
            'avg_length': statistics.mean(content_lengths),
            'median_length': statistics.median(content_lengths),
            'min_length': min(content_lengths),
            'max_length': max(content_lengths)
        },
        'link_stats': {
            'avg_links': statistics.mean(link_counts),
            'median_links': statistics.median(link_counts),
            'total_unique_links': len(set(link for page in pages for link in page['links']))
        },
        'category_stats': {
            'avg_categories': statistics.mean(category_counts),
            'total_unique_categories': len(set(cat for page in pages for cat in page['categories']))
        },
        'view_stats': {
            'avg_views': statistics.mean(view_counts) if view_counts else 0,
            'median_views': statistics.median(view_counts) if view_counts else 0
        }
    }

    return stats


def find_hot_spots(pages: List[Dict], top_n: int = 20) -> Dict:
    """Find the most referenced entities (hot spots for your thesis!)"""

    # Count how many times each page is linked to by others
    link_counter = Counter()
    for page in pages:
        for link in page['links']:
            link_counter[link] += 1

    # Count most common categories
    category_counter = Counter()
    for page in pages:
        for category in page['categories']:
            category_counter[category] += 1

    # Find pages that appear as links (potential hot spots)
    page_titles = {page['title'] for page in pages}
    internal_hot_spots = [(title, count) for title, count in link_counter.most_common(top_n)
                          if title in page_titles]

    return {
        'most_linked_internal': internal_hot_spots,
        'most_linked_external': link_counter.most_common(top_n),
        'most_common_categories': category_counter.most_common(top_n)
    }


def analyze_semantic_domains(pages: List[Dict]) -> Dict:
    """Analyze semantic domains in your collection"""

    # Define domain keywords for classification
    domain_keywords = {
        'science': ['physics', 'chemistry', 'biology', 'science', 'scientific', 'research', 'theory'],
        'history': ['war', 'century', 'ancient', 'empire', 'revolution', 'historical', 'medieval'],
        'geography': ['country', 'city', 'capital', 'continent', 'nation', 'republic', 'kingdom'],
        'arts': ['art', 'artist', 'painting', 'music', 'literature', 'culture', 'cultural'],
        'people': ['born', 'died', 'american', 'british', 'french', 'german', 'politician', 'writer'],
        'philosophy': ['philosophy', 'philosophical', 'religion', 'religious', 'belief', 'ethics']
    }

    domain_pages = defaultdict(list)

    for page in pages:
        page_text = (page['title'] + ' ' + ' '.join(page['categories'])).lower()

        for domain, keywords in domain_keywords.items():
            if any(keyword in page_text for keyword in keywords):
                domain_pages[domain].append(page['title'])

    return dict(domain_pages)


def main():

    setup_logging()
    logger = logging.getLogger(__name__)

    # Create output directories
    Path('data/logs').mkdir(parents=True, exist_ok=True)

    # Load your collected data
    data_file = "data/input/pages/base_pages.json"

    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Make sure you've run the data collection script first!")
        return

    logger.info(f"Loading data from {data_file}...")
    pages = load_pages(data_file)

    logger.info("=" * 60)
    logger.info("📊 WIKIPEDIA DATA ANALYSIS REPORT")
    logger.info("=" * 60)

    # Basic statistics
    logger.info("\n🔍 BASIC STATISTICS:")
    stats = analyze_basic_stats(pages)
    logger.info(f"Total pages collected: {stats['total_pages']}")
    logger.info(f"Average content length: {stats['content_stats']['avg_length']:.0f} characters")
    logger.info(f"Average links per page: {stats['link_stats']['avg_links']:.1f}")
    logger.info(f"Total unique outbound links: {stats['link_stats']['total_unique_links']:,}")
    logger.info(f"Average categories per page: {stats['category_stats']['avg_categories']:.1f}")
    logger.info(f"Total unique categories: {stats['category_stats']['total_unique_categories']}")

    # Hot spots analysis
    logger.info("\n🔥 HOT SPOTS ANALYSIS (Perfect for your thesis!):")
    hot_spots = find_hot_spots(pages)

    logger.info("\nMost linked INTERNAL pages (these will cause conflicts!):")
    for i, (title, count) in enumerate(hot_spots['most_linked_internal'][:10], 1):
        logger.info(f"  {i:2d}. {title} ({count} links)")

    logger.info("\nMost common categories:")
    for i, (category, count) in enumerate(hot_spots['most_common_categories'][:10], 1):
        logger.info(f"  {i:2d}. {category} ({count} pages)")

    # Semantic domains
    logger.info("\n🧠 SEMANTIC DOMAIN DISTRIBUTION:")
    domains = analyze_semantic_domains(pages)
    for domain, domain_pages in domains.items():
        logger.info(f"{domain.capitalize()}: {len(domain_pages)} pages")
        if domain_pages:
            sample = domain_pages[:3]
            logger.info(f"  Sample: {', '.join(sample)}")

    # Collection quality assessment
    logger.info("\n✅ COLLECTION QUALITY ASSESSMENT:")

    # Check for good semantic diversity
    domain_counts = [(domain, len(domain_pages)) for domain, domain_pages in domains.items()]
    domain_counts.sort(key=lambda x: x[1], reverse=True)

    if len(domains) >= 5:
        logger.info("✅ Good semantic diversity - 5+ domains represented")
    else:
        logger.info("⚠️  Limited semantic diversity - consider more varied content")

    # Check for hot spots
    internal_hot_spots = len(hot_spots['most_linked_internal'])
    if internal_hot_spots >= 10:
        logger.info(f"✅ Excellent hot spot potential - {internal_hot_spots} internal hot spots found")
    else:
        logger.info("⚠️  Few internal hot spots - may need more interconnected pages")

    # Check average connectivity
    avg_links = stats['link_stats']['avg_links']
    if avg_links >= 50:
        logger.info(f"✅ High connectivity - {avg_links:.0f} avg links per page")
    elif avg_links >= 20:
        logger.info(f"✅ Good connectivity - {avg_links:.0f} avg links per page")
    else:
        logger.info(f"⚠️  Low connectivity - {avg_links:.0f} avg links per page")

    logger.info("\n🎯 NEXT STEPS FOR YOUR THESIS:")
    logger.info("1. ✅ Data collection complete - great foundation!")
    logger.info("2. 🔄 Next: Entity extraction and cleaning (Week 2)")
    logger.info("3. 🏗️  Then: Neo4j setup and base graph loading")
    logger.info("4. 🧠 Finally: Your adaptive grouping algorithm!")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete! Your data looks ready for the next phase. 🚀")


if __name__ == "__main__":
    main()