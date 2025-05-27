"""
Wikipedia Data Collection Script
================================

This script handles the complete data collection process for Step 1.
Run this to collect your 10K Wikipedia pages for the thesis project.

Usage:
    python scripts/collect_wikipedia_data.py --pages 10000 --output data/input/pages/base_pages.json
"""

import argparse
import logging
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

from src.phase1_extraction.wikipedia_client import WikipediaClient


def setup_logging():
    """Setup logging for the collection process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/wikipedia_collection.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Collect Wikipedia data for thesis project')
    parser.add_argument('--pages', type=int, default=10000,
                        help='Number of pages to collect (default: 10000)')
    parser.add_argument('--output', type=str, default='data/input/pages/base_pages.json',
                        help='Output file path')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (collect only 50 pages)')

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create output directories
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path('data/logs').mkdir(parents=True, exist_ok=True)

    if args.test:
        args.pages = 50
        logger.info("Running in TEST MODE - collecting only 50 pages")

    logger.info(f"Starting Wikipedia data collection...")
    logger.info(f"Target pages: {args.pages}")
    logger.info(f"Output file: {args.output}")

    try:
        # Initialize client
        client = WikipediaClient()

        # Get popular page titles
        logger.info("Fetching popular page titles...")
        popular_titles = client.get_popular_pages(limit=args.pages)

        if not popular_titles:
            logger.error("Failed to get popular page titles!")
            return False

        logger.info(f"Found {len(popular_titles)} page titles")

        # Sample of titles we'll collect
        logger.info("Sample titles to collect:")
        for title in popular_titles[:5]:
            logger.info(f"  - {title}")

        # Collect page data
        logger.info("Starting page data collection...")
        pages = client.collect_pages(popular_titles, args.output)

        # Collection summary
        logger.info("=" * 50)
        logger.info("COLLECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Target pages: {args.pages}")
        logger.info(f"Successfully collected: {len(pages)}")
        logger.info(f"Success rate: {len(pages) / len(popular_titles) * 100:.1f}%")
        logger.info(f"Output file: {args.output}")

        # Sample statistics
        if pages:
            avg_content_length = sum(len(p.content) for p in pages) / len(pages)
            avg_links = sum(len(p.links) for p in pages) / len(pages)
            avg_categories = sum(len(p.categories) for p in pages) / len(pages)

            logger.info(f"Average content length: {avg_content_length:.0f} characters")
            logger.info(f"Average links per page: {avg_links:.1f}")
            logger.info(f"Average categories per page: {avg_categories:.1f}")

        # Save metadata
        metadata = {
            'collection_date': str(Path(args.output).stat().st_mtime),
            'total_pages': len(pages),
            'target_pages': args.pages,
            'success_rate': len(pages) / len(popular_titles) if popular_titles else 0,
            'sample_titles': popular_titles[:10] if popular_titles else []
        }

        metadata_file = str(Path(args.output).with_suffix('.metadata.json'))
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to: {metadata_file}")
        logger.info("Collection completed successfully! ✅")

        return True

    except Exception as e:
        logger.error(f"Collection failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
