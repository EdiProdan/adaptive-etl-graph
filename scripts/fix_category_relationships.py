# scripts/fix_category_relationships.py
"""
Category Relationship Correction for Research Database
====================================================

Addresses the missing BELONGS_TO relationships warning by properly
establishing entity-to-category connections in the research database.

Research Impact: Enhances data model completeness while maintaining
primary research objectives focused on entity-entity relationships.
"""

import sys
from pathlib import Path
import logging
import time


project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

from src.database.neo4j_connector import Neo4jConnector


def fix_category_relationships():
    """
    Establish missing entity-to-category relationships

    This enhances the data model without affecting core research objectives
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🔧 Fixing Category Relationships")
    logger.info("=" * 40)

    db = Neo4jConnector()

    try:
        # Check current state
        result = db.execute_query("""
            MATCH (e:Entity)
            WHERE e.categories IS NOT NULL AND size(e.categories) > 0
            RETURN count(e) as entities_with_categories
        """)

        entities_with_cats = result[0]['entities_with_categories']
        logger.info(f"📊 Found {entities_with_cats:,} entities with category data")

        if entities_with_cats == 0:
            logger.warning("⚠️  No entities have category data - check data loading")
            return False

        # Create category relationships in batches
        logger.info("🔄 Creating category relationships...")

        result = db.execute_query("""
            MATCH (e:Entity)
            WHERE e.categories IS NOT NULL AND size(e.categories) > 0
            WITH e, e.categories as cats
            UNWIND cats as category_name
            MERGE (c:Category {name: category_name})
            ON CREATE SET c.created_at = datetime()
            MERGE (e)-[r:BELONGS_TO]->(c)
            ON CREATE SET r.created_at = datetime()
            RETURN count(r) as relationships_created
        """)

        relationships_created = result[0]['relationships_created'] if result else 0
        logger.info(f"✅ Created {relationships_created:,} category relationships")

        # Verify fix
        verification = db.execute_query("""
            MATCH (e:Entity)-[r:BELONGS_TO]->(c:Category)
            RETURN count(r) as total_category_relationships,
                   count(DISTINCT e) as entities_with_categories,
                   count(DISTINCT c) as total_categories
        """)

        if verification:
            stats = verification[0]
            logger.info("📊 Verification Results:")
            logger.info(f"   Category relationships: {stats['total_category_relationships']:,}")
            logger.info(f"   Entities with categories: {stats['entities_with_categories']:,}")
            logger.info(f"   Total categories: {stats['total_categories']:,}")

        logger.info("✅ Category relationship fix completed")
        return True

    except Exception as e:
        logger.error(f"❌ Category relationship fix failed: {str(e)}")
        return False

    finally:
        db.close()


if __name__ == "__main__":
    success = fix_category_relationships()
    exit(0 if success else 1)