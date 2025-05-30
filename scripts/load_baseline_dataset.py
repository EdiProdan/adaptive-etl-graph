# scripts/load_baseline_dataset.py
"""
Baseline Dataset Loading for Research Validation
==============================================

Implements traditional static batching approach for comparison against
adaptive algorithm. Provides comprehensive performance measurement for
research validation and thesis documentation.

Key Features:
- Static batch processing (control group for experiments)
- Comprehensive performance metrics collection
- Memory-efficient processing for 8GB RAM systems
- Detailed logging for research documentation
"""

import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
from datetime import datetime
import psutil
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.neo4j_connector import Neo4jConnector, ConnectionConfig, BatchMetrics


class BaselineLoader:
    """
    Traditional static batching implementation for research comparison

    This represents the "control group" against which your adaptive
    algorithm will be evaluated. Implements straightforward batch
    processing without intelligent optimization.
    """

    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size
        self.db = Neo4jConnector()
        self.performance_metrics = []

        # Setup logging for research documentation
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/baseline_loading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Performance monitoring
        self.start_time = None
        self.total_entities_processed = 0
        self.total_relationships_processed = 0

    def load_consolidated_dataset(self, entities_file: str, relationships_file: str) -> Dict:
        """
        Load the consolidated research dataset using traditional static batching

        Args:
            entities_file: Path to consolidated_entities.json
            relationships_file: Path to consolidated_relationships.json

        Returns:
            Performance analysis results for research documentation
        """
        self.logger.info("=" * 60)
        self.logger.info("🏗️  BASELINE LOADING: Traditional Static Batching")
        self.logger.info("=" * 60)

        self.start_time = time.time()

        # Load dataset files
        self.logger.info("📂 Loading consolidated dataset files...")
        entities = self._load_json_file(entities_file)
        relationships = self._load_json_file(relationships_file)

        self.logger.info(f"📊 Dataset Statistics:")
        self.logger.info(f"   Total entities: {len(entities):,}")
        self.logger.info(f"   Total relationships: {len(relationships):,}")
        self.logger.info(f"   Batch size: {self.batch_size}")
        self.logger.info(f"   Expected batches: {len(entities) // self.batch_size + 1}")

        # Create database schema
        self.logger.info("🔧 Creating optimized database schema...")
        if not self.db.create_schema():
            raise RuntimeError("Failed to create database schema")

        # Process entities in static batches
        self.logger.info("⚡ Starting entity batch processing...")
        entity_metrics = self._process_entity_batches(entities)

        # Process relationships in static batches
        self.logger.info("🔗 Starting relationship batch processing...")
        relationship_metrics = self._process_relationship_batches(relationships)

        # Generate comprehensive analysis
        analysis_results = self._generate_performance_analysis()

        # Save results for research documentation
        self._save_baseline_results(analysis_results)

        self.logger.info("✅ Baseline loading completed successfully!")
        return analysis_results

    def _load_json_file(self, file_path: str) -> List[Dict]:
        """Load JSON file with memory optimization"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"✅ Loaded {len(data):,} records from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"❌ Failed to load {file_path}: {str(e)}")
            raise

    def _process_entity_batches(self, entities: List[Dict]) -> List[BatchMetrics]:
        """Process entities using static batching strategy"""
        entity_metrics = []
        total_batches = len(entities) // self.batch_size + (1 if len(entities) % self.batch_size else 0)

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(entities))

            batch_entities = entities[start_idx:end_idx]
            batch_id = f"entities_batch_{batch_idx:04d}"

            self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches}: "
                             f"{len(batch_entities)} entities")

            # Monitor system resources
            memory_before = psutil.virtual_memory().percent

            # Process batch
            start_time = time.time()
            try:
                # Convert to database format
                db_entities = self._convert_entities_for_db(batch_entities)

                # Load batch with monitoring
                metrics = self.db.load_batch_with_monitoring(
                    entities=db_entities,
                    relationships=[],  # Entities only
                    batch_id=batch_id
                )

                # Track performance
                metrics.memory_usage = psutil.virtual_memory().percent - memory_before
                entity_metrics.append(metrics)
                self.total_entities_processed += len(batch_entities)

                self.logger.info(f"✅ Batch completed in {metrics.processing_time:.2f}s")

            except Exception as e:
                self.logger.error(f"❌ Batch {batch_id} failed: {str(e)}")
                continue

            # Memory cleanup for resource-constrained environment
            if batch_idx % 10 == 0:
                gc.collect()

        return entity_metrics

    def _process_relationship_batches(self, relationships: List[Dict]) -> List[BatchMetrics]:
        """Process relationships using static batching strategy"""
        relationship_metrics = []
        total_batches = len(relationships) // self.batch_size + (1 if len(relationships) % self.batch_size else 0)

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(relationships))

            batch_relationships = relationships[start_idx:end_idx]
            batch_id = f"relationships_batch_{batch_idx:04d}"

            self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches}: "
                             f"{len(batch_relationships)} relationships")

            # Monitor system resources
            memory_before = psutil.virtual_memory().percent

            # Process batch
            try:
                # Load batch with monitoring
                metrics = self.db.load_batch_with_monitoring(
                    entities=[],  # Relationships only
                    relationships=batch_relationships,
                    batch_id=batch_id
                )

                # Track performance
                metrics.memory_usage = psutil.virtual_memory().percent - memory_before
                relationship_metrics.append(metrics)
                self.total_relationships_processed += len(batch_relationships)

                self.logger.info(f"✅ Batch completed in {metrics.processing_time:.2f}s")

            except Exception as e:
                self.logger.error(f"❌ Batch {batch_id} failed: {str(e)}")
                continue

            # Memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()

        return relationship_metrics

    def _convert_entities_for_db(self, entities: List[Dict]) -> List[Dict]:
        """Convert entity format for database loading"""
        db_entities = []

        for entity in entities:
            db_entity = {
                'title': entity.get('title', ''),
                'page_id': entity.get('page_id', 0),
                'content': entity.get('content', ''),
                'views': entity.get('views', 0),
                'url': entity.get('url', ''),
                'links': entity.get('links', []),
                'categories': entity.get('categories', [])
            }
            db_entities.append(db_entity)

        return db_entities

    def _generate_performance_analysis(self) -> Dict:
        """
        Generate comprehensive performance analysis for research documentation

        This provides the baseline metrics against which your adaptive
        algorithm performance will be compared.
        """
        total_time = time.time() - self.start_time

        # Collect all processing times
        all_metrics = self.performance_metrics
        if not all_metrics:
            return {"error": "No performance metrics collected"}

        processing_times = [m.processing_time for m in all_metrics]
        memory_usage = [m.memory_usage for m in all_metrics]
        entity_counts = [m.entity_count for m in all_metrics]
        relationship_counts = [m.relationship_count for m in all_metrics]

        # Database analysis
        db_analysis = self.db.analyze_graph_structure()
        hot_spots = self.db.get_hot_spots(limit=50)

        analysis = {
            "experiment_metadata": {
                "loading_strategy": "Traditional Static Batching",
                "batch_size": self.batch_size,
                "total_processing_time": total_time,
                "entities_processed": self.total_entities_processed,
                "relationships_processed": self.total_relationships_processed,
                "batches_completed": len(all_metrics),
                "timestamp": datetime.now().isoformat()
            },

            "performance_metrics": {
                "avg_batch_time": statistics.mean(processing_times),
                "median_batch_time": statistics.median(processing_times),
                "min_batch_time": min(processing_times),
                "max_batch_time": max(processing_times),
                "batch_time_stddev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,

                "avg_memory_usage": statistics.mean(memory_usage) if memory_usage else 0,
                "peak_memory_usage": max(memory_usage) if memory_usage else 0,

                "throughput_entities_per_second": self.total_entities_processed / total_time,
                "throughput_relationships_per_second": self.total_relationships_processed / total_time,

                "avg_entities_per_batch": statistics.mean(entity_counts) if entity_counts else 0,
                "avg_relationships_per_batch": statistics.mean(relationship_counts) if relationship_counts else 0
            },

            "database_analysis": db_analysis,

            "hot_spots_analysis": {
                "total_hot_spots": len(hot_spots),
                "top_10_hot_spots": hot_spots[:10],
                "hot_spot_link_distribution": [h['incoming_links'] for h in hot_spots[:20]]
            },

            "research_implications": {
                "baseline_established": True,
                "suitable_for_comparison": len(all_metrics) > 10,
                "hot_spots_present": len(hot_spots) > 10,
                "performance_variance": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                "algorithm_optimization_potential": max(processing_times) / min(
                    processing_times) if processing_times else 1.0
            }
        }

        return analysis

    def _save_baseline_results(self, analysis: Dict):
        """Save baseline results for research documentation"""

        # Save detailed analysis
        output_file = "data/processed/phase3/baseline_performance_analysis.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 Baseline analysis saved to: {output_file}")

        # Generate research summary
        self._generate_research_summary(analysis)

    def _generate_research_summary(self, analysis: Dict):
        """Generate human-readable research summary"""

        summary_lines = [
            "=" * 60,
            "📊 BASELINE PERFORMANCE ANALYSIS - RESEARCH SUMMARY",
            "=" * 60,
            "",
            "🎯 EXPERIMENT OVERVIEW:",
            f"   Strategy: {analysis['experiment_metadata']['loading_strategy']}",
            f"   Batch Size: {analysis['experiment_metadata']['batch_size']:,}",
            f"   Total Time: {analysis['experiment_metadata']['total_processing_time']:.2f} seconds",
            f"   Entities Processed: {analysis['experiment_metadata']['entities_processed']:,}",
            f"   Relationships Processed: {analysis['experiment_metadata']['relationships_processed']:,}",
            "",
            "⚡ PERFORMANCE METRICS:",
            f"   Average Batch Time: {analysis['performance_metrics']['avg_batch_time']:.3f}s",
            f"   Median Batch Time: {analysis['performance_metrics']['median_batch_time']:.3f}s",
            f"   Performance Variance: {analysis['performance_metrics']['batch_time_stddev']:.3f}s",
            f"   Entity Throughput: {analysis['performance_metrics']['throughput_entities_per_second']:.1f} entities/sec",
            f"   Relationship Throughput: {analysis['performance_metrics']['throughput_relationships_per_second']:.1f} rel/sec",
            "",
            "🔥 HOT SPOTS ANALYSIS:",
            f"   Total Hot Spots Identified: {analysis['hot_spots_analysis']['total_hot_spots']}",
            f"   Top Hot Spot: {analysis['hot_spots_analysis']['top_10_hot_spots'][0]['title']} "
            f"({analysis['hot_spots_analysis']['top_10_hot_spots'][0]['incoming_links']} links)" if
            analysis['hot_spots_analysis']['top_10_hot_spots'] else "None",
            "",
            "🎓 RESEARCH VALIDATION:",
            f"   Baseline Established: {'✅ Yes' if analysis['research_implications']['baseline_established'] else '❌ No'}",
            f"   Suitable for Algorithm Comparison: {'✅ Yes' if analysis['research_implications']['suitable_for_comparison'] else '❌ No'}",
            f"   Hot Spots Present for Conflict Testing: {'✅ Yes' if analysis['research_implications']['hot_spots_present'] else '❌ No'}",
            f"   Optimization Potential: {analysis['research_implications']['algorithm_optimization_potential']:.2f}x",
            "",
            "🚀 NEXT STEPS:",
            "   1. ✅ Baseline measurements complete",
            "   2. 🔄 Ready for adaptive algorithm implementation",
            "   3. 🧠 Begin SBERT-based semantic clustering development",
            "   4. 📊 Implement real-time conflict detection",
            "   5. 🎯 Develop dynamic rebalancing algorithms",
            "",
            "=" * 60,
            f"🎉 BASELINE LOADING SUCCESSFUL - READY FOR ALGORITHM DEVELOPMENT! 🎉",
            "=" * 60
        ]

        # Log summary
        for line in summary_lines:
            self.logger.info(line)

        # Save summary to file
        summary_file = "data/processed/phase3/baseline_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        self.logger.info(f"📄 Research summary saved to: {summary_file}")


def main():
    """
    Main execution function for baseline dataset loading

    This establishes the performance baseline against which your
    adaptive algorithm will be evaluated for thesis validation.
    """

    # Configuration
    BATCH_SIZE = 500  # Optimized for 8GB RAM system

    # Dataset file paths (from your consolidation work)
    ENTITIES_FILE = "data/processed/phase1/consolidated_entities.json"
    RELATIONSHIPS_FILE = "data/processed/phase1/consolidated_relationships.json"

    # Verify files exist
    if not Path(ENTITIES_FILE).exists():
        print(f"❌ Entities file not found: {ENTITIES_FILE}")
        print("Please ensure you've completed the data consolidation step.")
        return False

    if not Path(RELATIONSHIPS_FILE).exists():
        print(f"❌ Relationships file not found: {RELATIONSHIPS_FILE}")
        print("Please ensure you've completed the data consolidation step.")
        return False

    try:
        # Initialize baseline loader
        loader = BaselineLoader(batch_size=BATCH_SIZE)

        # Execute baseline loading with performance measurement
        analysis_results = loader.load_consolidated_dataset(
            entities_file=ENTITIES_FILE,
            relationships_file=RELATIONSHIPS_FILE
        )

        # Cleanup
        loader.db.close()

        print("\n🎯 Baseline loading completed successfully!")
        print("📊 Performance analysis saved for algorithm comparison.")
        print("🚀 Ready for adaptive algorithm development!")

        return True

    except Exception as e:
        logging.error(f"❌ Baseline loading failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)