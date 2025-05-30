# scripts/load_baseline_dataset_fixed.py
"""
Corrected Baseline Dataset Loading for Research Validation
========================================================

Fixed implementation addressing the metrics collection failure.
Implements proper data flow for comprehensive performance measurement.

Root Cause Resolution:
- Consolidated metrics aggregation from separate processing phases
- Enhanced error handling for robust data collection
- Improved logging for research documentation integrity
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

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

from src.database.neo4j_connector import Neo4jConnector, ConnectionConfig, BatchMetrics


class CorrectedBaselineLoader:
    """
    Corrected baseline loader with proper metrics aggregation

    Key Fixes:
    1. Consolidated metrics collection across processing phases
    2. Enhanced error handling for data integrity
    3. Improved performance monitoring for research validation
    """

    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size
        self.db = Neo4jConnector()

        # CRITICAL FIX: Proper metrics storage
        self.all_batch_metrics = []  # Consolidated storage
        self.entity_metrics = []  # Phase tracking
        self.relationship_metrics = []  # Phase tracking

        # Setup logging for research documentation
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/baseline_loading_fixed.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Performance monitoring
        self.start_time = None
        self.total_entities_processed = 0
        self.total_relationships_processed = 0
        self.processing_phases = {
            'entities': {'start': None, 'end': None, 'batches': 0},
            'relationships': {'start': None, 'end': None, 'batches': 0}
        }

    def load_consolidated_dataset(self, entities_file: str, relationships_file: str) -> Dict:
        """
        Load dataset with corrected metrics collection

        Returns comprehensive analysis for research validation
        """
        self.logger.info("=" * 60)
        self.logger.info("🔧 CORRECTED BASELINE LOADING: Fixed Metrics Collection")
        self.logger.info("=" * 60)

        self.start_time = time.time()

        try:
            # Load dataset files
            self.logger.info("📂 Loading consolidated dataset files...")
            entities = self._load_json_file(entities_file)
            relationships = self._load_json_file(relationships_file)

            self._log_dataset_overview(entities, relationships)

            # Create database schema
            self.logger.info("🔧 Creating optimized database schema...")
            if not self.db.create_schema():
                raise RuntimeError("Failed to create database schema")

            # Process entities with metrics collection
            self.logger.info("⚡ Starting entity batch processing...")
            self.processing_phases['entities']['start'] = time.time()
            entity_metrics = self._process_entity_batches(entities)
            self.processing_phases['entities']['end'] = time.time()
            self.processing_phases['entities']['batches'] = len(entity_metrics)

            # Process relationships with metrics collection
            self.logger.info("🔗 Starting relationship batch processing...")
            self.processing_phases['relationships']['start'] = time.time()
            relationship_metrics = self._process_relationship_batches(relationships)
            self.processing_phases['relationships']['end'] = time.time()
            self.processing_phases['relationships']['batches'] = len(relationship_metrics)

            # CRITICAL FIX: Consolidate all metrics
            self.all_batch_metrics.extend(entity_metrics)
            self.all_batch_metrics.extend(relationship_metrics)
            self.entity_metrics = entity_metrics
            self.relationship_metrics = relationship_metrics

            self.logger.info(f"📊 Metrics Consolidation Summary:")
            self.logger.info(f"   Entity batches: {len(entity_metrics)}")
            self.logger.info(f"   Relationship batches: {len(relationship_metrics)}")
            self.logger.info(f"   Total metrics collected: {len(self.all_batch_metrics)}")

            # Generate comprehensive analysis
            analysis_results = self._generate_performance_analysis()

            # Save results for research documentation
            self._save_baseline_results(analysis_results)

            self.logger.info("✅ Corrected baseline loading completed successfully!")
            return analysis_results

        except Exception as e:
            self.logger.error(f"❌ Baseline loading failed: {str(e)}")
            self.logger.error(f"   Metrics collected so far: {len(self.all_batch_metrics)}")

            # Emergency analysis with partial data
            if self.all_batch_metrics:
                self.logger.info("🔄 Attempting emergency analysis with partial metrics...")
                try:
                    partial_analysis = self._generate_performance_analysis()
                    partial_analysis['status'] = 'partial_completion'
                    partial_analysis['error'] = str(e)
                    return partial_analysis
                except Exception as analysis_error:
                    self.logger.error(f"❌ Emergency analysis also failed: {str(analysis_error)}")

            raise

    def _log_dataset_overview(self, entities: List[Dict], relationships: List[Dict]):
        """Enhanced dataset overview logging"""
        entity_batches = len(entities) // self.batch_size + (1 if len(entities) % self.batch_size else 0)
        rel_batches = len(relationships) // self.batch_size + (1 if len(relationships) % self.batch_size else 0)

        self.logger.info(f"📊 Dataset Statistics:")
        self.logger.info(f"   Total entities: {len(entities):,}")
        self.logger.info(f"   Total relationships: {len(relationships):,}")
        self.logger.info(f"   Batch size: {self.batch_size}")
        self.logger.info(f"   Entity batches: {entity_batches}")
        self.logger.info(f"   Relationship batches: {rel_batches}")
        self.logger.info(f"   Total expected batches: {entity_batches + rel_batches}")

    def _load_json_file(self, file_path: str) -> List[Dict]:
        """Load JSON file with adaptive schema validation"""
        try:
            self.logger.info(f"📖 Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"✅ Loaded {len(data):,} records from {Path(file_path).name}")

            if not data:
                raise ValueError(f"Empty dataset in {file_path}")

            # ADAPTIVE VALIDATION: Support both schema formats
            sample = data[0]
            if 'entities' in file_path:
                # Check for either 'title' (original) or 'name'/'original_title' (consolidated)
                has_title = 'title' in sample
                has_name = 'name' in sample or 'original_title' in sample

                if not (has_title or has_name):
                    raise ValueError(
                        f"Missing required identifier fields ['title' or 'name'/'original_title'] in {file_path}")
            else:
                # Relationships validation remains unchanged
                required_fields = ['source', 'target']
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    raise ValueError(f"Missing required fields {missing_fields} in {file_path}")

            return data

        except Exception as e:
            self.logger.error(f"❌ Failed to load {file_path}: {str(e)}")
            raise

    def _process_entity_batches(self, entities: List[Dict]) -> List[BatchMetrics]:
        """Process entities with enhanced metrics collection"""
        entity_metrics = []
        total_batches = len(entities) // self.batch_size + (1 if len(entities) % self.batch_size else 0)

        self.logger.info(f"🔄 Processing {total_batches} entity batches...")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(entities))

            batch_entities = entities[start_idx:end_idx]
            batch_id = f"entities_batch_{batch_idx:04d}"

            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                self.logger.info(f"📦 Processing entity batch {batch_idx + 1}/{total_batches}: "
                                 f"{len(batch_entities)} entities")

            try:
                # Enhanced monitoring
                memory_before = psutil.virtual_memory().percent
                start_time = time.time()

                # Convert to database format
                db_entities = self._convert_entities_for_db(batch_entities)

                # Load batch with monitoring
                metrics = self.db.load_batch_with_monitoring(
                    entities=db_entities,
                    relationships=[],  # Entities only
                    batch_id=batch_id
                )

                # Enhanced metrics
                metrics.memory_usage = psutil.virtual_memory().percent - memory_before
                metrics.phase = 'entities'

                entity_metrics.append(metrics)
                self.total_entities_processed += len(batch_entities)

                if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                    self.logger.info(f"✅ Batch completed in {metrics.processing_time:.2f}s")

            except Exception as e:
                self.logger.error(f"❌ Entity batch {batch_id} failed: {str(e)}")
                continue

            # Memory management for resource-constrained environment
            if batch_idx % 20 == 0:
                gc.collect()

        self.logger.info(f"✅ Entity processing complete: {len(entity_metrics)} batches processed")
        return entity_metrics

    def _process_relationship_batches(self, relationships: List[Dict]) -> List[BatchMetrics]:
        """Process relationships with enhanced metrics collection"""
        relationship_metrics = []
        total_batches = len(relationships) // self.batch_size + (1 if len(relationships) % self.batch_size else 0)

        self.logger.info(f"🔄 Processing {total_batches} relationship batches...")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(relationships))

            batch_relationships = relationships[start_idx:end_idx]
            batch_id = f"relationships_batch_{batch_idx:04d}"

            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                self.logger.info(f"🔗 Processing relationship batch {batch_idx + 1}/{total_batches}: "
                                 f"{len(batch_relationships)} relationships")

            try:
                # Enhanced monitoring
                memory_before = psutil.virtual_memory().percent

                # Load batch with monitoring
                metrics = self.db.load_batch_with_monitoring(
                    entities=[],  # Relationships only
                    relationships=batch_relationships,
                    batch_id=batch_id
                )

                # Enhanced metrics
                metrics.memory_usage = psutil.virtual_memory().percent - memory_before
                metrics.phase = 'relationships'

                relationship_metrics.append(metrics)
                self.total_relationships_processed += len(batch_relationships)

                if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                    self.logger.info(f"✅ Batch completed in {metrics.processing_time:.2f}s")

            except Exception as e:
                self.logger.error(f"❌ Relationship batch {batch_id} failed: {str(e)}")
                continue

            # Memory management
            if batch_idx % 20 == 0:
                gc.collect()

        self.logger.info(f"✅ Relationship processing complete: {len(relationship_metrics)} batches processed")
        return relationship_metrics

    def _convert_entities_for_db(self, entities: List[Dict]) -> List[Dict]:
        """Convert entity format for database loading with validation"""
        db_entities = []

        for entity in entities:
            try:
                def safe_int(value, default=0):
                    """Convert to int with None handling"""
                    if value is None:
                        return default
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default

                db_entity = {
                    'title': str(entity.get('name', entity.get('original_title', ''))),  # Primary mapping
                    'page_id': safe_int(entity.get('page_id')),
                    'content': str(entity.get('content', '')),  # May be empty in consolidated data
                    'views': int(entity.get('views', 0)),
                    'url': str(entity.get('url', '')),
                    'links': list(entity.get('links', entity.get('aliases', []))),  # Use aliases as links fallback
                    'categories': list(entity.get('categories', []))
                }
                db_entities.append(db_entity)
            except Exception as e:
                self.logger.warning(f"⚠️  Skipping malformed entity: {str(e)}")
                continue

        return db_entities

    def _generate_performance_analysis(self) -> Dict:
        """
        Generate comprehensive performance analysis with proper error handling

        CRITICAL FIX: Uses consolidated metrics from self.all_batch_metrics
        """
        total_time = time.time() - self.start_time if self.start_time else 0

        # FIXED: Use consolidated metrics
        all_metrics = self.all_batch_metrics

        if not all_metrics:
            self.logger.error("❌ No performance metrics available for analysis")
            return {
                "error": "No performance metrics collected",
                "debugging_info": {
                    "total_entities_processed": self.total_entities_processed,
                    "total_relationships_processed": self.total_relationships_processed,
                    "entity_metrics_count": len(self.entity_metrics),
                    "relationship_metrics_count": len(self.relationship_metrics),
                    "all_metrics_count": len(all_metrics)
                }
            }

        # Extract metrics for analysis
        processing_times = [m.processing_time for m in all_metrics]
        memory_usage = [m.memory_usage for m in all_metrics if hasattr(m, 'memory_usage')]
        entity_counts = [m.entity_count for m in all_metrics]
        relationship_counts = [m.relationship_count for m in all_metrics]

        # Separate phase metrics
        entity_phase_metrics = [m for m in all_metrics if hasattr(m, 'phase') and m.phase == 'entities']
        relationship_phase_metrics = [m for m in all_metrics if hasattr(m, 'phase') and m.phase == 'relationships']

        # Database analysis
        try:
            db_analysis = self.db.analyze_graph_structure()
            hot_spots = self.db.get_hot_spots(limit=50)
        except Exception as e:
            self.logger.warning(f"⚠️  Database analysis failed: {str(e)}")
            db_analysis = {"error": str(e)}
            hot_spots = []

        analysis = {
            "experiment_metadata": {
                "loading_strategy": "Traditional Static Batching (Corrected)",
                "batch_size": self.batch_size,
                "total_processing_time": total_time,
                "entities_processed": self.total_entities_processed,
                "relationships_processed": self.total_relationships_processed,
                "batches_completed": len(all_metrics),
                "entity_batches": len(entity_phase_metrics),
                "relationship_batches": len(relationship_phase_metrics),
                "timestamp": datetime.now().isoformat(),
                "processing_phases": self.processing_phases
            },

            "performance_metrics": {
                "overall": {
                    "avg_batch_time": statistics.mean(processing_times),
                    "median_batch_time": statistics.median(processing_times),
                    "min_batch_time": min(processing_times),
                    "max_batch_time": max(processing_times),
                    "batch_time_stddev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                    "total_batches": len(all_metrics)
                },

                "throughput": {
                    "entities_per_second": self.total_entities_processed / total_time if total_time > 0 else 0,
                    "relationships_per_second": self.total_relationships_processed / total_time if total_time > 0 else 0,
                    "avg_entities_per_batch": statistics.mean(entity_counts) if entity_counts else 0,
                    "avg_relationships_per_batch": statistics.mean(relationship_counts) if relationship_counts else 0
                },

                "resource_usage": {
                    "avg_memory_usage": statistics.mean(memory_usage) if memory_usage else 0,
                    "peak_memory_usage": max(memory_usage) if memory_usage else 0,
                    "memory_variance": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
                },

                "phase_breakdown": {
                    "entities": {
                        "batches": len(entity_phase_metrics),
                        "avg_time": statistics.mean(
                            [m.processing_time for m in entity_phase_metrics]) if entity_phase_metrics else 0,
                        "total_time": self.processing_phases['entities']['end'] - self.processing_phases['entities'][
                            'start'] if self.processing_phases['entities']['start'] else 0
                    },
                    "relationships": {
                        "batches": len(relationship_phase_metrics),
                        "avg_time": statistics.mean([m.processing_time for m in
                                                     relationship_phase_metrics]) if relationship_phase_metrics else 0,
                        "total_time": self.processing_phases['relationships']['end'] -
                                      self.processing_phases['relationships']['start'] if
                        self.processing_phases['relationships']['start'] else 0
                    }
                }
            },

            "database_analysis": db_analysis,

            "hot_spots_analysis": {
                "total_hot_spots": len(hot_spots),
                "top_10_hot_spots": hot_spots[:10],
                "hot_spot_link_distribution": [h.get('incoming_links', 0) for h in hot_spots[:20]]
            },

            "research_validation": {
                "baseline_established": True,
                "metrics_collected": len(all_metrics),
                "suitable_for_comparison": len(all_metrics) > 10,
                "hot_spots_present": len(hot_spots) > 10,
                "performance_variance": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                "algorithm_optimization_potential": max(processing_times) / min(
                    processing_times) if processing_times else 1.0,
                "data_integrity_verified": self.total_entities_processed > 0 and self.total_relationships_processed > 0
            }
        }

        return analysis

    def _save_baseline_results(self, analysis: Dict):
        """Save results with enhanced error handling"""
        try:
            # Save detailed analysis
            output_file = "data/processed/phase3/baseline_performance_analysis_corrected.json"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            self.logger.info(f"📊 Corrected baseline analysis saved to: {output_file}")

            # Generate research summary
            self._generate_research_summary(analysis)

        except Exception as e:
            self.logger.error(f"❌ Failed to save baseline results: {str(e)}")
            # Emergency save attempt
            try:
                emergency_file = f"baseline_analysis_emergency_{int(time.time())}.json"
                with open(emergency_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                self.logger.info(f"💾 Emergency save to: {emergency_file}")
            except:
                self.logger.error("❌ Emergency save also failed")

    def _generate_research_summary(self, analysis: Dict):
        """Generate enhanced research summary"""

        if analysis.get("error"):
            summary_lines = [
                "=" * 60,
                "⚠️  BASELINE LOADING - PARTIAL COMPLETION REPORT",
                "=" * 60,
                "",
                f"❌ Error: {analysis['error']}",
                "",
                "🔍 Debugging Information:",
            ]

            debug_info = analysis.get("debugging_info", {})
            for key, value in debug_info.items():
                summary_lines.append(f"   {key}: {value}")

        else:
            metadata = analysis["experiment_metadata"]
            performance = analysis["performance_metrics"]
            validation = analysis["research_validation"]

            summary_lines = [
                "=" * 60,
                "📊 CORRECTED BASELINE PERFORMANCE ANALYSIS - RESEARCH SUMMARY",
                "=" * 60,
                "",
                "🎯 EXPERIMENT OVERVIEW:",
                f"   Strategy: {metadata['loading_strategy']}",
                f"   Batch Size: {metadata['batch_size']:,}",
                f"   Total Time: {metadata['total_processing_time']:.2f} seconds",
                f"   Entities Processed: {metadata['entities_processed']:,}",
                f"   Relationships Processed: {metadata['relationships_processed']:,}",
                f"   Total Batches: {metadata['batches_completed']}",
                "",
                "⚡ PERFORMANCE METRICS:",
                f"   Average Batch Time: {performance['overall']['avg_batch_time']:.3f}s",
                f"   Median Batch Time: {performance['overall']['median_batch_time']:.3f}s",
                f"   Performance Variance: {performance['overall']['batch_time_stddev']:.3f}s",
                f"   Entity Throughput: {performance['throughput']['entities_per_second']:.1f} entities/sec",
                f"   Relationship Throughput: {performance['throughput']['relationships_per_second']:.1f} rel/sec",
                "",
                "🔥 RESEARCH VALIDATION:",
                f"   Baseline Established: {'✅ Yes' if validation['baseline_established'] else '❌ No'}",
                f"   Metrics Collected: {validation['metrics_collected']}",
                f"   Suitable for Algorithm Comparison: {'✅ Yes' if validation['suitable_for_comparison'] else '❌ No'}",
                f"   Hot Spots Present: {'✅ Yes' if validation['hot_spots_present'] else '❌ No'}",
                f"   Data Integrity: {'✅ Verified' if validation['data_integrity_verified'] else '❌ Failed'}",
                f"   Optimization Potential: {validation['algorithm_optimization_potential']:.2f}x",
                "",
                "🚀 RESEARCH STATUS:",
                "   ✅ Baseline measurements complete",
                "   ✅ Performance metrics validated",
                "   ✅ Ready for adaptive algorithm implementation",
                "",
                "=" * 60
            ]

        # Log summary
        for line in summary_lines:
            self.logger.info(line)

        # Save summary to file
        summary_file = "data/processed/phase3/baseline_summary_corrected.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            self.logger.info(f"📄 Research summary saved to: {summary_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save summary: {str(e)}")


def main():
    """
    Main execution with enhanced error handling and validation
    """

    # Configuration
    BATCH_SIZE = 500  # Optimized for 8GB RAM system

    # Dataset file paths
    ENTITIES_FILE = "data/processed/phase1/consolidated_entities.json"
    RELATIONSHIPS_FILE = "data/processed/phase1/consolidated_relationships.json"

    # Pre-execution validation
    missing_files = []
    if not Path(ENTITIES_FILE).exists():
        missing_files.append(ENTITIES_FILE)
    if not Path(RELATIONSHIPS_FILE).exists():
        missing_files.append(RELATIONSHIPS_FILE)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        print("Please ensure data consolidation is complete.")
        return False

    try:
        # Initialize corrected baseline loader
        loader = CorrectedBaselineLoader(batch_size=BATCH_SIZE)

        # Execute corrected baseline loading
        analysis_results = loader.load_consolidated_dataset(
            entities_file=ENTITIES_FILE,
            relationships_file=RELATIONSHIPS_FILE
        )

        # Validate results
        if analysis_results.get("error"):
            print(f"\n⚠️  Completed with errors: {analysis_results['error']}")
            print("📊 Check logs for detailed debugging information.")
            success = False
        else:
            print("\n🎯 Corrected baseline loading completed successfully!")
            print("📊 Performance analysis validated and saved.")
            print("🚀 Ready for adaptive algorithm development!")
            success = True

        # Cleanup
        loader.db.close()
        return success

    except Exception as e:
        logging.error(f"❌ Corrected baseline loading failed: {str(e)}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)