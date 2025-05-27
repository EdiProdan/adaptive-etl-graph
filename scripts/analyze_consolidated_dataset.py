# scripts/analyze_consolidated_dataset.py
"""
Consolidated Dataset Analysis Framework
=====================================

Analyzes the consolidated Wikipedia dataset to validate enhancement effectiveness
and assess readiness for adaptive batching algorithm development.
Designed for master's thesis validation requirements.
"""

import json
import logging
from pathlib import Path
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import statistics


class ConsolidatedDatasetAnalyzer:
    """
    Analyzes consolidated entity/relationship datasets for thesis validation
    Focuses on connectivity patterns, hot spot distribution, and algorithm readiness
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # File paths
        self.config = {
            'entities_file': 'data/processed/phase1/consolidated_entities.json',
            'relationships_file': 'data/processed/phase1/consolidated_relationships.json',
            'consolidation_report': 'data/processed/phase1/consolidation_report.json',
            'analysis_output': 'data/processed/phase1/dataset_analysis_final.json'
        }

    def load_consolidated_data(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """Load consolidated datasets and consolidation report"""

        self.logger.info("Loading consolidated datasets...")

        with open(self.config['entities_file'], 'r', encoding='utf-8') as f:
            entities = json.load(f)

        with open(self.config['relationships_file'], 'r', encoding='utf-8') as f:
            relationships = json.load(f)

        with open(self.config['consolidation_report'], 'r', encoding='utf-8') as f:
            consolidation_report = json.load(f)

        self.logger.info(f"Loaded: {len(entities):,} entities, {len(relationships):,} relationships")
        return entities, relationships, consolidation_report

    def analyze_hot_spot_effectiveness(self, entities: List[Dict], relationships: List[Dict]) -> Dict:
        """
        Analyze hot spot distribution for adaptive batching algorithm validation
        Focus on entities that will create realistic database conflicts
        """
        self.logger.info("Analyzing hot spot effectiveness for algorithm validation...")

        # Count incoming references (targets that will cause write conflicts)
        target_frequency = Counter(rel['target'] for rel in relationships)

        # Identify internal page entities (actual content vs metadata)
        page_entities = {entity['name'] for entity in entities if entity.get('entity_type') == 'page'}

        # Categorize hot spots by conflict potential
        conflict_categories = {
            'extreme_conflict': [],  # >3000 references
            'high_conflict': [],  # 1000-3000 references
            'moderate_conflict': [],  # 100-1000 references
            'low_conflict': []  # 10-100 references
        }

        for entity_name, count in target_frequency.most_common(200):
            if entity_name in page_entities:  # Internal entities only
                if count >= 3000:
                    conflict_categories['extreme_conflict'].append((entity_name, count))
                elif count >= 1000:
                    conflict_categories['high_conflict'].append((entity_name, count))
                elif count >= 100:
                    conflict_categories['moderate_conflict'].append((entity_name, count))
                elif count >= 10:
                    conflict_categories['low_conflict'].append((entity_name, count))

        # Semantic categorization of hot spots
        semantic_hot_spots = self._categorize_semantic_hot_spots(conflict_categories)

        analysis = {
            'conflict_distribution': {
                category: len(entities) for category, entities in conflict_categories.items()
            },
            'conflict_entities': conflict_categories,
            'semantic_categorization': semantic_hot_spots,
            'algorithm_validation_readiness': {
                'sufficient_extreme_conflicts': len(conflict_categories['extreme_conflict']) >= 3,
                'sufficient_high_conflicts': len(conflict_categories['high_conflict']) >= 10,
                'diverse_conflict_spectrum': len(conflict_categories['moderate_conflict']) >= 50,
                'total_conflict_entities': sum(len(entities) for entities in conflict_categories.values())
            }
        }

        return analysis

    def _categorize_semantic_hot_spots(self, conflict_categories: Dict) -> Dict:
        """Categorize hot spots by semantic domain for cross-domain conflict analysis"""

        semantic_categories = {
            'bibliographic': ['ISBN', 'ISSN', 'DOI', 'PMID', 'arXiv'],
            'media': ['IMDb', 'TMDb', 'MusicBrainz'],
            'geographic': ['coordinates', 'location', 'place'],
            'temporal': ['century', 'year', 'period', 'era'],
            'institutional': ['authority', 'control', 'library', 'archive'],
            'biographical': ['people', 'person', 'biography'],
            'scientific': ['journal', 'research', 'science'],
            'cultural': ['museum', 'art', 'culture', 'heritage']
        }

        semantic_distribution = defaultdict(list)

        # Analyze all conflict entities
        all_conflict_entities = []
        for category_entities in conflict_categories.values():
            all_conflict_entities.extend(category_entities)

        for entity_name, count in all_conflict_entities:
            entity_lower = entity_name.lower()
            categorized = False

            for semantic_type, keywords in semantic_categories.items():
                if any(keyword.lower() in entity_lower for keyword in keywords):
                    semantic_distribution[semantic_type].append((entity_name, count))
                    categorized = True
                    break

            if not categorized:
                semantic_distribution['other'].append((entity_name, count))

        return dict(semantic_distribution)

    def analyze_graph_connectivity_patterns(self, relationships: List[Dict]) -> Dict:
        """
        Analyze graph connectivity patterns relevant to batch processing optimization
        """
        self.logger.info("Analyzing graph connectivity patterns...")

        # Source analysis (entities with many outgoing links)
        source_frequency = Counter(rel['source'] for rel in relationships)

        # Relationship type analysis
        relationship_types = Counter(rel['relationship_type'] for rel in relationships)

        # Connectivity distribution analysis
        outgoing_counts = list(source_frequency.values())
        incoming_counts = list(Counter(rel['target'] for rel in relationships).values())

        connectivity_analysis = {
            'outgoing_link_statistics': {
                'mean': statistics.mean(outgoing_counts),
                'median': statistics.median(outgoing_counts),
                'std_dev': statistics.stdev(outgoing_counts) if len(outgoing_counts) > 1 else 0,
                'max': max(outgoing_counts),
                'entities_with_high_outgoing': len([c for c in outgoing_counts if c > 500])
            },
            'incoming_link_statistics': {
                'mean': statistics.mean(incoming_counts),
                'median': statistics.median(incoming_counts),
                'std_dev': statistics.stdev(incoming_counts) if len(incoming_counts) > 1 else 0,
                'max': max(incoming_counts),
                'entities_with_high_incoming': len([c for c in incoming_counts if c > 100])
            },
            'relationship_type_distribution': dict(relationship_types),
            'graph_characteristics': {
                'total_edges': len(relationships),
                'unique_sources': len(source_frequency),
                'unique_targets': len(set(rel['target'] for rel in relationships)),
                'average_degree': len(relationships) / max(len(source_frequency), 1)
            }
        }

        return connectivity_analysis

    def assess_algorithm_validation_readiness(self, hot_spot_analysis: Dict,
                                              connectivity_analysis: Dict,
                                              consolidation_report: Dict) -> Dict:
        """
        Comprehensive assessment of dataset readiness for adaptive batching algorithm validation
        """
        self.logger.info("Assessing algorithm validation readiness...")

        # Extract key metrics
        dataset_metrics = consolidation_report['dataset_metrics']['consolidated_dataset']
        quality_metrics = consolidation_report['quality_analysis']

        # Validation criteria for thesis requirements
        validation_criteria = {
            'dataset_scale': {
                'entities_sufficient': dataset_metrics['entities'] >= 500000,
                'relationships_sufficient': dataset_metrics['relationships'] >= 2000000,
                'complexity_adequate': dataset_metrics['entities'] * dataset_metrics['relationships'] >= 1e12
            },
            'conflict_generation': {
                'extreme_conflicts_available': hot_spot_analysis['algorithm_validation_readiness'][
                    'sufficient_extreme_conflicts'],
                'conflict_diversity': hot_spot_analysis['algorithm_validation_readiness']['diverse_conflict_spectrum'],
                'cross_domain_conflicts': len(hot_spot_analysis['semantic_categorization']) >= 5
            },
            'connectivity_patterns': {
                'high_connectivity_entities': connectivity_analysis['incoming_link_statistics'][
                                                  'entities_with_high_incoming'] >= 20,
                'graph_density_appropriate': quality_metrics['relationship_metrics']['graph_density_indicator'] > 1e-6,
                'relationship_diversity': len(connectivity_analysis['relationship_type_distribution']) >= 2
            },
            'processing_feasibility': {
                'memory_manageable': dataset_metrics['entities'] < 2000000,  # <2M entities for reasonable memory usage
                'computation_tractable': dataset_metrics['relationships'] < 10000000,  # <10M relationships
                'batch_size_optimization_possible': hot_spot_analysis['conflict_distribution']['high_conflict'] >= 5
            }
        }

        # Overall readiness assessment
        category_scores = {}
        for category, criteria in validation_criteria.items():
            passed_criteria = sum(1 for criterion in criteria.values() if criterion)
            total_criteria = len(criteria)
            category_scores[category] = {
                'score': passed_criteria / total_criteria,
                'passed_criteria': passed_criteria,
                'total_criteria': total_criteria,
                'details': criteria
            }

        overall_score = sum(scores['score'] for scores in category_scores.values()) / len(category_scores)

        readiness_assessment = {
            'overall_readiness_score': overall_score,
            'category_assessments': category_scores,
            'thesis_validation_status': {
                'ready_for_neo4j_setup': overall_score >= 0.8,
                'ready_for_algorithm_development': overall_score >= 0.7,
                'suitable_for_thesis_validation': overall_score >= 0.6
            },
            'recommended_next_steps': self._generate_next_steps_recommendations(overall_score, category_scores)
        }

        return readiness_assessment

    def _generate_next_steps_recommendations(self, overall_score: float, category_scores: Dict) -> List[str]:
        """Generate specific recommendations based on readiness assessment"""

        recommendations = []

        if overall_score >= 0.8:
            recommendations.extend([
                "✅ Dataset fully ready for Neo4j setup and base graph loading",
                "✅ Proceed immediately to Week 3: Neo4j setup and schema design",
                "✅ Begin baseline performance measurement preparation"
            ])
        elif overall_score >= 0.7:
            recommendations.extend([
                "⚠️ Dataset mostly ready with minor optimization opportunities",
                "✅ Proceed to Neo4j setup with monitoring for performance bottlenecks",
                "⚠️ Consider additional hot spot analysis during initial testing"
            ])
        else:
            recommendations.append("❌ Dataset requires additional enhancement before algorithm development")

        # Specific category recommendations
        for category, assessment in category_scores.items():
            if assessment['score'] < 0.7:
                if category == 'conflict_generation':
                    recommendations.append("⚠️ Consider collecting additional high-connectivity pages")
                elif category == 'dataset_scale':
                    recommendations.append("⚠️ Dataset may be insufficient for robust algorithm validation")
                elif category == 'connectivity_patterns':
                    recommendations.append("⚠️ Graph connectivity may not provide sufficient complexity")

        return recommendations

    def execute_comprehensive_analysis(self) -> bool:
        """Execute complete dataset analysis workflow"""

        self.logger.info("🔍 Comprehensive Dataset Analysis Started")
        self.logger.info("=" * 60)

        try:
            # Load data
            entities, relationships, consolidation_report = self.load_consolidated_data()

            # Hot spot analysis
            hot_spot_analysis = self.analyze_hot_spot_effectiveness(entities, relationships)

            # Connectivity analysis
            connectivity_analysis = self.analyze_graph_connectivity_patterns(relationships)

            # Readiness assessment
            readiness_assessment = self.assess_algorithm_validation_readiness(
                hot_spot_analysis, connectivity_analysis, consolidation_report
            )

            # Compile comprehensive analysis
            comprehensive_analysis = {
                'analysis_metadata': {
                    'dataset_source': 'consolidated_wikipedia_enhancement',
                    'analysis_framework': 'thesis_validation_focused',
                    'entities_analyzed': len(entities),
                    'relationships_analyzed': len(relationships)
                },
                'hot_spot_analysis': hot_spot_analysis,
                'connectivity_analysis': connectivity_analysis,
                'readiness_assessment': readiness_assessment,
                'consolidation_summary': consolidation_report
            }

            # Save analysis
            with open(self.config['analysis_output'], 'w', encoding='utf-8') as f:
                json.dump(comprehensive_analysis, f, indent=2, ensure_ascii=False)

            # Report results
            self._report_analysis_results(comprehensive_analysis)

            return True

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return False

    def _report_analysis_results(self, analysis: Dict):
        """Generate comprehensive analysis report"""

        self.logger.info("=" * 60)
        self.logger.info("📊 DATASET ANALYSIS RESULTS")
        self.logger.info("=" * 60)

        # Hot spot summary
        hot_spot_data = analysis['hot_spot_analysis']
        self.logger.info(f"\n🔥 HOT SPOT ANALYSIS:")
        for category, count in hot_spot_data['conflict_distribution'].items():
            self.logger.info(f"  {category.replace('_', ' ').title()}: {count} entities")

        # Top conflict entities
        self.logger.info(f"\nTop Extreme Conflict Entities:")
        for entity, count in hot_spot_data['conflict_entities']['extreme_conflict'][:5]:
            self.logger.info(f"  • {entity}: {count:,} references")

        # Connectivity summary
        connectivity_data = analysis['connectivity_analysis']
        self.logger.info(f"\n📈 CONNECTIVITY ANALYSIS:")
        graph_chars = connectivity_data['graph_characteristics']
        self.logger.info(f"  Total Edges: {graph_chars['total_edges']:,}")
        self.logger.info(f"  Unique Sources: {graph_chars['unique_sources']:,}")
        self.logger.info(f"  Average Degree: {graph_chars['average_degree']:.1f}")

        # Readiness assessment
        readiness_data = analysis['readiness_assessment']
        self.logger.info(f"\n🎯 ALGORITHM VALIDATION READINESS:")
        self.logger.info(f"  Overall Score: {readiness_data['overall_readiness_score']:.2f}/1.00")

        validation_status = readiness_data['thesis_validation_status']
        for status, ready in validation_status.items():
            status_symbol = "✅" if ready else "❌"
            self.logger.info(f"  {status_symbol} {status.replace('_', ' ').title()}: {ready}")

        # Recommendations
        self.logger.info(f"\n🚀 RECOMMENDATIONS:")
        for recommendation in readiness_data['recommended_next_steps']:
            self.logger.info(f"  {recommendation}")

        self.logger.info(f"\n📁 Analysis saved to: {self.config['analysis_output']}")


def main():
    """Execute consolidated dataset analysis"""
    logging.basicConfig(level=logging.INFO)

    analyzer = ConsolidatedDatasetAnalyzer()
    success = analyzer.execute_comprehensive_analysis()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)