#!/usr/bin/env python3
"""
Dataset Pruning Script for Adaptive Batching Research
Author: Research Framework
Purpose: Generate focused datasets (1K, 2K, 4K relationships) while preserving conflict patterns
"""

import json
import random
from collections import defaultdict, Counter
from pathlib import Path


class DatasetPruner:
    def __init__(self, relationships_file, conflict_analysis_file, processed_pages_file):
        """Initialize with the three core data files."""
        self.relationships_file = Path(relationships_file)
        self.conflict_analysis_file = Path(conflict_analysis_file)
        self.processed_pages_file = Path(processed_pages_file)

        # Load data
        with open(self.relationships_file, 'r') as f:
            self.relationships_data = json.load(f)

        with open(self.conflict_analysis_file, 'r') as f:
            self.conflict_analysis = json.load(f)

        with open(self.processed_pages_file, 'r') as f:
            self.processed_pages = json.load(f)

        self.relationships = self.relationships_data.get('links_to', [])
        print(f"Loaded {len(self.relationships)} total relationships")

        # Create entity importance rankings
        self._analyze_entity_importance()

    def _analyze_entity_importance(self):
        """Create comprehensive entity importance rankings."""

        # Extract hub entities from conflict analysis
        self.hub_entities = set()
        self.hub_entities.update(self.conflict_analysis.get('moderate_conflict', {}).keys())
        self.hub_entities.update(self.conflict_analysis.get('high_conflict', {}).keys())
        self.hub_entities.update(self.conflict_analysis.get('extreme_conflict', {}).keys())

        print(f"Identified {len(self.hub_entities)} hub entities from conflict analysis")

        # Calculate relationship frequency for all entities
        entity_frequency = Counter()
        entity_similarity_scores = defaultdict(list)
        hub_involvement = Counter()

        for rel in self.relationships:
            entity_frequency[rel['from']] += 1
            entity_frequency[rel['to']] += 1

            # Track similarity scores
            entity_similarity_scores[rel['from']].append(rel['similarity'])
            entity_similarity_scores[rel['to']].append(rel['similarity'])

            # Track hub involvement
            if rel.get('involves_hub', False):
                hub_involvement[rel['from']] += 1
                hub_involvement[rel['to']] += 1

        # Create composite entity importance score
        self.entity_importance = {}

        for entity in entity_frequency:
            frequency_score = entity_frequency[entity]
            avg_similarity = sum(entity_similarity_scores[entity]) / len(entity_similarity_scores[entity])
            hub_bonus = 50 if entity in self.hub_entities else 0
            hub_involvement_bonus = hub_involvement.get(entity, 0) * 2

            # Composite importance score
            importance_score = (
                    frequency_score * 0.4 +  # Relationship frequency
                    avg_similarity * 100 * 0.3 +  # Average similarity strength
                    hub_bonus +  # Hub entity bonus
                    hub_involvement_bonus  # Hub relationship involvement
            )

            self.entity_importance[entity] = importance_score

        # Sort entities by importance
        self.top_entities = sorted(
            self.entity_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"Top 10 most important entities:")
        for entity, score in self.top_entities[:10]:
            is_hub = "🔥 HUB" if entity in self.hub_entities else ""
            print(f"  {entity}: {score:.2f} {is_hub}")

    def create_scenario_datasets(self):
        """Generate three focused datasets with different complexity levels."""

        scenarios = {
            'scenario_a': {'target_relationships': 1000, 'description': 'Development and initial validation'},
            'scenario_b': {'target_relationships': 2000, 'description': 'Core algorithm validation'},
            'scenario_c': {'target_relationships': 4000, 'description': 'Scalability demonstration'}
        }

        results = {}

        for scenario_name, config in scenarios.items():
            print(f"\n{'=' * 60}")
            print(f"GENERATING {scenario_name.upper()}")
            print(f"Target: {config['target_relationships']} relationships")
            print(f"Strategy: {config['description']}")
            print(f"{'=' * 60}")

            pruned_relationships = self._prune_relationships_intelligently(
                target_count=config['target_relationships'],
                scenario_name=scenario_name
            )

            # Generate conflict analysis for this scenario
            conflict_analysis = self._analyze_conflicts_for_scenario(pruned_relationships)

            # Create filtered processed pages
            filtered_pages = self._filter_processed_pages(pruned_relationships)

            results[scenario_name] = {
                'relationships': pruned_relationships,
                'conflict_analysis': conflict_analysis,
                'processed_pages': filtered_pages,
                'metadata': {
                    'total_relationships': len(pruned_relationships),
                    'unique_entities': len(
                        set([r['from'] for r in pruned_relationships] + [r['to'] for r in pruned_relationships])),
                    'hub_relationships': sum(1 for r in pruned_relationships if r.get('involves_hub', False)),
                    'avg_similarity': sum(r['similarity'] for r in pruned_relationships) / len(
                        pruned_relationships) if pruned_relationships else 0
                }
            }

            self._print_scenario_summary(scenario_name, results[scenario_name])

        return results

    def _prune_relationships_intelligently(self, target_count, scenario_name):
        """Intelligent relationship pruning strategy."""

        if target_count >= len(self.relationships):
            return self.relationships.copy()

        # Strategy varies by scenario
        if scenario_name == 'scenario_a':
            # High-precision: Focus on strongest semantic connections and all hub relationships
            return self._select_high_precision_relationships(target_count)
        elif scenario_name == 'scenario_b':
            # Intermediate: Balanced approach with moderate conflict preservation
            return self._select_balanced_relationships(target_count)
        else:  # scenario_c
            # Full complexity: Maximize diversity while preserving conflicts
            return self._select_diverse_relationships(target_count)

    def _select_high_precision_relationships(self, target_count):
        """Select relationships for Scenario A - CONFLICT-FIRST APPROACH."""

        # RESEARCH INSIGHT: We need conflicts to validate adaptive algorithms
        # Strategy: Start with known conflict entities, then add quality relationships

        conflict_entities = set(self.hub_entities)
        print(f"   Working with {len(conflict_entities)} conflict entities: {list(conflict_entities)[:5]}...")

        # Step 1: Get ALL relationships involving conflict entities
        conflict_relationships = []
        for rel in self.relationships:
            if rel['from'] in conflict_entities or rel['to'] in conflict_entities:
                conflict_relationships.append(rel)

        print(f"   Found {len(conflict_relationships)} relationships involving conflict entities")

        # Step 2: Sort by similarity but don't filter too aggressively
        conflict_relationships.sort(key=lambda x: x['similarity'], reverse=True)

        # Step 3: Take a substantial portion of conflict relationships
        conflict_allocation = min(len(conflict_relationships), int(target_count * 0.8))  # 80% conflicts
        selected_conflicts = conflict_relationships[:conflict_allocation]

        # Step 4: Fill remaining with high-quality non-conflict relationships
        remaining_slots = target_count - len(selected_conflicts)
        non_conflict_rels = [r for r in self.relationships
                             if r not in selected_conflicts and r['similarity'] >= 0.70]

        if remaining_slots > 0 and non_conflict_rels:
            non_conflict_rels.sort(key=lambda x: x['similarity'], reverse=True)
            selected_conflicts.extend(non_conflict_rels[:remaining_slots])

        print(f"   Final allocation: {len(selected_conflicts)} total ({conflict_allocation} conflict-related)")
        return selected_conflicts[:target_count]

    def _select_balanced_relationships(self, target_count):
        """Select relationships for Scenario B - ENHANCED CONFLICT PRESERVATION."""

        # Strategy: Deliberately include relationships that will create processing conflicts
        conflict_entities = set(self.hub_entities)

        # Categorize relationships by conflict potential
        high_conflict_rels = [r for r in self.relationships
                              if r['from'] in conflict_entities and r['to'] in conflict_entities]
        medium_conflict_rels = [r for r in self.relationships
                                if (r['from'] in conflict_entities) != (r['to'] in conflict_entities)]
        low_conflict_rels = [r for r in self.relationships
                             if r['from'] not in conflict_entities and r['to']]

        # Strategic allocation to ensure conflict distribution
        high_allocation = min(len(high_conflict_rels), int(target_count * 0.3))  # 30% high conflict
        medium_allocation = min(len(medium_conflict_rels), int(target_count * 0.5))  # 50% medium conflict
        low_allocation = target_count - high_allocation - medium_allocation  # 20% low conflict

        selected = []

        # Sample from each category to ensure conflict representation
        if high_conflict_rels:
            selected.extend(random.sample(high_conflict_rels, high_allocation))

        if medium_conflict_rels:
            available_medium = [r for r in medium_conflict_rels if r not in selected]
            take_medium = min(medium_allocation, len(available_medium))
            selected.extend(random.sample(available_medium, take_medium))

        if low_conflict_rels and low_allocation > 0:
            available_low = [r for r in low_conflict_rels if r not in selected]
            take_low = min(low_allocation, len(available_low))
            selected.extend(random.sample(available_low, take_low))

        print(
            f"   Scenario B allocation: {high_allocation} high-conflict + {len([r for r in selected if r in medium_conflict_rels])} medium-conflict + {len([r for r in selected if r in low_conflict_rels])} low-conflict")

        return selected[:target_count]

    def _select_diverse_relationships(self, target_count):
        """Select relationships for Scenario C - maximum diversity."""

        # Strategy: Systematic sampling across all relationship types
        # Ensure we capture the full spectrum of conflicts

        relationships_by_similarity = defaultdict(list)
        for rel in self.relationships:
            sim_bucket = int(rel['similarity'] * 10) / 10  # Round to nearest 0.1
            relationships_by_similarity[sim_bucket].append(rel)

        # Sample proportionally from each similarity bucket
        selected = []
        total_buckets = len(relationships_by_similarity)
        per_bucket = target_count // total_buckets
        remaining = target_count % total_buckets

        for sim_level in sorted(relationships_by_similarity.keys(), reverse=True):
            bucket_relationships = relationships_by_similarity[sim_level]
            take_count = per_bucket + (1 if remaining > 0 else 0)
            if remaining > 0:
                remaining -= 1

            if len(bucket_relationships) <= take_count:
                selected.extend(bucket_relationships)
            else:
                selected.extend(random.sample(bucket_relationships, take_count))

        return selected[:target_count]

    def _analyze_conflicts_for_scenario(self, relationships):
        """Generate conflict analysis with scale-adaptive thresholds - RESEARCH METHODOLOGY FIX."""

        entity_relationship_count = Counter()
        for rel in relationships:
            entity_relationship_count[rel['from']] += 1
            entity_relationship_count[rel['to']] += 1

        # RESEARCH INNOVATION: Scale-adaptive conflict thresholds
        # Base thresholds on dataset size, not absolute numbers
        total_relationships = len(relationships)

        # Calculate percentile-based thresholds (more methodologically sound)
        if total_relationships <= 1500:
            # Small datasets: lower absolute thresholds
            moderate_threshold = max(8, total_relationships // 200)  # ~0.5% of relationships
            high_threshold = max(15, total_relationships // 100)  # ~1% of relationships
            extreme_threshold = max(25, total_relationships // 50)  # ~2% of relationships
        elif total_relationships <= 3000:
            # Medium datasets: intermediate thresholds
            moderate_threshold = max(12, total_relationships // 150)
            high_threshold = max(25, total_relationships // 75)
            extreme_threshold = max(40, total_relationships // 40)
        else:
            # Large datasets: higher thresholds
            moderate_threshold = max(20, total_relationships // 100)
            high_threshold = max(50, total_relationships // 50)
            extreme_threshold = max(100, total_relationships // 25)

        print(f"   📊 Scale-adaptive thresholds for {total_relationships} relationships:")
        print(f"      Moderate: {moderate_threshold}+, High: {high_threshold}+, Extreme: {extreme_threshold}+")

        conflicts = {
            'low_conflict': {},
            'moderate_conflict': {},
            'high_conflict': {},
            'extreme_conflict': {}
        }

        # Debug: Print the top entities to understand distribution
        top_entities = entity_relationship_count.most_common(10)
        print(f"   🔍 Top entities by relationship count:")
        for entity, count in top_entities:
            conflict_level = "EXTREME" if count >= extreme_threshold else \
                "HIGH" if count >= high_threshold else \
                    "MODERATE" if count >= moderate_threshold else "LOW"
            print(f"      {entity}: {count} relationships [{conflict_level}]")

        for entity, count in entity_relationship_count.items():
            if count >= extreme_threshold:
                conflicts['extreme_conflict'][entity] = count
            elif count >= high_threshold:
                conflicts['high_conflict'][entity] = count
            elif count >= moderate_threshold:
                conflicts['moderate_conflict'][entity] = count
            else:
                conflicts['low_conflict'][entity] = count

        conflicts['total_entities'] = len(entity_relationship_count)
        conflicts['conflict_distribution'] = {
            'low': len(conflicts['low_conflict']),
            'moderate': len(conflicts['moderate_conflict']),
            'high': len(conflicts['high_conflict']),
            'extreme': len(conflicts['extreme_conflict'])
        }

        # Research validation feedback
        total_conflicts = conflicts['conflict_distribution']['moderate'] + \
                          conflicts['conflict_distribution']['high'] + \
                          conflicts['conflict_distribution']['extreme']

        if total_conflicts == 0:
            print(f"   ⚠️  WARNING: No conflicts detected even with adaptive thresholds")
            print(f"   ⚠️  Consider: Entity overlap may be insufficient for conflict generation")
        else:
            print(f"   ✅ SUCCESS: {total_conflicts} conflict entities detected")

        return conflicts

    def _filter_processed_pages(self, relationships):
        """Filter processed pages to only include those with entities in relationships."""

        # Extract all entities involved in relationships
        relationship_entities = set()
        for rel in relationships:
            relationship_entities.add(rel['from'])
            relationship_entities.add(rel['to'])

        # Filter pages to only include those with relevant entities
        filtered_pages = []
        for page in self.processed_pages:
            page_entities = {entity['name'] for entity in page.get('entities', [])}
            if relationship_entities & page_entities:  # If there's any overlap
                # Filter entities within the page too
                relevant_entities = [
                    entity for entity in page.get('entities', [])
                    if entity['name'] in relationship_entities
                ]
                filtered_page = page.copy()
                filtered_page['entities'] = relevant_entities
                filtered_pages.append(filtered_page)

        return filtered_pages

    def _print_scenario_summary(self, scenario_name, scenario_data):
        """Print a comprehensive summary of the scenario."""

        metadata = scenario_data['metadata']
        conflicts = scenario_data['conflict_analysis']['conflict_distribution']

        print(f"\n📊 {scenario_name.upper()} SUMMARY:")
        print(f"   Relationships: {metadata['total_relationships']:,}")
        print(f"   Unique Entities: {metadata['unique_entities']:,}")
        print(
            f"   Hub Relationships: {metadata['hub_relationships']:,} ({metadata['hub_relationships'] / metadata['total_relationships'] * 100:.1f}%)")
        print(f"   Avg Similarity: {metadata['avg_similarity']:.3f}")
        print(f"\n🎯 Conflict Distribution:")
        print(f"   Low: {conflicts['low']:,} entities")
        print(f"   Moderate: {conflicts['moderate']:,} entities")
        print(f"   High: {conflicts['high']:,} entities")
        print(f"   Extreme: {conflicts['extreme']:,} entities")

        if conflicts['moderate'] > 0 or conflicts['high'] > 0:
            print(f"\n⚡ Top Conflict Entities:")
            all_conflicts = {}
            all_conflicts.update(scenario_data['conflict_analysis']['moderate_conflict'])
            all_conflicts.update(scenario_data['conflict_analysis']['high_conflict'])
            all_conflicts.update(scenario_data['conflict_analysis']['extreme_conflict'])

            top_conflicts = sorted(all_conflicts.items(), key=lambda x: x[1], reverse=True)[:5]
            for entity, count in top_conflicts:
                print(f"     {entity}: {count} relationships")

    def export_scenarios(self, output_dir="data/output/experimental_scenarios"):
        """Export all scenarios to organized directory structure."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        scenarios = self.create_scenario_datasets()

        # Use development-focused naming to avoid confusion with original scenarios
        scenario_mapping = {
            'scenario_a': 'dev_a_precision',  # 1K relationships - High precision
            'scenario_b': 'dev_b_balanced',  # 2K relationships - Balanced complexity
            'scenario_c': 'dev_c_diverse'  # 4K relationships - Maximum diversity
        }

        for scenario_name, scenario_data in scenarios.items():
            # Use clearer directory names for development datasets
            export_name = scenario_mapping[scenario_name]
            scenario_dir = output_path / export_name
            scenario_dir.mkdir(exist_ok=True)

            # Export relationships
            relationships_export = {'links_to': scenario_data['relationships']}
            with open(scenario_dir / 'relationships.json', 'w') as f:
                json.dump(relationships_export, f, indent=2)

            # Export conflict analysis
            with open(scenario_dir / 'conflict_analysis.json', 'w') as f:
                json.dump(scenario_data['conflict_analysis'], f, indent=2)

            # Export processed pages
            with open(scenario_dir / 'processed_pages.json', 'w') as f:
                json.dump(scenario_data['processed_pages'], f, indent=2)

            # Export metadata summary
            with open(scenario_dir / 'metadata.json', 'w') as f:
                json.dump(scenario_data['metadata'], f, indent=2)

        print(f"\n✅ All scenarios exported to: {output_path}")
        print(f"   📁 scenario_a/ (1K relationships)")
        print(f"   📁 scenario_b/ (2K relationships)")
        print(f"   📁 scenario_c/ (4K relationships)")

        return scenarios


def main():
    """Main execution function."""
    print("🔬 DATASET PRUNING FOR ADAPTIVE BATCHING RESEARCH")
    print("=" * 60)

    # CONFIGURATION: Choose your source dataset based on research priorities
    # Option 1: Use Scenario 1 (smallest, highest semantic quality)
    # Option 2: Use Scenario 3 (largest, more synthetic but comprehensive)

    print("📊 DATASET SOURCE SELECTION:")
    print("   Scenario 1: High semantic quality, dense real conflicts")
    print("   Scenario 3: Large scale, broader relationship patterns")
    print()

    # RECOMMENDED: Start with Scenario 1 for semantic preservation
    source_scenario = "scenario_1"  # Change to "scenario_3" if preferred

    relationships_file = f"data/output/{source_scenario}/relationships.json"
    processed_pages_file = f"data/output/{source_scenario}/processed_pages.json"

    # Conflict analysis should match your source scenario
    conflict_analysis_file = f"data/output/{source_scenario}/conflict_analysis.json"
    # If conflict analysis is in root for scenario 1, use:
    # conflict_analysis_file = "conflict_analysis.json"

    print(f"📁 SELECTED SOURCE: {source_scenario.upper()}")
    print(f"   Relationships: {relationships_file}")
    print(f"   Conflicts: {conflict_analysis_file}")
    print(f"   Pages: {processed_pages_file}")
    print(f"\n📁 OUTPUT: data/output/experimental_scenarios/dev_[a|b|c]/")
    print(f"   Purpose: Rapid development and algorithm validation\n")

    try:
        pruner = DatasetPruner(relationships_file, conflict_analysis_file, processed_pages_file)
        scenarios = pruner.export_scenarios()

        print(f"\n🎉 SUCCESS: Three experimental scenarios generated!")
        print(f"   Ready for adaptive batching algorithm validation")
        print(f"   Estimated processing time per scenario: 15-30 minutes")
        print(f"   Total experimental time: ~2 hours (vs. 13 hours)")

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find required file: {e}")
        print("   Please ensure these files exist:")
        print(f"   - {relationships_file}")
        print(f"   - {conflict_analysis_file}")
        print(f"   - {processed_pages_file}")
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    main()