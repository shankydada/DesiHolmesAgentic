# src/theory_generation.py
import logging
from typing import List, Dict, Any
import random
from collections import Counter
import networkx as nx
from transformers import pipeline

from src.models import Evidence, Patterns, Theory, Pattern

class TheoryGenerator:
    """Generates theories based on evidence and identified patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize transformer model for text generation
        try:
            self.text_generator = pipeline("text-generation", model="gpt2")
            self.logger.info("Loaded text generation model")
        except Exception as e:
            self.logger.error(f"Failed to load text generation model: {e}")
            self.text_generator = None
    
    def generate(self, evidence: Evidence, patterns: Patterns) -> List[Theory]:
        """
        Generate theories based on evidence and patterns.
        
        Args:
            evidence: Evidence object containing evidence items
            patterns: Patterns object containing identified patterns
            
        Returns:
            List of Theory objects
        """
        self.logger.info("Starting theory generation")
        theories = []
        
        if not evidence or not evidence.items:
            self.logger.warning("No evidence items for theory generation")
            return theories
        
        if not patterns or not patterns.patterns:
            self.logger.warning("No patterns identified for theory generation")
            return theories
        
        # Generate theories using different methods
        self.logger.info("Generating theories based on evidence clusters")
        cluster_theories = self._generate_from_clusters(evidence, patterns)
        theories.extend(cluster_theories)
        
        self.logger.info("Generating theories based on temporal patterns")
        temporal_theories = self._generate_from_temporal_patterns(evidence, patterns)
        theories.extend(temporal_theories)
        
        self.logger.info("Generating theories based on entity connections")
        entity_theories = self._generate_from_entity_connections(evidence, patterns)
        theories.extend(entity_theories)
        
        self.logger.info(f"Generated {len(theories)} theories")
        return theories
    
    def _generate_from_clusters(self, evidence: Evidence, patterns: Patterns) -> List[Theory]:
        """Generate theories from evidence clusters."""
        theories = []
        
        # Get evidence cluster patterns
        cluster_patterns = patterns.get_patterns_by_type("evidence_cluster")
        
        for pattern in cluster_patterns:
            # Get the evidence items in this cluster
            evidence_items = [item for item in evidence.items if item.item_id in pattern.evidence_items]
            
            # Collect entities from the evidence items
            entities = Counter()
            for item in evidence_items:
                if 'entities' in item.metadata:
                    for entity_type, entity_list in item.metadata['entities'].items():
                        for entity in entity_list:
                            entities[f"{entity_type}:{entity}"] += 1
            
            # Use the most common entities for theory generation
            top_entities = entities.most_common(5)
            
            if top_entities:
                # Create connections to evidence
                evidence_connections = []
                for item in evidence_items:
                    connection = {
                        "evidence_id": item.item_id,
                        "strength": 0.7,  # Default strength
                        "relevance": "direct"  # All items in cluster are directly relevant
                    }
                    evidence_connections.append(connection)
                
                # Generate theory title and description
                title_entities = [entity.split(':', 1)[1] for entity, _ in top_entities[:3]]
                title = f"Connection between {', '.join(title_entities)}"
                
                description = self._generate_theory_description(evidence_items, top_entities)
                
                theory = Theory(
                    title=title,
                    description=description,
                    evidence_connections=evidence_connections,
                    confidence=pattern.confidence,
                    generation_method="evidence_cluster",
                    scores={
                        "evidence_support": 0.7,
                        "internal_consistency": 0.8,
                        "simplicity": 0.6
                    }
                )
                
                theories.append(theory)
        
        return theories
    
    def _generate_from_temporal_patterns(self, evidence: Evidence, patterns: Patterns) -> List[Theory]:
        """Generate theories from temporal patterns."""
        theories = []
        
        # Get temporal patterns
        temporal_patterns = patterns.get_patterns_by_type("temporal_cluster")
        
        for pattern in temporal_patterns:
            # Get the evidence items in this temporal cluster
            evidence_items = [item for item in evidence.items if item.item_id in pattern.evidence_items]
            
            if len(evidence_items) < 2:
                continue
            
            # Sort items by timestamp
            evidence_items.sort(key=lambda x: x.timestamp)
            
            # Create connections to evidence
            evidence_connections = []
            for item in evidence_items:
                connection = {
                    "evidence_id": item.item_id,
                    "strength": 0.7,
                    "relevance": "temporal_sequence"
                }
                evidence_connections.append(connection)
            
            # Generate theory title and description
            start_time = pattern.metadata.get("start_time", "unknown time")
            end_time = pattern.metadata.get("end_time", "unknown time")
            
            title = f"Sequence of events between {start_time} and {end_time}"
            
            description = "A series of connected events occurred in the following sequence:\n\n"
            for i, item in enumerate(evidence_items):
                item_desc = f"{item.type.capitalize()} evidence"
                if 'extracted_text' in item.metadata and item.metadata['extracted_text']:
                    item_desc += f": {item.metadata['extracted_text'][:100]}..."
                description += f"{i+1}. {item_desc} (at {item.timestamp})\n"
            
            description += "\nThese events appear to be causally related based on their temporal proximity."
            
            theory = Theory(
                title=title,
                description=description,
                evidence_connections=evidence_connections,
                confidence=pattern.confidence * 0.9,  # Slightly lower confidence for temporal theories
                generation_method="temporal_sequence",
                scores={
                    "evidence_support": 0.6,
                    "internal_consistency": 0.7,
                    "simplicity": 0.8
                }
            )
            
            theories.append(theory)
        
        return theories
    
    def _generate_from_entity_connections(self, evidence: Evidence, patterns: Patterns) -> List[Theory]:
        """Generate theories from entity connections across evidence."""
        theories = []
        
        # Get entity occurrence patterns
        entity_patterns = patterns.get_patterns_by_type("entity_occurrence")
        
        # Group patterns by entity type
        entity_type_patterns = {}
        for pattern in entity_patterns:
            entity_type = pattern.metadata.get("entity_type", "UNKNOWN")
            if entity_type not in entity_type_patterns:
                entity_type_patterns[entity_type] = []
            entity_type_patterns[entity_type].append(pattern)
        
        # Generate theories for each entity type
        for entity_type, type_patterns in entity_type_patterns.items():
            if entity_type in ["PERSON", "ORG", "GPE", "LOC"]:  # Focus on important entity types
                # Build a graph of entity connections
                G = nx.Graph()
                
                # Add entities as nodes
                for pattern in type_patterns:
                    entity = pattern.metadata.get("entity", "")
                    G.add_node(entity, 
                              count=pattern.metadata.get("occurrence_count", 0),
                              evidence=pattern.evidence_items)
                
                # Add edges between entities that share evidence
                for i, pattern1 in enumerate(type_patterns):
                    entity1 = pattern1.metadata.get("entity", "")
                    evidence1 = set(pattern1.evidence_items)
                    
                    for pattern2 in type_patterns[i+1:]:
                        entity2 = pattern2.metadata.get("entity", "")
                        evidence2 = set(pattern2.evidence_items)
                        
                        # Find shared evidence
                        shared = evidence1.intersection(evidence2)
                        if shared:
                            G.add_edge(entity1, entity2, weight=len(shared), shared_evidence=list(shared))
                
                # Find connected components
                for component in nx.connected_components(G):
                    if len(component) >= 2:  # Only consider components with at least 2 entities
                        subgraph = G.subgraph(component)
                        
                        # Collect all evidence items for this component
                        all_evidence = set()
                        for entity in component:
                            all_evidence.update(G.nodes[entity]["evidence"])
                        
                        # Create connections to evidence
                        evidence_connections = []
                        for evidence_id in all_evidence:
                            connection = {
                                "evidence_id": evidence_id,
                                "strength": 0.7,
                                "relevance": f"contains_{entity_type.lower()}"
                            }
                            evidence_connections.append(connection)
                        
                        # Generate theory title and description
                        title = f"Connection between {entity_type.title()} entities: {', '.join(component)}"
                        
                        description = f"The investigation reveals connections between the following {entity_type.lower()} entities:\n\n"
                        for entity in component:
                            count = G.nodes[entity]["count"]
                            description += f"- {entity} (appears in {count} pieces of evidence)\n"
                        
                        description += f"\nThese {entity_type.lower()} entities are connected through shared evidence, suggesting a possible relationship or network."
                        
                        theory = Theory(
                            title=title,
                            description=description,
                            evidence_connections=evidence_connections,
                            confidence=0.6 + (len(component) * 0.05),  # Higher confidence for more connected entities
                            generation_method="entity_network",
                            scores={
                                "evidence_support": 0.7,
                                "internal_consistency": 0.6,
                                "simplicity": 0.5 + (1 / len(component))  # Simpler for fewer entities
                            }
                        )
                        
                        theories.append(theory)
        
        return theories
    
    def _generate_theory_description(self, evidence_items, top_entities):
        """Generate a theory description based on evidence and entities."""
        # Extract entities for the description
        entity_strings = []
        for entity_key, count in top_entities:
            entity_type, entity = entity_key.split(':', 1)
            entity_strings.append(f"{entity} ({entity_type.lower()})")
        
        description = f"Analysis of {len(evidence_items)} related evidence items reveals connections between {', '.join(entity_strings)}.\n\n"
        
        # Add evidence summary
        description += "Key evidence includes:\n"
        for i, item in enumerate(evidence_items[:5]):  # Limit to top 5 items
            if item.type == "text":
                if 'extracted_text' in item.metadata and item.metadata['extracted_text']:
                    content = item.metadata['extracted_text'][:100] + "..."
                else:
                    content = "Text document"
            elif item.type == "image":
                if 'caption' in item.metadata:
                    content = f"Image: {item.metadata['caption']}"
                elif 'extracted_text' in item.metadata and item.metadata['extracted_text']:
                    content = f"Image with text: {item.metadata['extracted_text'][:100]}..."
                else:
                    content = "Image"
            else:
                content = f"{item.type} evidence"
            
            description += f"{i+1}. {content}\n"
        
        if len(evidence_items) > 5:
            description += f"... and {len(evidence_items) - 5} more evidence items.\n"
        
        # Use GPT-2 to enhance the description if available
        if self.text_generator:
            try:
                prompt = f"In a criminal investigation, we found connections between {', '.join(entity_strings)}. The evidence suggests that"
                generated_text = self.text_generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
                theory_extension = generated_text.replace(prompt, "").strip()
                
                description += f"\nTheory: {prompt} {theory_extension}"
            except Exception as e:
                self.logger.warning(f"Failed to generate theory description with GPT-2: {e}")
                description += "\nThe pattern of evidence suggests a potential connection that warrants further investigation."
        else:
            description += "\nThe pattern of evidence suggests a potential connection that warrants further investigation."
        
        return description
