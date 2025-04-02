# src/pattern_recognition.py
import logging
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime

from src.models import Evidence, Patterns, Pattern, EvidenceItem

class PatternAnalyzer:
    """Analyzes evidence to identify patterns and connections."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, evidence: Evidence) -> Patterns:
        """
        Analyze evidence to identify patterns.
        
        Args:
            evidence: Evidence object containing evidence items
            
        Returns:
            Patterns object containing identified patterns
        """
        self.logger.info("Starting pattern analysis")
        patterns = Patterns()
        
        if not evidence or not evidence.items:
            self.logger.warning("No evidence items to analyze")
            return patterns
        
        # Extract temporal patterns
        self.logger.info("Analyzing temporal patterns")
        temporal_patterns = self._analyze_temporal_patterns(evidence)
        for pattern in temporal_patterns:
            patterns.add_pattern(pattern)
        
        # Extract entity co-occurrence patterns
        self.logger.info("Analyzing entity co-occurrence patterns")
        entity_patterns = self._analyze_entity_patterns(evidence)
        for pattern in entity_patterns:
            patterns.add_pattern(pattern)
        
        # Build and analyze evidence graph
        self.logger.info("Building and analyzing evidence graph")
        graph_patterns = self._analyze_evidence_graph(evidence)
        for pattern in graph_patterns:
            patterns.add_pattern(pattern)
        
        self.logger.info(f"Identified {len(patterns.patterns)} patterns")
        return patterns
    
    def _analyze_temporal_patterns(self, evidence: Evidence) -> List[Pattern]:
        """Analyze temporal aspects of evidence items."""
        patterns = []
        
        # Extract timestamps from evidence
        timestamps = []
        evidence_ids = []
        
        for item in evidence.items:
            if hasattr(item, 'timestamp') and item.timestamp:
                timestamps.append(item.timestamp)
                evidence_ids.append(item.item_id)
        
        if not timestamps:
            return patterns
        
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = [timestamps[i] for i in sorted_indices]
        sorted_evidence_ids = [evidence_ids[i] for i in sorted_indices]
        
        # Look for clusters of events
        timestamp_array = np.array([(ts - datetime(1970, 1, 1)).total_seconds() for ts in sorted_timestamps]).reshape(-1, 1)
        clustering = DBSCAN(eps=3600, min_samples=2).fit(timestamp_array)  # Cluster events within 1 hour
        
        # Process clusters
        labels = clustering.labels_
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
            cluster_evidence_ids = [sorted_evidence_ids[i] for i in cluster_indices]
            cluster_timestamps = [sorted_timestamps[i] for i in cluster_indices]
            
            start_time = min(cluster_timestamps)
            end_time = max(cluster_timestamps)
            
            pattern = Pattern(
                type="temporal_cluster",
                description=f"Cluster of {len(cluster_indices)} related events between {start_time} and {end_time}",
                evidence_items=cluster_evidence_ids,
                confidence=0.7,  # Arbitrary confidence score
                metadata={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": (end_time - start_time).total_seconds()
                }
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_entity_patterns(self, evidence: Evidence) -> List[Pattern]:
        """Analyze entity co-occurrence patterns in evidence items."""
        patterns = []
        
        # Extract entities from evidence
        entity_items = {}  # Map from entity to list of evidence items
        
        for item in evidence.items:
            if 'entities' in item.metadata:
                for entity_type, entities in item.metadata['entities'].items():
                    for entity in entities:
                        key = f"{entity_type}:{entity}"
                        if key not in entity_items:
                            entity_items[key] = []
                        entity_items[key].append(item.item_id)
        
        # Find entities that appear in multiple evidence items
        for key, items in entity_items.items():
            if len(items) >= 2:  # Entity appears in at least 2 evidence items
                entity_type, entity = key.split(':', 1)
                
                pattern = Pattern(
                    type="entity_occurrence",
                    description=f"Entity '{entity}' ({entity_type}) appears in multiple evidence items",
                    evidence_items=items,
                    confidence=min(0.5 + (len(items) * 0.1), 0.9),  # Higher confidence for more occurrences
                    metadata={
                        "entity": entity,
                        "entity_type": entity_type,
                        "occurrence_count": len(items)
                    }
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_evidence_graph(self, evidence: Evidence) -> List[Pattern]:
        """Build and analyze a graph of evidence connections."""
        patterns = []
        
        # Create a graph where nodes are evidence items
        G = nx.Graph()
        
        # Add nodes
        for item in evidence.items:
            G.add_node(item.item_id, type=item.type)
        
        # Add edges between items with shared entities
        for i, item1 in enumerate(evidence.items):
            entities1 = set()
            if 'entities' in item1.metadata:
                for entity_type, entities in item1.metadata['entities'].items():
                    for entity in entities:
                        entities1.add(f"{entity_type}:{entity}")
            
            for item2 in evidence.items[i+1:]:
                entities2 = set()
                if 'entities' in item2.metadata:
                    for entity_type, entities in item2.metadata['entities'].items():
                        for entity in entities:
                            entities2.add(f"{entity_type}:{entity}")
                
                # Find shared entities
                shared_entities = entities1.intersection(entities2)
                if shared_entities:
                    G.add_edge(
                        item1.item_id, 
                        item2.item_id, 
                        weight=len(shared_entities),
                        shared_entities=list(shared_entities)
                    )
        
        # Find connected components
        for i, component in enumerate(nx.connected_components(G)):
            if len(component) >= 3:  # Only consider components with at least 3 items
                pattern = Pattern(
                    type="evidence_cluster",
                    description=f"Cluster of {len(component)} interconnected evidence items",
                    evidence_items=list(component),
                    confidence=0.8,
                    metadata={
                        "cluster_id": i,
                        "node_count": len(component),
                        "edge_count": G.subgraph(component).number_of_edges()
                    }
                )
                
                patterns.append(pattern)
        
        return patterns
