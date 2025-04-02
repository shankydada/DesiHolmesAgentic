# src/visualization.py
import logging
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.models import Theory, Evidence, Visualization

class CrimeSceneVisualizer:
    """Creates visualizations of theories and evidence connections."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def visualize(self, theory: Theory, evidence: Evidence) -> list[Visualization]:
        """
        Create visualizations for a theory.
        
        Args:
            theory: Theory object to visualize
            evidence: Evidence object containing evidence items
            
        Returns:
            List of Visualization objects
        """
        self.logger.info(f"Creating visualizations for theory: {theory.title}")
        visualizations = []
        
        # Create output directory
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create evidence network visualization
        self.logger.info("Creating evidence network visualization")
        try:
            evidence_network_path = self._create_evidence_network(theory, evidence, output_dir, timestamp)
            if evidence_network_path:
                viz = Visualization(
                    type="evidence_network",
                    title="Evidence Connection Network",
                    description="Network visualization showing connections between evidence items supporting the theory",
                    file_path=evidence_network_path
                )
                visualizations.append(viz)
        except Exception as e:
            self.logger.error(f"Failed to create evidence network: {e}")
        
        # Create timeline visualization
        self.logger.info("Creating timeline visualization")
        try:
            timeline_path = self._create_timeline(theory, evidence, output_dir, timestamp)
            if timeline_path:
                viz = Visualization(
                    type="timeline",
                    title="Evidence Timeline",
                    description="Timeline visualization of evidence items supporting the theory",
                    file_path=timeline_path
                )
                visualizations.append(viz)
        except Exception as e:
            self.logger.error(f"Failed to create timeline: {e}")
        
        # Create entity relationship visualization
        self.logger.info("Creating entity relationship visualization")
        try:
            entity_relationship_path = self._create_entity_relationships(theory, evidence, output_dir, timestamp)
            if entity_relationship_path:
                viz = Visualization(
                    type="entity_relationship",
                    title="Entity Relationships",
                    description="Visualization of relationships between key entities in the theory",
                    file_path=entity_relationship_path
                )
                visualizations.append(viz)
        except Exception as e:
            self.logger.error(f"Failed to create entity relationships: {e}")
        
        self.logger.info(f"Created {len(visualizations)} visualizations")
        return visualizations
    
    def _create_evidence_network(self, theory: Theory, evidence: Evidence, output_dir: str, timestamp: str) -> str:
        """Create a network visualization of evidence connections."""
        # Get evidence items relevant to this theory
        theory_evidence_ids = [conn["evidence_id"] for conn in theory.evidence_connections]
        evidence_items = [item for item in evidence.items if item.item_id in theory_evidence_ids]
        
        if not evidence_items:
            self.logger.warning("No evidence items for network visualization")
            return None
        
        # Create a graph
        G = nx.Graph()
        
        # Add evidence items as nodes
        for item in evidence_items:
            G.add_node(item.item_id, 
                      type=item.type, 
                      label=f"{item.type.capitalize()}: {item.item_id[:8]}")
        
        # Add connections between evidence items with shared entities
        for i, item1 in enumerate(evidence_items):
            entities1 = set()
            if 'entities' in item1.metadata:
                for entity_type, entities in item1.metadata['entities'].items():
                    for entity in entities:
                        entities1.add(f"{entity_type}:{entity}")
            
            for item2 in evidence_items[i+1:]:
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
                        label=f"{len(shared_entities)} shared entities"
                    )
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        
        # Set node colors based on evidence type
        node_colors = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'text':
                node_colors.append('skyblue')
            elif G.nodes[node]['type'] == 'image':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        # Set edge widths based on weight
        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Create network layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'))
        
        # Add title
        plt.title(f"Evidence Network: {theory.title}")
        plt.axis('off')
        
        # Save the visualization
        filename = f"evidence_network_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_timeline(self, theory: Theory, evidence: Evidence, output_dir: str, timestamp: str) -> str:
        """Create a timeline visualization of evidence."""
        # Get evidence items relevant to this theory
        theory_evidence_ids = [conn["evidence_id"] for conn in theory.evidence_connections]
        evidence_items = [item for item in evidence.items if item.item_id in theory_evidence_ids]
        
        # Filter items with timestamps
        timeline_items = [item for item in evidence_items if hasattr(item, 'timestamp') and item.timestamp]
        
        if not timeline_items:
            self.logger.warning("No evidence items with timestamps for timeline visualization")
            return None
        
        # Sort items by timestamp
        timeline_items.sort(key=lambda x: x.timestamp)
        
        # Extract data for visualization
        dates = [item.timestamp for item in timeline_items]
        labels = [f"{item.type.capitalize()} ({item.item_id[:8]})" for item in timeline_items]
        types = [item.type for item in timeline_items]
        
        # Create colors based on evidence type
        colors = []
        for item_type in types:
            if item_type == 'text':
                colors.append('royalblue')
            elif item_type == 'image':
                colors.append('green')
            else:
                colors.append('gray')
        
        # Create the timeline using plotly
        fig = go.Figure()
        
        # Add timeline points
        fig.add_trace(go.Scatter(
            x=dates,
            y=[1] * len(dates),
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                symbol='circle'
            ),
            text=labels,
            textposition='top center',
            hoverinfo='text'
        ))
        
        # Add connecting line
        fig.add_trace(go.Scatter(
            x=dates,
            y=[1] * len(dates),
            mode='lines',
            line=dict(
                color='gray',
                width=1,
                dash='dot'
            ),
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Timeline of Evidence: {theory.title}",
            xaxis=dict(
                title="Date/Time",
                showgrid=True,
                showticklabels=True,
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[0.5, 2]
            ),
            showlegend=False,
            height=400
        )
        
        # Save the visualization
        filename = f"timeline_{timestamp}.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        
        return filepath
    
    def _create_entity_relationships(self, theory: Theory, evidence: Evidence, output_dir: str, timestamp: str) -> str:
        """Create a visualization of entity relationships."""
        # Get evidence items relevant to this theory
        theory_evidence_ids = [conn["evidence_id"] for conn in theory.evidence_connections]
        evidence_items = [item for item in evidence.items if item.item_id in theory_evidence_ids]
        
        if not evidence_items:
            self.logger.warning("No evidence items for entity relationship visualization")
            return None
        
        # Extract entities
        entity_occurrences = {}  # Entity -> count
        entity_types = {}  # Entity -> type
        
        for item in evidence_items:
            if 'entities' in item.metadata:
                for entity_type, entities in item.metadata['entities'].items():
                    for entity in entities:
                        key = f"{entity_type}:{entity}"
                        if key not in entity_occurrences:
                            entity_occurrences[key] = 0
                            entity_types[key] = entity_type
                        entity_occurrences[key] += 1
        
        if not entity_occurrences:
            self.logger.warning("No entities found for relationship visualization")
            return None
        
        # Create a graph of entity co-occurrences
        G = nx.Graph()
        
        # Add entities as nodes
        for entity_key, count in entity_occurrences.items():
            _, entity = entity_key.split(':', 1)
            G.add_node(entity_key, 
                      label=entity, 
                      type=entity_types[entity_key], 
                      count=count)
        
        # Add edges between entities that co-occur in the same evidence
        for item in evidence_items:
            if 'entities' in item.metadata:
                # Get all entities in this evidence item
                item_entities = []
                for entity_type, entities in item.metadata['entities'].items():
                    for entity in entities:
                        item_entities.append(f"{entity_type}:{entity}")
                
                # Connect all pairs of entities
                for i, entity1 in enumerate(item_entities):
                    for entity2 in item_entities[i+1:]:
                        if G.has_edge(entity1, entity2):
                            G[entity1][entity2]['weight'] += 1
                        else:
                            G.add_edge(entity1, entity2, weight=1)
        
        # Remove nodes with no connections
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        if not G.nodes():
            self.logger.warning("No connected entities for relationship visualization")
            return None
        
        # Create the visualization
        plt.figure(figsize=(14, 12))
        
        # Set node colors based on entity type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            if node_type == 'PERSON':
                node_colors.append('lightcoral')
            elif node_type == 'ORG':
                node_colors.append('skyblue')
            elif node_type == 'GPE' or node_type == 'LOC':
                node_colors.append('lightgreen')
            elif node_type == 'DATE' or node_type == 'TIME':
                node_colors.append('khaki')
            else:
                node_colors.append('lightgray')
        
        # Set node sizes based on occurrence count
        node_sizes = [G.nodes[node]['count'] * 100 for node in G.nodes()]
        
        # Set edge widths based on weight
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        # Create network layout - use Kamada-Kawai for better layout with weighted edges
        pos = nx.kamada_kawai_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Person'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Organization'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Location'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='khaki', markersize=10, label='Date/Time'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Other')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        plt.title(f"Entity Relationships: {theory.title}")
        plt.axis('off')
        
        # Save the visualization
        filename = f"entity_relationships_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
