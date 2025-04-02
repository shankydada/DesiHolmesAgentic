# src/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os
import uuid

@dataclass
class EvidenceItem:
    """Represents a single piece of evidence."""
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    source: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "item_id": self.item_id,
            "type": self.type,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class Evidence:
    """Collection of evidence items for a case."""
    items: List[EvidenceItem] = field(default_factory=list)
    
    def add_item(self, item: EvidenceItem):
        """Add an evidence item to the collection."""
        self.items.append(item)
    
    def get_items_by_type(self, item_type: str) -> List[EvidenceItem]:
        """Get all evidence items of a specific type."""
        return [item for item in self.items if item.type == item_type]
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "items": [item.to_dict() for item in self.items]
        }

@dataclass
class Pattern:
    """Represents a pattern identified in the evidence."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    evidence_items: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "pattern_id": self.pattern_id,
            "type": self.type,
            "description": self.description,
            "evidence_items": self.evidence_items,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

@dataclass
class Patterns:
    """Collection of patterns identified in the evidence."""
    patterns: List[Pattern] = field(default_factory=list)
    
    def add_pattern(self, pattern: Pattern):
        """Add a pattern to the collection."""
        self.patterns.append(pattern)
    
    def get_patterns_by_type(self, pattern_type: str) -> List[Pattern]:
        """Get all patterns of a specific type."""
        return [pattern for pattern in self.patterns if pattern.type == pattern_type]
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "patterns": [pattern.to_dict() for pattern in self.patterns]
        }

@dataclass
class Theory:
    """Represents a theory about the case."""
    theory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    evidence_connections: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    generation_method: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "theory_id": self.theory_id,
            "title": self.title,
            "description": self.description,
            "evidence_connections": self.evidence_connections,
            "confidence": self.confidence,
            "generation_method": self.generation_method,
            "scores": self.scores
        }

@dataclass
class Visualization:
    """Represents a visualization of a theory."""
    viz_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    title: str = ""
    description: str = ""
    file_path: str = ""
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "viz_id": self.viz_id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path
        }

@dataclass
class Case:
    """Represents an investigation case."""
    case_id: str
    title: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    evidence: Optional[Evidence] = None
    patterns: Optional[Patterns] = None
    theories: List[Theory] = field(default_factory=list)
    ranked_theories: List[Theory] = field(default_factory=list)
    visualizations: List[Visualization] = field(default_factory=list)
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence to the case."""
        self.evidence = evidence
    
    def set_patterns(self, patterns: Patterns):
        """Set the patterns identified in the evidence."""
        self.patterns = patterns
    
    def set_theories(self, theories: List[Theory]):
        """Set the generated theories."""
        self.theories = theories
    
    def set_ranked_theories(self, ranked_theories: List[Theory]):
        """Set the ranked theories."""
        self.ranked_theories = ranked_theories
    
    def set_visualizations(self, visualizations: List[Visualization]):
        """Set the visualizations."""
        self.visualizations = visualizations
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "case_id": self.case_id,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "patterns": self.patterns.to_dict() if self.patterns else None,
            "theories": [theory.to_dict() for theory in self.theories],
            "ranked_theories": [theory.to_dict() for theory in self.ranked_theories],
            "visualizations": [viz.to_dict() for viz in self.visualizations]
        }
    
    def save(self, output_dir: str):
        """Save the case to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save case metadata
        with open(os.path.join(output_dir, "case.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)