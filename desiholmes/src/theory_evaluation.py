# src/theory_evaluation.py
import logging
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

from src.models import Evidence, Theory

class TheoryEvaluator:
    """Evaluates and ranks theories based on evidence support and coherence."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define evaluation criteria weights
        self.criteria_weights = {
            "evidence_support": 0.4,
            "internal_consistency": 0.3,
            "simplicity": 0.2,
            "explanatory_power": 0.1
        }
    
    def evaluate(self, theories: List[Theory], evidence: Evidence) -> List[Theory]:
        """
        Evaluate and rank theories based on multiple criteria.
        
        Args:
            theories: List of Theory objects to evaluate
            evidence: Evidence object containing evidence items
            
        Returns:
            List of Theory objects sorted by overall score
        """
        self.logger.info(f"Evaluating {len(theories)} theories")
        
        if not theories:
            self.logger.warning("No theories to evaluate")
            return []
        
        # Evaluate each theory
        for theory in theories:
            self._evaluate_theory(theory, evidence)
        
        # Rank theories by overall score
        ranked_theories = sorted(theories, key=lambda t: t.confidence, reverse=True)
        
        # Log top theories
        self.logger.info("Top theories:")
        for i, theory in enumerate(ranked_theories[:3]):
            self.logger.info(f"{i+1}. {theory.title} (confidence: {theory.confidence:.2f})")
        
        return ranked_theories
    
    def _evaluate_theory(self, theory: Theory, evidence: Evidence):
        """Evaluate a single theory across multiple criteria."""
        scores = {}
        
        # Evaluate evidence support if not already scored
        if "evidence_support" not in theory.scores:
            scores["evidence_support"] = self._evaluate_evidence_support(theory, evidence)
        else:
            scores["evidence_support"] = theory.scores["evidence_support"]
        
        # Evaluate internal consistency if not already scored
        if "internal_consistency" not in theory.scores:
            scores["internal_consistency"] = self._evaluate_internal_consistency(theory)
        else:
            scores["internal_consistency"] = theory.scores["internal_consistency"]
        
        # Evaluate simplicity if not already scored
        if "simplicity" not in theory.scores:
            scores["simplicity"] = self._evaluate_simplicity(theory)
        else:
            scores["simplicity"] = theory.scores["simplicity"]
        
        # Evaluate explanatory power if not already scored
        if "explanatory_power" not in theory.scores:
            scores["explanatory_power"] = self._evaluate_explanatory_power(theory, evidence)
        else:
            scores["explanatory_power"] = theory.scores["explanatory_power"]
        
        # Update theory scores
        theory.scores.update(scores)
        
        # Calculate overall confidence score as weighted average
        overall_score = sum(scores[criterion] * weight 
                           for criterion, weight in self.criteria_weights.items())
        
        theory.confidence = overall_score
    
    def _evaluate_evidence_support(self, theory: Theory, evidence: Evidence) -> float:
        """Evaluate how well the theory is supported by evidence."""
        if not theory.evidence_connections:
            return 0.0
        
        # Calculate average strength of evidence connections
        avg_strength = np.mean([conn.get("strength", 0) for conn in theory.evidence_connections])
        
        # Calculate coverage of available evidence
        evidence_coverage = len(theory.evidence_connections) / max(len(evidence.items), 1)
        
        # Combine both metrics
        evidence_support = (avg_strength * 0.7) + (evidence_coverage * 0.3)
        
        return min(evidence_support, 1.0)  # Cap at 1.0
    
    def _evaluate_internal_consistency(self, theory: Theory) -> float:
        """Evaluate the internal consistency of the theory."""
        # This is a simplified placeholder implementation
        # A real implementation would check for contradictions in the theory
        
        # For now, we'll use a random score if one wasn't provided
        if theory.generation_method == "evidence_cluster":
            return 0.8  # Evidence clusters tend to be consistent
        elif theory.generation_method == "temporal_sequence":
            return 0.7  # Temporal sequences are usually consistent
        elif theory.generation_method == "entity_network":
            return 0.6  # Entity networks may have some inconsistencies
        else:
            return 0.5  # Default consistency score
    
    def _evaluate_simplicity(self, theory: Theory) -> float:
        """Evaluate the simplicity of the theory."""
        # Count the number of evidence connections
        num_connections = len(theory.evidence_connections)
        
        # Theories with fewer connections are simpler
        if num_connections <= 3:
            return 0.9  # Very simple
        elif num_connections <= 7:
            return 0.7  # Moderately simple
        elif num_connections <= 15:
            return 0.5  # Moderately complex
        else:
            return 0.3  # Very complex
    
    def _evaluate_explanatory_power(self, theory: Theory, evidence: Evidence) -> float:
        """Evaluate how much of the evidence the theory explains."""
        # Get the IDs of evidence items explained by this theory
        explained_evidence = set(conn["evidence_id"] for conn in theory.evidence_connections)
        
        # Get all evidence IDs
        all_evidence = set(item.item_id for item in evidence.items)
        
        # Calculate the proportion of evidence explained
        if not all_evidence:
            return 0.0
        
        proportion_explained = len(explained_evidence) / len(all_evidence)
        
        # Theories that explain more evidence have higher explanatory power
        if proportion_explained >= 0.8:
            return 0.9  # Excellent explanatory power
        elif proportion_explained >= 0.5:
            return 0.7  # Good explanatory power
        elif proportion_explained >= 0.3:
            return 0.5  # Moderate explanatory power
        else:
            return 0.3  # Poor explanatory power
