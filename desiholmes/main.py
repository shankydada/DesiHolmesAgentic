# main.py
import argparse
import logging
import os
from datetime import datetime

from src.evidence_collection import EvidenceCollector
from src.pattern_recognition import PatternAnalyzer
from src.theory_generation import TheoryGenerator
from src.theory_evaluation import TheoryEvaluator
from src.visualization import CrimeSceneVisualizer
from src.models import Case, Evidence

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"desiholmes_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='DesiHolmes AI Investigation System')
    parser.add_argument('--case_id', type=str, help='Case identifier')
    parser.add_argument('--evidence_dir', type=str, help='Directory containing evidence files')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("DesiHolmes AI Investigation System starting up")
    
    if not args.case_id:
        logger.error("No case ID provided")
        return
    
    if not args.evidence_dir or not os.path.isdir(args.evidence_dir):
        logger.error(f"Evidence directory not found: {args.evidence_dir}")
        return
    
    # Create a new case
    case = Case(case_id=args.case_id)
    
    # Initialize core components
    evidence_collector = EvidenceCollector()
    pattern_analyzer = PatternAnalyzer()
    theory_generator = TheoryGenerator()
    theory_evaluator = TheoryEvaluator()
    visualizer = CrimeSceneVisualizer()
    
    # Step 1: Collect evidence
    logger.info(f"Collecting evidence for case {args.case_id}")
    evidence = evidence_collector.collect(args.evidence_dir)
    case.add_evidence(evidence)
    
    # Step 2: Analyze patterns
    logger.info("Analyzing patterns in evidence")
    patterns = pattern_analyzer.analyze(evidence)
    case.set_patterns(patterns)
    
    # Step 3: Generate theories
    logger.info("Generating theories based on evidence and patterns")
    theories = theory_generator.generate(evidence, patterns)
    case.set_theories(theories)
    
    # Step 4: Evaluate theories
    logger.info("Evaluating and ranking theories")
    ranked_theories = theory_evaluator.evaluate(theories, evidence)
    case.set_ranked_theories(ranked_theories)
    
    # Step 5: Visualize best theory
    logger.info("Creating visualizations for best theory")
    best_theory = ranked_theories[0]
    visualizations = visualizer.visualize(best_theory, evidence)
    case.set_visualizations(visualizations)
    
    # Output results
    logger.info(f"Investigation complete for case {args.case_id}")
    logger.info(f"Best theory: {best_theory.title}")
    logger.info(f"Confidence score: {best_theory.confidence:.2f}")
    
    # Save case results
    output_dir = f"results/{args.case_id}"
    os.makedirs(output_dir, exist_ok=True)
    case.save(output_dir)
    
    return case

if __name__ == "__main__":
    main()