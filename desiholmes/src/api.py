# src/api.py
import logging
import os
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from pydantic import BaseModel

from src.evidence_collection import EvidenceCollector
from src.pattern_recognition import PatternAnalyzer
from src.theory_generation import TheoryGenerator
from src.theory_evaluation import TheoryEvaluator
from src.visualization import CrimeSceneVisualizer
from src.models import Case

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DesiHolmes AI Investigation System", 
             description="An AI system for investigating and analyzing evidence")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Mount static files
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Response models
class CaseResponse(BaseModel):
    case_id: str
    status: str
    message: str

class TheoryResponse(BaseModel):
    theory_id: str
    title: str
    description: str
    confidence: float

class VisualizationResponse(BaseModel):
    viz_id: str
    type: str
    title: str
    file_path: str

class InvestigationResponse(BaseModel):
    case_id: str
    evidence_count: int
    pattern_count: int
    theories: List[TheoryResponse]
    visualizations: List[VisualizationResponse]

# Store active cases
active_cases = {}

@app.post("/cases/create", response_model=CaseResponse)
async def create_case(title: str = Form(...), description: Optional[str] = Form("")):
    """Create a new investigation case."""
    case_id = str(uuid.uuid4())
    case_dir = os.path.join("uploads", case_id)
    os.makedirs(case_dir, exist_ok=True)
    
    case = Case(case_id=case_id, title=title, description=description)
    active_cases[case_id] = case
    
    return CaseResponse(
        case_id=case_id,
        status="created",
        message=f"Case '{title}' created successfully"
    )

@app.post("/cases/{case_id}/upload", response_model=CaseResponse)
async def upload_evidence(case_id: str, files: List[UploadFile] = File(...)):
    """Upload evidence files for a case."""
    if case_id not in active_cases:
        return JSONResponse(
            status_code=404,
            content={"message": f"Case {case_id} not found"}
        )
    
    case_dir = os.path.join("uploads", case_id)
    
    # Save uploaded files
    for file in files:
        file_path = os.path.join(case_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    return CaseResponse(
        case_id=case_id,
        status="uploaded",
        message=f"Uploaded {len(files)} evidence files"
    )

@app.post("/cases/{case_id}/investigate", response_model=CaseResponse)
async def investigate_case(case_id: str, background_tasks: BackgroundTasks):
    """Start the investigation process for a case."""
    if case_id not in active_cases:
        return JSONResponse(
            status_code=404,
            content={"message": f"Case {case_id} not found"}
        )
    
    case = active_cases[case_id]
    evidence_dir = os.path.join("uploads", case_id)
    
    # Start investigation in background
    background_tasks.add_task(run_investigation, case, evidence_dir)
    
    return CaseResponse(
        case_id=case_id,
        status="investigating",
        message="Investigation started"
    )

@app.get("/cases/{case_id}/results", response_model=InvestigationResponse)
async def get_results(case_id: str):
    """Get the results of an investigation."""
    if case_id not in active_cases:
        return JSONResponse(
            status_code=404,
            content={"message": f"Case {case_id} not found"}
        )
    
    case = active_cases[case_id]
    
    # Check if investigation is complete
    if not hasattr(case, 'ranked_theories') or not case.ranked_theories:
        return JSONResponse(
            status_code=400,
            content={"message": "Investigation is not complete yet"}
        )
    
    # Prepare response
    theories = []
    for theory in case.ranked_theories:
        theories.append(TheoryResponse(
            theory_id=theory.theory_id,
            title=theory.title,
            description=theory.description,
            confidence=theory.confidence
        ))
    
    visualizations = []
    for viz in case.visualizations:
        visualizations.append(VisualizationResponse(
            viz_id=viz.viz_id,
            type=viz.type,
            title=viz.title,
            file_path=viz.file_path
        ))
    
    return InvestigationResponse(
        case_id=case_id,
        evidence_count=len(case.evidence.items) if case.evidence else 0,
        pattern_count=len(case.patterns.patterns) if case.patterns else 0,
        theories=theories,
        visualizations=visualizations
    )

@app.get("/cases/{case_id}/visualization/{viz_id}")
async def get_visualization(case_id: str, viz_id: str):
    """Get a specific visualization file."""
    if case_id not in active_cases:
        return JSONResponse(
            status_code=404,
            content={"message": f"Case {case_id} not found"}
        )
    
    case = active_cases[case_id]
    
    # Find the visualization
    viz = next((v for v in case.visualizations if v.viz_id == viz_id), None)
    if not viz:
        return JSONResponse(
            status_code=404,
            content={"message": f"Visualization {viz_id} not found"}
        )
    
    # Return the file
    return FileResponse(viz.file_path)

async def run_investigation(case: Case, evidence_dir: str):
    """Run the investigation pipeline."""
    logger.info(f"Starting investigation for case {case.case_id}")
    
    # Initialize core components
    evidence_collector = EvidenceCollector()
    pattern_analyzer = PatternAnalyzer()
    theory_generator = TheoryGenerator()
    theory_evaluator = TheoryEvaluator()
    visualizer = CrimeSceneVisualizer()
    
    try:
        # Step 1: Collect evidence
        logger.info("Collecting evidence")
        evidence = evidence_collector.collect(evidence_dir)
        case.add_evidence(evidence)
        
        # Step 2: Analyze patterns
        logger.info("Analyzing patterns")
        patterns = pattern_analyzer.analyze(evidence)
        case.set_patterns(patterns)
        
        # Step 3: Generate theories
        logger.info("Generating theories")
        theories = theory_generator.generate(evidence, patterns)
        case.set_theories(theories)
        
        # Step 4: Evaluate theories
        logger.info("Evaluating theories")
        ranked_theories = theory_evaluator.evaluate(theories, evidence)
        case.set_ranked_theories(ranked_theories)
        
        # Step 5: Create visualizations for the best theory
        if ranked_theories:
            logger.info("Creating visualizations")
            best_theory = ranked_theories[0]
            visualizations = visualizer.visualize(best_theory, evidence)
            case.set_visualizations(visualizations)
        
        # Save results
        output_dir = os.path.join("results", case.case_id)
        case.save(output_dir)
        
        logger.info(f"Investigation completed for case {case.case_id}")
    except Exception as e:
        logger.error(f"Investigation failed: {e}")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
