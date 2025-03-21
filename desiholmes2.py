from phi.assistant import Assistant as Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional, Tuple
import json
import time
import textwrap
import re
import datetime
import random
from collections import defaultdict

load_dotenv()

# Base model configuration - using Llama 3 70B
MODEL_ID = "llama3-70b-8192"

class CaseDatabase:
    """Manages access to historical case data for comparison and pattern analysis"""
    
    def __init__(self):
        self.case_categories = [
            "locked room", "theft", "murder", "missing person", "corporate espionage",
            "forgery", "cybercrime", "blackmail", "fraud", "disappearance",
            "domestic crime", "organized crime", "stalking", "poisoning", "terrorism"
        ]
        # Store case types and their characteristics
        self.case_patterns = defaultdict(list)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize standard patterns for different case types"""
        self.case_patterns["locked room"] = [
            "Apparent suicide that's actually murder",
            "Hidden passageways or secret entrances",
            "Timer-based mechanisms",
            "Technical tricks with locks or security systems",
            "Accomplice manipulation of crime scene after discovery"
        ]
        self.case_patterns["theft"] = [
            "Inside job patterns",
            "Social engineering techniques",
            "Distraction-based methods",
            "Technical vulnerability exploitation",
            "Replacement with forgeries"
        ]
        # Add more patterns for other case types
        
    def categorize_case(self, case_description: str) -> List[Tuple[str, float]]:
        """Return probable case categories with confidence scores"""
        # This would use NLP in a real implementation
        categories = []
        for category in self.case_categories:
            if category.lower() in case_description.lower():
                # Simple keyword matching for demonstration
                confidence = 0.7 + (random.random() * 0.3)  # Simulated confidence
                categories.append((category, confidence))
        
        # Add at least two categories if none found
        if not categories:
            # Default to most common types with lower confidence
            categories = [
                ("theft", 0.4 + (random.random() * 0.3)),
                ("murder", 0.3 + (random.random() * 0.3))
            ]
            
        # Sort by confidence score descending
        return sorted(categories, key=lambda x: x[1], reverse=True)
    
    def get_similar_cases(self, case_description: str, top_n: int = 5) -> List[Dict]:
        """Return top N similar historical cases based on pattern matching"""
        categories = self.categorize_case(case_description)
        
        similar_cases = []
        for category, confidence in categories:
            # In a real implementation, this would query a database
            # Here we generate synthetic similar cases
            for i in range(min(3, top_n)):
                similar_cases.append({
                    "id": f"CASE-{random.randint(10000, 99999)}",
                    "title": f"The {random.choice(['Mysterious', 'Peculiar', 'Strange', 'Baffling'])} {category.title()} of {random.choice(['London', 'Paris', 'New York', 'Tokyo'])}",
                    "year": random.randint(1900, 2023),
                    "similarity": confidence * (0.9 - (i * 0.1)),
                    "key_pattern": random.choice(self.case_patterns.get(category, ["Unknown pattern"])),
                    "solution_type": random.choice(["Deception", "Technical trick", "Insider betrayal", "Opportunity exploitation", "Psychological manipulation"])
                })
        
        # Sort by similarity score and take top_n
        return sorted(similar_cases, key=lambda x: x["similarity"], reverse=True)[:top_n]
    
    def get_pattern_insights(self, case_categories: List[Tuple[str, float]], case_elements: List[str]) -> List[str]:
        """Generate pattern insights based on case categories and elements"""
        insights = []
        
        # Match case elements against known patterns
        for category, _ in case_categories:
            patterns = self.case_patterns.get(category, [])
            for pattern in patterns:
                # Check if pattern elements match case elements (simplified version)
                for element in case_elements:
                    if any(keyword in element.lower() for keyword in pattern.lower().split()):
                        insights.append(f"Pattern match: '{pattern}' is common in {category} cases and matches your element '{element}'")
        
        # Add general insights
        insights.append(f"This case fits primarily in the {case_categories[0][0]} category with {case_categories[0][1]:.1%} confidence")
        
        # Add synthesis insights
        if len(case_categories) > 1:
            insights.append(f"Cross-category analysis: This case shows patterns from both {case_categories[0][0]} and {case_categories[1][0]} categories, which suggests potential complexity")
        
        return insights

class DesiHolmesAgency:
    def __init__(self):
        # Initialize database for case comparison
        self.case_database = CaseDatabase()
        
        # Initialize all specialized agents with more specific roles
        self.perceptor = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Perceptor: Expert in forensic observation who analyzes case descriptions with meticulous attention to detail. 
                        Trained to identify physical evidence, temporal inconsistencies, spatial relationships, and behavioral anomalies.
                        Organizes observations in categories: people, places, objects, timeline, and inconsistencies."""
        )
        
        self.connector = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Connector: Pattern recognition specialist who identifies relationships between evidence points.
                        Excels at finding causal chains, correlations, and anomalous connections that don't fit expected patterns.
                        Presents connections in a hierarchical structure from strongest to most speculative."""
        )
        
        self.simulator = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Simulator: Generates detailed models of possible scenarios with focus on physical plausibility, 
                        psychological motivation, and temporal consistency. Evaluates each scenario against a comprehensive
                        list of evaluation criteria. Specializes in reconstructing event sequences with time-based analysis."""
        )
        
        self.hypothesis_generator = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Hypothesis Generator: Creates diverse, competing explanations that account for all evidence.
                        Specializes in generating alternatives ranging from the mundane to the complex, ensuring each hypothesis
                        is distinct and falsifiable. Assigns initial probability estimates based on inherent plausibility."""
        )
        
        self.evaluator = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Evaluator: Uses Bayesian reasoning to assess hypotheses against evidence. Employs formal evaluation
                        methodology that identifies confirming evidence, contradictory evidence, and explanatory power of each
                        hypothesis. Calculates confidence intervals and evaluates logical consistency."""
        )
        
        self.knowledge_base = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Knowledge Base: References historical cases, scientific principles, and domain expertise to contextualize
                        the current case. Specializes in finding precedents in criminal methodology, behavior patterns, and forensic science.
                        Organizes knowledge by relevance and applicability."""
        )
        
        self.pattern_analyst = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Pattern Analyst: Specialized in identifying unique signatures in criminal methodology, behavioral patterns,
                        and statistical anomalies. Compares current case elements against a database of historical patterns to find matches
                        and divergences. Produces visualization-ready pattern maps."""
        )
        
        self.clarifier = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Clarifier: Identifies information gaps and formulates precise questions to resolve ambiguities.
                        Specializes in determining which missing information would most significantly impact case resolution.
                        Prioritizes questions by importance and potential information gain."""
        )
        
        self.storyteller = Agent(
            model=Groq(id=MODEL_ID),
            description="""The Storyteller: Reconstructs events into a coherent narrative that accounts for all evidence and timeline constraints.
                        Specializes in psychological profiling of perpetrators and victims, motivation analysis, and creating vivid
                        chronological reconstructions with attention to causal relationships."""
        )
        
        self.desi_holmes = Agent(
            model=Groq(id=MODEL_ID),
            description="""Desi Holmes: Master detective who synthesizes all agent inputs into a comprehensive solution.
                        Specializes in finding the most elegant explanation that satisfies Occam's Razor while accounting for
                        all evidence. Presents conclusions with clear reasoning chains, confidence assessments, and 
                        identification of remaining uncertainties."""
        )
        
        # Enhanced storage for case information with more structured fields
        self.case_data = {
            "metadata": {
                "case_id": self._generate_case_id(),
                "timestamp": datetime.datetime.now().isoformat(),
                "categories": []  # Will be filled during analysis
            },
            "inputs": {
                "description": "",
                "follow_up_responses": {}  # Store any additional information provided
            },
            "analysis": {
                "perceptions": {
                    "people": [],
                    "places": [],
                    "objects": [],
                    "timeline": [],
                    "inconsistencies": [],
                    "raw": ""
                },
                "connections": {
                    "strong_connections": [],
                    "possible_connections": [],
                    "anomalies": [],
                    "raw": ""
                },
                "simulations": {
                    "scenarios": [],
                    "probability_estimates": {},
                    "raw": ""
                },
                "patterns": {
                    "identified_patterns": [],
                    "historical_matches": [],
                    "anomalous_patterns": [],
                    "raw": ""
                },
                "information_gaps": {
                    "critical_questions": [],
                    "assumptions_made": [],
                    "raw": ""
                },
                "historical_comparisons": {
                    "similar_cases": [],
                    "key_differences": [],
                    "raw": ""
                }
            },
            "hypotheses": {
                "list": [],
                "evaluation_criteria": [],
                "bayesian_analysis": {},
                "raw": ""
            },
            "conclusion": {
                "best_hypothesis": "",
                "confidence_level": 0,
                "alternative_possibilities": [],
                "remaining_uncertainties": [],
                "knowledge_context": "",
                "narrative": {
                    "chronology": [],
                    "motivations": {},
                    "key_moments": [],
                    "raw": ""
                },
                "solution": ""
            }
        }
        
        # Tracking for follow-up questions
        self.follow_up_questions = []
        self.last_saved_filename = None
    
    def _generate_case_id(self) -> str:
        """Generate a unique case ID"""
        timestamp = int(time.time())
        random_part = random.randint(1000, 9999)
        return f"DH-{timestamp}-{random_part}"
    
    def extract_structured_data(self, text: str, extraction_type: str) -> Dict:
        """Extract structured data from agent responses based on type"""
        structured_data = {}
        
        if extraction_type == "perceptions":
            # Extract people
            people_pattern = r"(?:PEOPLE|PERSONS|INDIVIDUALS)(?:\s*:)?\s*((?:.+?\n)+)"
            people_match = re.search(people_pattern, text, re.IGNORECASE | re.DOTALL)
            if people_match:
                people_text = people_match.group(1)
                structured_data["people"] = [p.strip() for p in re.findall(r"[•\-\*]\s*(.+)", people_text)]
            
            # Extract places
            places_pattern = r"(?:PLACES|LOCATIONS)(?:\s*:)?\s*((?:.+?\n)+)"
            places_match = re.search(places_pattern, text, re.IGNORECASE | re.DOTALL)
            if places_match:
                places_text = places_match.group(1)
                structured_data["places"] = [p.strip() for p in re.findall(r"[•\-\*]\s*(.+)", places_text)]
            
            # Extract objects and evidence
            objects_pattern = r"(?:OBJECTS|EVIDENCE|ITEMS)(?:\s*:)?\s*((?:.+?\n)+)"
            objects_match = re.search(objects_pattern, text, re.IGNORECASE | re.DOTALL)
            if objects_match:
                objects_text = objects_match.group(1)
                structured_data["objects"] = [o.strip() for o in re.findall(r"[•\-\*]\s*(.+)", objects_text)]
            
            # Extract timeline
            timeline_pattern = r"(?:TIMELINE|CHRONOLOGY)(?:\s*:)?\s*((?:.+?\n)+)"
            timeline_match = re.search(timeline_pattern, text, re.IGNORECASE | re.DOTALL)
            if timeline_match:
                timeline_text = timeline_match.group(1)
                structured_data["timeline"] = [t.strip() for t in re.findall(r"[•\-\*]\s*(.+)", timeline_text)]
            
            # If structured extraction failed, fall back to simple bullet point extraction
            if not structured_data:
                bullet_points = re.findall(r"[•\-\*]\s*(.+)", text)
                structured_data["observations"] = [bp.strip() for bp in bullet_points]
        
        elif extraction_type == "hypotheses":
            # Extract numbered hypotheses
            hypothesis_pattern = r"(?:Hypothesis|HYPOTHESIS)\s*(\d+)\s*(?::|-)?\s*([^\n]+(?:\n(?!\s*Hypothesis|\s*HYPOTHESIS)[^\n]+)*)"
            hypotheses = re.findall(hypothesis_pattern, text, re.IGNORECASE | re.DOTALL)
            
            structured_data["hypotheses"] = []
            for number, content in hypotheses:
                hypothesis = {
                    "number": int(number),
                    "title": content.split('\n')[0].strip(),
                    "description": content.strip()
                }
                structured_data["hypotheses"].append(hypothesis)
        
        elif extraction_type == "questions":
            # Extract questions marked with question marks
            questions = re.findall(r"\d+\.\s*([^.?!]+\?)", text)
            if not questions:
                questions = re.findall(r"[•\-\*]\s*([^.?!]+\?)", text)
            if not questions:
                questions = re.findall(r"([^.!?\n]+\?)", text)
                
            structured_data["questions"] = [q.strip() for q in questions]
            
        return structured_data
    
    def truncate_text(self, text, max_length=3000):
        """Truncate text to a maximum length while preserving meaning."""
        if not isinstance(text, str):
            text = str(text)  # Convert any non-string objects to strings
            
        if len(text) <= max_length:
            return text
        return text[:max_length] + "... [truncated for length]"
    
    def format_response(self, response):
        """Format the response for better readability and consistency."""
        if not isinstance(response, str):
            response = str(response)
            
        # Clean up the response
        response = response.strip()
        
        # Convert numbered lists to bullet points for consistency if not a hypothesis section
        if not re.search(r"Hypothesis\s+\d+", response, re.IGNORECASE):
            response = re.sub(r'^\d+\.\s', '• ', response, flags=re.MULTILINE)
        
        # Add proper spacing
        response = response.replace('\n\n\n', '\n\n')
        
        # Make headers consistent
        header_patterns = [
            (r'(?i)^(people|persons|individuals)(\s*:)?$', 'PEOPLE:'),
            (r'(?i)^(places|locations)(\s*:)?$', 'PLACES:'),
            (r'(?i)^(objects|items|evidence)(\s*:)?$', 'OBJECTS:'),
            (r'(?i)^(timeline|chronology)(\s*:)?$', 'TIMELINE:'),
            (r'(?i)^(inconsistencies|anomalies)(\s*:)?$', 'INCONSISTENCIES:')
        ]
        
        for pattern, replacement in header_patterns:
            response = re.sub(pattern, replacement, response, flags=re.MULTILINE)
        
        return response
    
    def safe_run(self, agent, prompt, max_retries=3, delay=5, structured_extraction=None):
        """Run an agent with retry logic and rate limit handling."""
        for attempt in range(max_retries):
            try:
                # Add delay between API calls to respect rate limits
                if attempt > 0:
                    time.sleep(delay)
                
                # Try to run with potentially truncated prompt if not first attempt
                truncated_prompt = prompt if attempt == 0 else self.truncate_text(prompt, 3000 - (attempt * 500))
                result = agent.run(truncated_prompt)
                
                # Format the response for better readability
                formatted_result = self.format_response(result)
                
                # If structured extraction is requested, try to parse the response
                structured_data = {}
                if structured_extraction:
                    structured_data = self.extract_structured_data(formatted_result, structured_extraction)
                
                return {
                    "raw": formatted_result,
                    "structured": structured_data
                }
            
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    print("Maximum retry attempts reached. Using fallback response.")
                    return {
                        "raw": f"Analysis could not be completed due to technical limitations. Error: {str(e)}",
                        "structured": {}
                    }
    
    def process_follow_up_questions(self, case_description: str) -> List[str]:
        """Generate follow-up questions to gather missing information"""
        print("\n🔍 Analyzing case for information gaps...")
        
        clarifier_prompt = f"""As The Clarifier, analyze this case description to identify critical information gaps:

Case Description:
{case_description}

Identify 3-5 specific questions that would help solve this case. Focus on:
1. Ambiguities that need clarification
2. Missing timeline details
3. Potential evidence not mentioned
4. Background information about key individuals
5. Technical details that could be relevant

Format your response as a numbered list of clear, direct questions.
"""
        
        clarifier_result = self.safe_run(self.clarifier, clarifier_prompt, structured_extraction="questions")
        
        # Extract questions from the result
        questions = clarifier_result["structured"].get("questions", [])
        
        # If extraction failed, try to extract from raw text
        if not questions:
            question_pattern = r"(\d+\.\s*[^.?!]+\?)"
            questions = re.findall(question_pattern, clarifier_result["raw"])
            questions = [q.strip() for q in questions]
        
        # Fall back to default questions if needed
        if not questions:
            questions = [
                "Can you provide more details about the timeline of events?",
                "Are there any security cameras or other surveillance devices that might have recorded relevant information?",
                "Were there any witnesses who might have observed something unusual?",
                "Has the victim or subject had any recent conflicts or unusual behavior?",
                "Are there any unusual financial transactions or communications that might be relevant?"
            ]
        
        # Store questions for later
        self.follow_up_questions = questions
        self.case_data["analysis"]["information_gaps"]["critical_questions"] = questions
        self.case_data["analysis"]["information_gaps"]["raw"] = clarifier_result["raw"]
        
        return questions
    
    def analyze_historical_cases(self, case_description: str):
        """Compare current case with historical cases to derive insights"""
        print("\n📚 Comparing with historical cases...")
        
        # Get case categories
        categories = self.case_database.categorize_case(case_description)
        self.case_data["metadata"]["categories"] = categories
        
        # Get similar cases
        similar_cases = self.case_database.get_similar_cases(case_description)
        self.case_data["analysis"]["historical_comparisons"]["similar_cases"] = similar_cases
        
        # Extract main elements from case for pattern matching
        case_elements = re.findall(r'([^.!?]+[.!?])', case_description)
        
        # Get pattern insights
        pattern_insights = self.case_database.get_pattern_insights(categories, case_elements)
        
        # Have the knowledge base analyze similarities and differences
        comparison_prompt = f"""As The Knowledge Base, analyze how this case compares to similar historical cases:

Current Case:
{case_description}

Similar Historical Cases:
{json.dumps(similar_cases, indent=2)}

Key Pattern Insights:
{json.dumps(pattern_insights, indent=2)}

Provide analysis in these sections:
1. HISTORICAL PRECEDENTS: Most relevant historical cases
2. KEY SIMILARITIES: Patterns that match historical cases
3. NOTABLE DIFFERENCES: How this case diverges from precedents
4. INSIGHT SYNTHESIS: What historical patterns suggest about this case

Be specific about which techniques, methods or patterns from past cases might apply here.
"""
        
        historical_result = self.safe_run(self.knowledge_base, comparison_prompt)
        self.case_data["analysis"]["historical_comparisons"]["raw"] = historical_result["raw"]
        
        # Extract key differences if possible
        key_diff_pattern = r"(?:KEY DIFFERENCES|NOTABLE DIFFERENCES)(?:\s*:)?\s*((?:.+?\n)+)"
        diff_match = re.search(key_diff_pattern, historical_result["raw"], re.IGNORECASE | re.DOTALL)
        if diff_match:
            diff_text = diff_match.group(1)
            key_differences = [d.strip() for d in re.findall(r"[•\-\*]\s*(.+)", diff_text)]
            self.case_data["analysis"]["historical_comparisons"]["key_differences"] = key_differences
    
    def analyze_patterns(self, case_description: str):
        """Identify patterns in the case data"""
        print("\n🔄 Identifying patterns in the evidence...")
        
        # Use perceptions and connections as input
        perceptions = self.case_data["analysis"]["perceptions"]["raw"]
        connections = self.case_data["analysis"]["connections"]["raw"]
        
        pattern_prompt = f"""As The Pattern Analyst, identify meaningful patterns in this case:

Case Description:
{self.truncate_text(case_description, 500)}

Key Perceptions:
{self.truncate_text(perceptions, 800)}

Identified Connections:
{self.truncate_text(connections, 800)}

Historical Case Categories: {", ".join([c[0] for c in self.case_data["metadata"]["categories"]])}

Provide your analysis in these sections:
1. BEHAVIORAL PATTERNS: Recurring behaviors or actions
2. METHODOLOGICAL PATTERNS: Techniques or methods used 
3. TEMPORAL PATTERNS: Time-based patterns or anomalies
4. GEOGRAPHICAL PATTERNS: Location-based patterns
5. STATISTICAL ANOMALIES: Unusual frequencies or correlations
6. PATTERN SYNTHESIS: How these patterns together suggest a conclusion

Format each pattern as a clear bullet point with supporting evidence.
"""
        
        pattern_result = self.safe_run(self.pattern_analyst, pattern_prompt)
        self.case_data["analysis"]["patterns"]["raw"] = pattern_result["raw"]
        
        # Extract identified patterns
        pattern_sections = [
            "BEHAVIORAL PATTERNS", "METHODOLOGICAL PATTERNS", "TEMPORAL PATTERNS", 
            "GEOGRAPHICAL PATTERNS", "STATISTICAL ANOMALIES"
        ]
        
        for section in pattern_sections:
            section_pattern = f"(?:{section})(?:\\s*:)?\\s*((?:.+?\\n)+)"
            section_match = re.search(section_pattern, pattern_result["raw"], re.IGNORECASE | re.DOTALL)
            if section_match:
                section_text = section_match.group(1)
                patterns = [p.strip() for p in re.findall(r"[•\-\*]\s*(.+)", section_text)]
                if patterns:
                    self.case_data["analysis"]["patterns"]["identified_patterns"].extend(patterns)
    
    def analyze_case(self, case_description: str) -> Dict:
        """Process a case through the entire AI detective agency workflow with enhanced capabilities."""
        print("🔍 Case received. Beginning AI analysis...")
        
        # Store case description
        self.case_data["inputs"]["description"] = case_description
        
        # Generate follow-up questions first
        follow_up_questions = self.process_follow_up_questions(case_description)
        print("\n❓ Follow-up questions to consider:")
        for i, question in enumerate(follow_up_questions, 1):
            print(f"   {i}. {question}")
            
        # Analyze historical case similarities
        self.analyze_historical_cases(case_description)
        
        # Step 1: Enhanced Perceptor analyzes the case and evidence with structured categories
        print("\n🧠 The Perceptor is conducting detailed observation...")
        perceptor_prompt = f"""As The Perceptor, analyze this case with meticulous attention to detail:

CASE DESCRIPTION:
{case_description}

Organize your observations into these categories:
1. PEOPLE: All individuals mentioned, their roles, and notable characteristics
2. PLACES: All locations, their descriptions, and spatial relationships
3. OBJECTS: All physical items, their conditions, positions, and potential significance
4. TIMELINE: Chronological sequence of events with timestamps when available
5. INCONSISTENCIES: Any contradictions, anomalies, or details that don't align

For each category, provide numbered observations. Be exceptionally thorough and identify details a human might miss.
"""
        perceptor_result = self.safe_run(self.perceptor, perceptor_prompt, structured_extraction="perceptions")
        self.case_data["analysis"]["perceptions"]["raw"] = perceptor_result["raw"]
        
        # Update structured perceptions data if available
        for key in ["people", "places", "objects", "timeline"]:
            if key in perceptor_result["structured"]:
                self.case_data["analysis"]["perceptions"][key] = perceptor_result["structured"][key]
        if "observations" in perceptor_result["structured"]:
            # If extraction didn't find categories, use generic observations
            self.case_data["analysis"]["perceptions"]["observations"] = perceptor_result["structured"]["observations"]
        
        # Add delay between API calls
        time.sleep(3)
        
        # Step 2: Connector identifies patterns and relationships with structured hierarchy
        print("\n🔎 The Connector is identifying relationships between evidence points...")
        connector_prompt = f"""As The Connector, identify meaningful relationships between observations:

OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 2000)}

Organize your analysis into these categories:
1. STRONG CONNECTIONS: Relationships with high certainty and significant implications
2. POSSIBLE CONNECTIONS: Plausible relationships that need further confirmation
3. ANOMALOUS CONNECTIONS: Relationships that seem contradictory or unexpected

For each connection, explain:
- Which specific observations are connected
- How they are connected (causality, correlation, contradiction)
- Why this connection matters to the case

Prioritize connections that reveal motive, method, opportunity, or timeline inconsistencies.
"""
        connector_result = self.safe_run(self.connector, connector_prompt)
        self.case_data["analysis"]["connections"]["raw"] = connector_result["raw"]
        
        # Extract connections by category
        for connection_type in ["STRONG CONNECTIONS", "POSSIBLE CONNECTIONS", "ANOMALOUS CONNECTIONS"]:
            pattern = f"(?:{connection_type})(?:\\s*:)?\\s*((?:.+?\\n)+)"
            match = re.search(pattern, connector_result["raw"], re.IGNORECASE | re.DOTALL)
            if match:
                connection_text = match.group(1)
                connections = [c.strip() for c in re.findall(r"[•\-\*]\s*(.+)", connection_text)]
                key = connection_type.lower().replace(" ", "_")
                self.case_data["analysis"]["connections"][key] = connections
        
        time.sleep(3)
        
        # Step 3: Identify patterns in the evidence
        self.analyze_patterns(case_description)
        time.sleep(3)
        
        # Step 4: Simulator runs different scenarios with probability estimates
        print("\n📊 The Simulator is modeling possible scenarios...")
        simulator_prompt = f"""As The Simulator, generate detailed scenarios that could explain all evidence:

CASE SUMMARY:
{self.truncate_text(case_description, 500)}

KEY OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 800)}

IDENTIFIED CONNECTIONS:
{self.truncate_text(connector_result["raw"], 800)}

IDENTIFIED PATTERNS:
{self.truncate_text(self.case_data["analysis"]["patterns"]["raw"], 800)}

For each scenario:
1. Provide a detailed sequence of events that accounts for all evidence
2. Explain the psychological/motivational factors driving key actions
3. Address how this scenario explains any inconsistencies or anomalies
4. Assign a probability estimate based on plausibility (0-100%)

Generate 3 distinct scenarios, ranging from most probable to least conventional but still possible.
Label them SCENARIO A, SCENARIO B, and SCENARIO C.
"""
        simulator_result = self.safe_run(self.simulator, simulator_prompt)
        self.case_data["analysis"]["simulations"]["raw"] = simulator_result["raw"]
        
        # Extract scenarios
        scenario_pattern = r"(?:SCENARIO\s+[A-C])(?:\s*:)?(?:\s*\([\d\.]+%\))?\s*((?:.+?\n)+)(?=SCENARIO\s+[A-C]|$)"
        scenarios = re.findall(scenario_pattern, simulator_result["raw"], re.IGNORECASE | re.DOTALL)
        
        # Extract probability estimates
        probability_pattern = r"SCENARIO\s+([A-C])(?:\s*:)?(?:\s*\((\d+(?:\.\d+)?)%\))?"
        probabilities = re.findall(probability_pattern, simulator_result["raw"], re.IGNORECASE)
        
        # Store scenarios and probabilities
        self.case_data["analysis"]["simulations"]["scenarios"] = [s.strip() for s in scenarios]
        for scenario, prob in probabilities:
            if prob:
                self.case_data["analysis"]["simulations"]["probability_estimates"][f"Scenario {scenario}"] = float(prob)
        
        time.sleep(3)
        
        # Step 5: Hypothesis Generator develops multiple hypotheses with initial probability estimates
        print("\n💡 The Hypothesis Generator is formulating comprehensive explanations...")
        hypothesis_prompt = f"""As The Hypothesis Generator, create distinct hypotheses that could explain all evidence:

CASE SUMMARY:
I'll complete the code after line 687, which appears to be in the middle of a method called `analyze_case` within the `DesiHolmesAgency` class.

```python
CASE SUMMARY:
{self.truncate_text(case_description, 500)}

KEY OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 800)}

IDENTIFIED CONNECTIONS:
{self.truncate_text(connector_result["raw"], 800)}

SIMULATED SCENARIOS:
{self.truncate_text(simulator_result["raw"], 800)}

Generate 3-5 distinct hypotheses that explain the case. Each hypothesis should:
1. Be specific and falsifiable
2. Account for ALL evidence, not just select pieces
3. Explain any apparent inconsistencies or anomalies
4. Assign an initial probability estimate based on inherent plausibility

Format each as "Hypothesis X: [Title]" followed by explanation and probability estimate.
Make each hypothesis genuinely different, not minor variations of the same idea.
"""
        hypothesis_result = self.safe_run(self.hypothesis_generator, hypothesis_prompt, structured_extraction="hypotheses")
        self.case_data["hypotheses"]["raw"] = hypothesis_result["raw"]

        # Store structured hypotheses
        if "hypotheses" in hypothesis_result["structured"]:
            self.case_data["hypotheses"]["list"] = hypothesis_result["structured"]["hypotheses"]
a
        time.sleep(3)

        # Step 6: Evaluator assesses hypotheses with Bayesian reasoning
        print("\n⚖️ The Evaluator is assessing competing hypotheses...")
        evaluator_prompt = f"""As The Evaluator, use Bayesian reasoning to assess each hypothesis against evidence:

HYPOTHESES TO EVALUATE:
{self.truncate_text(hypothesis_result["raw"], 1500)}

KEY EVIDENCE:
{self.truncate_text(perceptor_result["raw"], 1000)}

IDENTIFIED CONNECTIONS:
{self.truncate_text(connector_result["raw"], 800)}

HISTORICAL CASE INSIGHTS:
{self.truncate_text(self.case_data["analysis"]["historical_comparisons"]["raw"], 800)}

For each hypothesis:
1. Identify evidence that CONFIRMS the hypothesis (increases probability)
2. Identify evidence that CONTRADICTS the hypothesis (decreases probability)
3. Assess the hypothesis's EXPLANATORY POWER (how many evidence points it explains)
4. Calculate REVISED PROBABILITY based on all evidence
5. Test for LOGICAL CONSISTENCY and identify any fallacies

Then provide a COMPARATIVE ANALYSIS that ranks hypotheses from most to least probable.
For the most probable hypothesis, state your confidence level and what evidence would conclusively prove or disprove it.
"""
        evaluator_result = self.safe_run(self.evaluator, evaluator_prompt)
        self.case_data["hypotheses"]["bayesian_analysis"] = evaluator_result["raw"]

        # Extract evaluation criteria if present
        criteria_pattern = r"(?:EVALUATION CRITERIA|CRITERIA)(?:\s*:)?\s*((?:.+?\n)+)"
        criteria_match = re.search(criteria_pattern, evaluator_result["raw"], re.IGNORECASE | re.DOTALL)
        if criteria_match:
            criteria_text = criteria_match.group(1)
            criteria = [c.strip() for c in re.findall(r"[•\-\*]\s*(.+)", criteria_text)]
            self.case_data["hypotheses"]["evaluation_criteria"] = criteria

        time.sleep(3)

        # Step 7: Storyteller reconstructs the most probable narrative
        print("\n📖 The Storyteller is reconstructing the most probable narrative...")
        storyteller_prompt = f"""As The Storyteller, reconstruct the most probable sequence of events:

EVALUATED HYPOTHESES:
{self.truncate_text(evaluator_result["raw"], 1500)}

KEY OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 800)}

Create a vivid chronological reconstruction that explains all evidence in these sections:
1. SETUP: Background context and initial conditions
2. CHRONOLOGY: Timeline of events in sequence
3. MOTIVATIONS: Psychological analysis of key actors
4. KEY MOMENTS: Critical decision points or actions
5. RESOLUTION: How events concluded or will conclude

Your narrative should prioritize the most probable hypothesis while acknowledging any remaining uncertainties.
Make it compelling and psychologically realistic while staying strictly grounded in evidence.
"""
        storyteller_result = self.safe_run(self.storyteller, storyteller_prompt)
        self.case_data["conclusion"]["narrative"]["raw"] = storyteller_result["raw"]

        # Extract narrative components
        for section in ["CHRONOLOGY", "MOTIVATIONS", "KEY MOMENTS"]:
            section_pattern = f"(?:{section})(?:\\s*:)?\\s*((?:.+?\\n)+)"
            section_match = re.search(section_pattern, storyteller_result["raw"], re.IGNORECASE | re.DOTALL)
            if section_match:
                section_text = section_match.group(1)
                elements = [e.strip() for e in re.findall(r"[•\-\*]\s*(.+)", section_text)]
                if section == "CHRONOLOGY":
                    self.case_data["conclusion"]["narrative"]["chronology"] = elements
                elif section == "MOTIVATIONS":
                    # Convert motivations to a dictionary of character:motivation
                    motivations = {}
                    for element in elements:
                        parts = element.split(":", 1)
                        if len(parts) == 2:
                            motivations[parts[0].strip()] = parts[1].strip()
                    self.case_data["conclusion"]["narrative"]["motivations"] = motivations
                elif section == "KEY MOMENTS":
                    self.case_data["conclusion"]["narrative"]["key_moments"] = elements

        time.sleep(3)

        # Step 8: Desi Holmes synthesizes a final solution
        print("\n🕵️ Desi Holmes is synthesizing the final solution...")
        desi_prompt = f"""As Desi Holmes, provide your comprehensive case solution:

CASE DESCRIPTION:
{self.truncate_text(case_description, 500)}

BAYESIAN ANALYSIS OF HYPOTHESES:
{self.truncate_text(evaluator_result["raw"], 1000)}

NARRATIVE RECONSTRUCTION:
{self.truncate_text(storyteller_result["raw"], 1000)}

HISTORICAL PATTERNS:
{self.truncate_text(self.case_data["analysis"]["historical_comparisons"]["raw"], 500)}

Deliver your solution in these sections:
1. CASE SUMMARY: Brief overview of the case
2. SOLUTION: Clear statement of what happened, who was responsible, and how it was accomplished
3. EVIDENCE CHAIN: Key evidence supporting your conclusion
4. ALTERNATIVE EXPLANATIONS: Other possibilities and why they're less likely
5. REMAINING UNCERTAINTIES: What we still don't know and how it could be resolved
6. RECOMMENDATIONS: What should be done next

Your solution should be elegant, accounting for all evidence while applying Occam's Razor.
Be confident when evidence warrants it, but transparent about probabilities and uncertainties.
"""
        desi_result = self.safe_run(self.desi_holmes, desi_prompt)
        self.case_data["conclusion"]["solution"] = desi_result["raw"]

        # Extract best hypothesis
        solution_pattern = r"(?:SOLUTION)(?:\s*:)?\s*((?:.+?\n)+)"
        solution_match = re.search(solution_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if solution_match:
            self.case_data["conclusion"]["best_hypothesis"] = solution_match.group(1).strip()

        # Extract remaining uncertainties
        uncertainty_pattern = r"(?:REMAINING UNCERTAINTIES|UNCERTAINTIES)(?:\s*:)?\s*((?:.+?\n)+)"
        uncertainty_match = re.search(uncertainty_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if uncertainty_match:
            uncertainty_text = uncertainty_match.group(1)
            uncertainties = [u.strip() for u in re.findall(r"[•\-\*]\s*(.+)", uncertainty_text)]
            self.case_data["conclusion"]["remaining_uncertainties"] = uncertainties

        # Extract alternatives
        alternatives_pattern = r"(?:ALTERNATIVE EXPLANATIONS|ALTERNATIVES)(?:\s*:)?\s*((?:.+?\n)+)"
        alternatives_match = re.search(alternatives_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if alternatives_match:
            alternatives_text = alternatives_match.group(1)
            alternatives = [a.strip() for a in re.findall(r"[•\-\*]\s*(.+)", alternatives_text)]
            self.case_data["conclusion"]["alternative_possibilities"] = alternatives

        # Save the case data to a file
        self.save_case_data()

        print("\n✅ Analysis complete!")

        return {
            "solution": desi_result["raw"],
            "follow_up_questions": follow_up_questions,
            "case_id": self.case_data["metadata"]["case_id"]
        }

    def save_case_data(self):
        """Save the current case data to a JSON file"""
        case_id = self.case_data["metadata"]["case_id"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"case_{case_id}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.case_data, f, indent=2)

        self.last_saved_filename = filename
        print(f"\n💾 Case data saved to {filename}")

    def incorporate_follow_up_response(self, question_number: int, response: str):
        """Incorporate a response to a follow-up question into the analysis"""
        if 0 <= question_number < len(self.follow_up_questions):
            question = self.follow_up_questions[question_number]

            # Store the response
            self.case_data["inputs"]["follow_up_responses"][question] = response

            print(f"\n📝 Response to question '{question}' recorded")

            # Update the case analysis with the new information
            # This would ideally trigger a re-analysis of the relevant components
            # For now, we'll just update the case data
        else:
            print(f"❌ Invalid question number: {question_number}")
I'll complete the code after line 687, which appears to be in the middle of a method called `analyze_case` within the `DesiHolmesAgency` class.

```python
CASE SUMMARY:
{self.truncate_text(case_description, 500)}

KEY OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 800)}

IDENTIFIED CONNECTIONS:
{self.truncate_text(connector_result["raw"], 800)}

SIMULATED SCENARIOS:
{self.truncate_text(simulator_result["raw"], 800)}

Generate 3-5 distinct hypotheses that explain the case. Each hypothesis should:
1. Be specific and falsifiable
2. Account for ALL evidence, not just select pieces
3. Explain any apparent inconsistencies or anomalies
4. Assign an initial probability estimate based on inherent plausibility

Format each as "Hypothesis X: [Title]" followed by explanation and probability estimate.
Make each hypothesis genuinely different, not minor variations of the same idea.
"""
        hypothesis_result = self.safe_run(self.hypothesis_generator, hypothesis_prompt, structured_extraction="hypotheses")
        self.case_data["hypotheses"]["raw"] = hypothesis_result["raw"]

        # Store structured hypotheses
        if "hypotheses" in hypothesis_result["structured"]:
            self.case_data["hypotheses"]["list"] = hypothesis_result["structured"]["hypotheses"]

        time.sleep(3)

        # Step 6: Evaluator assesses hypotheses with Bayesian reasoning
        print("\n⚖️ The Evaluator is assessing competing hypotheses...")
        evaluator_prompt = f"""As The Evaluator, use Bayesian reasoning to assess each hypothesis against evidence:

HYPOTHESES TO EVALUATE:
{self.truncate_text(hypothesis_result["raw"], 1500)}

KEY EVIDENCE:
{self.truncate_text(perceptor_result["raw"], 1000)}

IDENTIFIED CONNECTIONS:
{self.truncate_text(connector_result["raw"], 800)}

HISTORICAL CASE INSIGHTS:
{self.truncate_text(self.case_data["analysis"]["historical_comparisons"]["raw"], 800)}

For each hypothesis:
1. Identify evidence that CONFIRMS the hypothesis (increases probability)
2. Identify evidence that CONTRADICTS the hypothesis (decreases probability)
3. Assess the hypothesis's EXPLANATORY POWER (how many evidence points it explains)
4. Calculate REVISED PROBABILITY based on all evidence
5. Test for LOGICAL CONSISTENCY and identify any fallacies

Then provide a COMPARATIVE ANALYSIS that ranks hypotheses from most to least probable.
For the most probable hypothesis, state your confidence level and what evidence would conclusively prove or disprove it.
"""
        evaluator_result = self.safe_run(self.evaluator, evaluator_prompt)
        self.case_data["hypotheses"]["bayesian_analysis"] = evaluator_result["raw"]

        # Extract evaluation criteria if present
        criteria_pattern = r"(?:EVALUATION CRITERIA|CRITERIA)(?:\s*:)?\s*((?:.+?\n)+)"
        criteria_match = re.search(criteria_pattern, evaluator_result["raw"], re.IGNORECASE | re.DOTALL)
        if criteria_match:
            criteria_text = criteria_match.group(1)
            criteria = [c.strip() for c in re.findall(r"[•\-\*]\s*(.+)", criteria_text)]
            self.case_data["hypotheses"]["evaluation_criteria"] = criteria

        time.sleep(3)

        # Step 7: Storyteller reconstructs the most probable narrative
        print("\n📖 The Storyteller is reconstructing the most probable narrative...")
        storyteller_prompt = f"""As The Storyteller, reconstruct the most probable sequence of events:

EVALUATED HYPOTHESES:
{self.truncate_text(evaluator_result["raw"], 1500)}

KEY OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 800)}

Create a vivid chronological reconstruction that explains all evidence in these sections:
1. SETUP: Background context and initial conditions
2. CHRONOLOGY: Timeline of events in sequence
3. MOTIVATIONS: Psychological analysis of key actors
4. KEY MOMENTS: Critical decision points or actions
5. RESOLUTION: How events concluded or will conclude

Your narrative should prioritize the most probable hypothesis while acknowledging any remaining uncertainties.
Make it compelling and psychologically realistic while staying strictly grounded in evidence.
"""
        storyteller_result = self.safe_run(self.storyteller, storyteller_prompt)
        self.case_data["conclusion"]["narrative"]["raw"] = storyteller_result["raw"]

        # Extract narrative components
        for section in ["CHRONOLOGY", "MOTIVATIONS", "KEY MOMENTS"]:
            section_pattern = f"(?:{section})(?:\\s*:)?\\s*((?:.+?\\n)+)"
            section_match = re.search(section_pattern, storyteller_result["raw"], re.IGNORECASE | re.DOTALL)
            if section_match:
                section_text = section_match.group(1)
                elements = [e.strip() for e in re.findall(r"[•\-\*]\s*(.+)", section_text)]
                if section == "CHRONOLOGY":
                    self.case_data["conclusion"]["narrative"]["chronology"] = elements
                elif section == "MOTIVATIONS":
                    # Convert motivations to a dictionary of character:motivation
                    motivations = {}
                    for element in elements:
                        parts = element.split(":", 1)
                        if len(parts) == 2:
                            motivations[parts[0].strip()] = parts[1].strip()
                    self.case_data["conclusion"]["narrative"]["motivations"] = motivations
                elif section == "KEY MOMENTS":
                    self.case_data["conclusion"]["narrative"]["key_moments"] = elements

        time.sleep(3)

        # Step 8: Desi Holmes synthesizes a final solution
        print("\n🕵️ Desi Holmes is synthesizing the final solution...")
        desi_prompt = f"""As Desi Holmes, provide your comprehensive case solution:

CASE DESCRIPTION:
{self.truncate_text(case_description, 500)}

BAYESIAN ANALYSIS OF HYPOTHESES:
{self.truncate_text(evaluator_result["raw"], 1000)}

NARRATIVE RECONSTRUCTION:
{self.truncate_text(storyteller_result["raw"], 1000)}

HISTORICAL PATTERNS:
{self.truncate_text(self.case_data["analysis"]["historical_comparisons"]["raw"], 500)}

Deliver your solution in these sections:
1. CASE SUMMARY: Brief overview of the case
2. SOLUTION: Clear statement of what happened, who was responsible, and how it was accomplished
3. EVIDENCE CHAIN: Key evidence supporting your conclusion
4. ALTERNATIVE EXPLANATIONS: Other possibilities and why they're less likely
5. REMAINING UNCERTAINTIES: What we still don't know and how it could be resolved
6. RECOMMENDATIONS: What should be done next

Your solution should be elegant, accounting for all evidence while applying Occam's Razor.
Be confident when evidence warrants it, but transparent about probabilities and uncertainties.
"""
        desi_result = self.safe_run(self.desi_holmes, desi_prompt)
        self.case_data["conclusion"]["solution"] = desi_result["raw"]

        # Extract best hypothesis
        solution_pattern = r"(?:SOLUTION)(?:\s*:)?\s*((?:.+?\n)+)"
        solution_match = re.search(solution_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if solution_match:
            self.case_data["conclusion"]["best_hypothesis"] = solution_match.group(1).strip()

        # Extract remaining uncertainties
        uncertainty_pattern = r"(?:REMAINING UNCERTAINTIES|UNCERTAINTIES)(?:\s*:)?\s*((?:.+?\n)+)"
        uncertainty_match = re.search(uncertainty_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if uncertainty_match:
            uncertainty_text = uncertainty_match.group(1)
            uncertainties = [u.strip() for u in re.findall(r"[•\-\*]\s*(.+)", uncertainty_text)]
            self.case_data["conclusion"]["remaining_uncertainties"] = uncertainties

        # Extract alternatives
        alternatives_pattern = r"(?:ALTERNATIVE EXPLANATIONS|ALTERNATIVES)(?:\s*:)?\s*((?:.+?\n)+)"
        alternatives_match = re.search(alternatives_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if alternatives_match:
            alternatives_text = alternatives_match.group(1)
            alternatives = [a.strip() for a in re.findall(r"[•\-\*]\s*(.+)", alternatives_text)]
            self.case_data["conclusion"]["alternative_possibilities"] = alternatives

        # Save the case data to a file
        self.save_case_data()

        print("\n✅ Analysis complete!")

        return {
            "solution": desi_result["raw"],
            "follow_up_questions": follow_up_questions,
            "case_id": self.case_data["metadata"]["case_id"]
        }

    def save_case_data(self):
        """Save the current case data to a JSON file"""
        case_id = self.case_data["metadata"]["case_id"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"case_{case_id}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.case_data, f, indent=2)

        self.last_saved_filename = filename
        print(f"\n💾 Case data saved to {filename}")

    def incorporate_follow_up_response(self, question_number: int, response: str):
        """Incorporate a response to a follow-up question into the analysis"""
        if 0 <= question_number < len(self.follow_up_questions):
            question = self.follow_up_questions[question_number]

            # Store the response
            self.case_data["inputs"]["follow_up_responses"][question] = response

            print(f"\n📝 Response to question '{question}' recorded")

            # Update the case analysis with the new information
            # This would ideally trigger a re-analysis of the relevant components
            # For now, we'll just update the case data
        else:
            print(f"❌ Invalid question number: {question_number}")
