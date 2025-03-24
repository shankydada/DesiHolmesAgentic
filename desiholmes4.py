from phi.assistant import Assistant as Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
from typing import Dict, List, Tuple
import json
import time
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
        self.case_patterns["murder"] = [
            "Disguised cause of death",
            "Misdirection of time of death",
            "Staged crime scenes",
            "Alibi deception",
            "Multiple perpetrators coordination"
        ]
        self.case_patterns["missing person"] = [
            "Voluntary disappearance",
            "Identity change patterns",
            "Ransom or extortion situations",
            "Staged abductions",
            "Criminal network involvement"
        ]
        self.case_patterns["corporate espionage"] = [
            "Insider threat patterns",
            "Digital infiltration techniques",
            "Social engineering methods",
            "Physical security breaches",
            "Long-term intelligence gathering"
        ]
        self.case_patterns["forgery"] = [
            "Authentication bypass techniques",
            "Materials substitution",
            "Historical inconsistencies",
            "Technical imperfections",
            "Distribution patterns"
        ]

    def categorize_case(self, case_description: str) -> List[Tuple[str, float]]:
        """Return probable case categories with confidence scores"""
        categories = []
        for category in self.case_categories:
            if category.lower() in case_description.lower():
                confidence = 0.7 + (random.random() * 0.3)  # Simulated confidence
                categories.append((category, confidence))

        if not categories:
            categories = [
                ("theft", 0.4 + (random.random() * 0.3)),
                ("murder", 0.3 + (random.random() * 0.3))
            ]

        return sorted(categories, key=lambda x: x[1], reverse=True)

    def get_similar_cases(self, case_description: str, top_n: int = 5) -> List[Dict]:
        """Return top N similar historical cases based on pattern matching"""
        categories = self.categorize_case(case_description)

        similar_cases = []
        for category, confidence in categories:
            for i in range(min(3, top_n)):
                similar_cases.append({
                    "id": f"CASE-{random.randint(10000, 99999)}",
                    "title": f"The {random.choice(['Mysterious', 'Peculiar', 'Strange', 'Baffling'])} {category.title()} of {random.choice(['London', 'Paris', 'New York', 'Tokyo'])}",
                    "year": random.randint(1900, 2023),
                    "similarity": confidence * (0.9 - (i * 0.1)),
                    "key_pattern": random.choice(self.case_patterns.get(category, ["Unknown pattern"])),
                    "solution_type": random.choice(["Deception", "Technical trick", "Insider betrayal", "Opportunity exploitation", "Psychological manipulation"])
                })

        return sorted(similar_cases, key=lambda x: x["similarity"], reverse=True)[:top_n]

    def get_pattern_insights(self, case_categories: List[Tuple[str, float]], case_elements: List[str]) -> List[str]:
        """Generate pattern insights based on case categories and elements"""
        insights = []

        for category, _ in case_categories:
            patterns = self.case_patterns.get(category, [])
            for pattern in patterns:
                for element in case_elements:
                    if any(keyword in element.lower() for keyword in pattern.lower().split()):
                        insights.append(f"Pattern match: '{pattern}' is common in {category} cases and matches your element '{element}'")

        insights.append(f"This case fits primarily in the {case_categories[0][0]} category with {case_categories[0][1]:.1%} confidence")

        if len(case_categories) > 1:
            insights.append(f"Cross-category analysis: This case shows patterns from both {case_categories[0][0]} and {case_categories[1][0]} categories, which suggests potential complexity")

        return insights

class DesiHolmesAgency:
    def __init__(self):
        self.case_database = CaseDatabase()

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

        self.case_data = {
            "metadata": {
                "case_id": self._generate_case_id(),
                "timestamp": datetime.datetime.now().isoformat(),
                "categories": []
            },
            "inputs": {
                "description": "",
                "follow_up_responses": {}
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

            if not structured_data:
                bullet_points = re.findall(r"[•\-\*]\s*(.+)", text)
                structured_data["observations"] = [bp.strip() for bp in bullet_points]

        elif extraction_type == "hypotheses":
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
            text = str(text)

        if len(text) <= max_length:
            return text
        return text[:max_length] + "... [truncated for length]"

    def format_response(self, response):
        """Format the response for better readability and consistency."""
        if not isinstance(response, str):
            response = str(response)

        response = response.strip()
        if not re.search(r"Hypothesis\s+\d+", response, re.IGNORECASE):
            response = re.sub(r'^\d+\.\s', '• ', response, flags=re.MULTILINE)

        response = response.replace('\n\n\n', '\n\n')

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
                if attempt > 0:
                    time.sleep(delay)

                truncated_prompt = prompt if attempt == 0 else self.truncate_text(prompt, 3000 - (attempt * 500))
                result = agent.run(truncated_prompt)

                formatted_result = self.format_response(result)

                structured_data = {}
                if structured_extraction:
                    structured_data = self.extract_structured_data(formatted_result, structured_extraction)

                return {
                    "raw": formatted_result,
                    "structured": structured_data
                }

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
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

        questions = clarifier_result["structured"].get("questions", [])

        if not questions:
            question_pattern = r"(\d+\.\s*[^.?!]+\?)"
            questions = re.findall(question_pattern, clarifier_result["raw"])
            questions = [q.strip() for q in questions]

        if not questions:
            questions = [
                "Can you provide more details about the timeline of events?",
                "Are there any security cameras or other surveillance devices that might have recorded relevant information?",
                "Were there any witnesses who might have observed something unusual?",
                "Has the victim or subject had any recent conflicts or unusual behavior?",
                "Are there any unusual financial transactions or communications that might be relevant?"
            ]

        self.follow_up_questions = questions
        self.case_data["analysis"]["information_gaps"]["critical_questions"] = questions
        self.case_data["analysis"]["information_gaps"]["raw"] = clarifier_result["raw"]

        return questions

    def analyze_historical_cases(self, case_description: str):
        """Compare current case with historical cases to derive insights"""
        print("\n📚 Comparing with historical cases...")

        categories = self.case_database.categorize_case(case_description)
        self.case_data["metadata"]["categories"] = categories

        similar_cases = self.case_database.get_similar_cases(case_description)
        self.case_data["analysis"]["historical_comparisons"]["similar_cases"] = similar_cases

        case_elements = re.findall(r'([^.!?]+[.!?])', case_description)

        pattern_insights = self.case_database.get_pattern_insights(categories, case_elements)

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

        key_diff_pattern = r"(?:KEY DIFFERENCES|NOTABLE DIFFERENCES)(?:\s*:)?\s*((?:.+?\n)+)"
        diff_match = re.search(key_diff_pattern, historical_result["raw"], re.IGNORECASE | re.DOTALL)
        if diff_match:
            diff_text = diff_match.group(1)
            key_differences = [d.strip() for d in re.findall(r"[•\-\*]\s*(.+)", diff_text)]
            self.case_data["analysis"]["historical_comparisons"]["key_differences"] = key_differences

    def analyze_patterns(self, case_description: str):
        """Identify patterns in the case data"""
        print("\n🔄 Identifying patterns in the evidence...")

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

        self.case_data = {
            "metadata": {
                "case_id": self._generate_case_id(),
                "timestamp": datetime.datetime.now().isoformat(),
                "categories": []
            },
            "inputs": {
                "description": "",
                "follow_up_responses": {}
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

        self.follow_up_questions = []
        self.case_data["inputs"]["description"] = case_description

        follow_up_questions = self.process_follow_up_questions(case_description)
        print("\n❓ Follow-up questions to consider:")
        for i, question in enumerate(follow_up_questions, 1):
            print(f"   {i}. {question}")

        self.analyze_historical_cases(case_description)

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

        for key in ["people", "places", "objects", "timeline"]:
            if key in perceptor_result["structured"]:
                self.case_data["analysis"]["perceptions"][key] = perceptor_result["structured"][key]
        if "observations" in perceptor_result["structured"]:
            self.case_data["analysis"]["perceptions"]["observations"] = perceptor_result["structured"]["observations"]

        time.sleep(3)

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

        for connection_type in ["STRONG CONNECTIONS", "POSSIBLE CONNECTIONS", "ANOMALOUS CONNECTIONS"]:
            pattern = f"(?:{connection_type})(?:\\s*:)?\\s*((?:.+?\\n)+)"
            match = re.search(pattern, connector_result["raw"], re.IGNORECASE | re.DOTALL)
            if match:
                connection_text = match.group(1)
                connections = [c.strip() for c in re.findall(r"[•\-\*]\s*(.+)", connection_text)]
                key = connection_type.lower().replace(" ", "_")
                self.case_data["analysis"]["connections"][key] = connections

        time.sleep(3)

        self.analyze_patterns(case_description)
        time.sleep(3)

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
4. Assess the physical and temporal feasibility of the scenario

Present 2-3 distinct scenarios with different explanations. For each, assign a probability estimate based on how well it fits all evidence.
"""
        simulator_result = self.safe_run(self.simulator, simulator_prompt)
        self.case_data["analysis"]["simulations"]["raw"] = simulator_result["raw"]

        scenario_pattern = r"(?:SCENARIO|Scenario)\s*(\d+)\s*(?:\(.*?(\d+)%.*?\))?(?::|-)?\s*([^\n]+(?:\n(?!\s*SCENARIO|\s*Scenario)[^\n]+)*)"
        scenarios = re.findall(scenario_pattern, simulator_result["raw"], re.IGNORECASE | re.DOTALL)

        for number, probability, content in scenarios:
            scenario = {
                "number": int(number),
                "title": content.split('\n')[0].strip(),
                "description": content.strip(),
                "probability": int(probability) if probability else 0
            }
            self.case_data["analysis"]["simulations"]["scenarios"].append(scenario)
            if probability:
                self.case_data["analysis"]["simulations"]["probability_estimates"][f"scenario_{number}"] = int(probability)

        time.sleep(3)

        print("\n🧩 The Hypothesis Generator is formulating possible solutions...")
        hypothesis_prompt = f"""As The Hypothesis Generator, create distinct hypotheses to explain this case:

CASE SUMMARY:
{self.truncate_text(case_description, 500)}

KEY OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 800)}

KEY CONNECTIONS:
{self.truncate_text(connector_result["raw"], 800)}

SIMULATED SCENARIOS:
{self.truncate_text(simulator_result["raw"], 800)}

Generate 3-5 competing hypotheses that could explain the case. Each hypothesis should:
1. Have a clear, concise title
2. Propose a specific explanation for what happened
3. Account for all critical evidence
4. Assign an initial probability based on plausibility (e.g., "Initial probability: 40%")

Make each hypothesis distinct and falsifiable. Include at least one unconventional hypothesis.

Format as:
HYPOTHESIS 1: [Title]
[Detailed explanation]
Initial probability: [X]%

HYPOTHESIS 2: [Title]
...etc.
"""
        hypothesis_result = self.safe_run(self.hypothesis_generator, hypothesis_prompt, structured_extraction="hypotheses")
        self.case_data["hypotheses"]["raw"] = hypothesis_result["raw"]

        if "hypotheses" in hypothesis_result["structured"]:
            for hypothesis in hypothesis_result["structured"]["hypotheses"]:
                probability_match = re.search(r"initial probability:?\s*(\d+)%", hypothesis["description"], re.IGNORECASE)
                probability = int(probability_match.group(1)) if probability_match else 0

                hypothesis_data = {
                    "number": hypothesis["number"],
                    "title": hypothesis["title"],
                    "description": hypothesis["description"],
                    "probability": probability
                }
                self.case_data["hypotheses"]["list"].append(hypothesis_data)

        time.sleep(3)

        print("\n⚖️ The Evaluator is assessing the hypotheses...")
        evaluator_prompt = f"""As The Evaluator, assess each hypothesis against the evidence:

HYPOTHESES:
{self.truncate_text(hypothesis_result["raw"], 1500)}

KEY EVIDENCE:
{self.truncate_text(perceptor_result["raw"], 800)}

CONNECTIONS AND PATTERNS:
{self.truncate_text(connector_result["raw"], 600)}

For each hypothesis, provide:
1. CONFIRMING EVIDENCE: Evidence that supports this hypothesis
2. CONTRADICTORY EVIDENCE: Evidence that conflicts with this hypothesis
3. EXPLANATORY POWER: How well it explains key elements (rate 1-10)
4. PARSIMONY ASSESSMENT: How well it satisfies Occam's Razor (rate 1-10)
5. FINAL PROBABILITY: Updated probability percentage based on all factors

Then provide an OVERALL ASSESSMENT indicating which hypothesis best explains all evidence and why.
"""
        evaluator_result = self.safe_run(self.evaluator, evaluator_prompt)
        self.case_data["hypotheses"]["evaluation"] = evaluator_result["raw"]

        # Extract Bayesian analysis if possible
        # Extract Bayesian analysis if possible
        hypothesis_evaluations = {}
        for hypothesis in self.case_data["hypotheses"]["list"]:
            number = hypothesis["number"]
            eval_pattern = r"(?:HYPOTHESIS|Hypothesis)\s*" + str(number) + r".*?(?:FINAL PROBABILITY|Final Probability|final probability):?\s*(\d+)%"

            prob_match = re.search(eval_pattern, evaluator_result["raw"], re.IGNORECASE | re.DOTALL)
            if prob_match:
                hypothesis_evaluations[f"hypothesis_{number}"] = int(prob_match.group(1))

        self.case_data["hypotheses"]["bayesian_analysis"] = hypothesis_evaluations
        time.sleep(3)

        print("\n📝 The Storyteller is constructing the narrative...")
        storyteller_prompt = f"""As The Storyteller, reconstruct the most likely sequence of events:

CASE SUMMARY:
{self.truncate_text(case_description, 500)}

EVALUATED HYPOTHESES:
{self.truncate_text(evaluator_result["raw"], 1000)}

TIMELINE AND OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 800)}

Create a compelling, chronological narrative that explains:
1. CHRONOLOGY: The precise sequence of events from beginning to end
2. MOTIVATIONS: The psychological drivers behind key actions
3. KEY MOMENTS: The critical decision points or actions that led to the outcome
4. RESOLUTION: How all elements of the case are resolved by this explanation

Your narrative should be detailed yet concise, accounting for all evidence while maintaining logical and psychological consistency.
"""
        storyteller_result = self.safe_run(self.storyteller, storyteller_prompt)
        self.case_data["conclusion"]["narrative"]["raw"] = storyteller_result["raw"]

        # Extract chronology if possible
        chronology_pattern = r"(?:CHRONOLOGY|Timeline)(?:\s*:)?\s*((?:.+?\n)+)"
        chronology_match = re.search(chronology_pattern, storyteller_result["raw"], re.IGNORECASE | re.DOTALL)
        if chronology_match:
            chronology_text = chronology_match.group(1)
            chronology = [c.strip() for c in re.findall(r"[•\-\*]\s*(.+)", chronology_text)]
            self.case_data["conclusion"]["narrative"]["chronology"] = chronology
        time.sleep(3)

        print("\n🕵️ Desi Holmes is delivering the solution...")
        desi_prompt = f"""As Desi Holmes, master detective, synthesize all analysis into a comprehensive solution:

CASE DESCRIPTION:
{self.truncate_text(case_description, 800)}

KEY OBSERVATIONS:
{self.truncate_text(perceptor_result["raw"], 700)}

EVALUATED HYPOTHESES:
{self.truncate_text(evaluator_result["raw"], 800)}

NARRATIVE RECONSTRUCTION:
{self.truncate_text(storyteller_result["raw"], 800)}

HISTORICAL CASE COMPARISONS:
{self.truncate_text(self.case_data["analysis"]["historical_comparisons"]["raw"], 500)}

Provide your definitive solution in these sections:
1. THE SOLUTION: A concise summary of what actually happened
2. KEY EVIDENCE: The most important evidence supporting your conclusion
3. REASONING CHAIN: The logical steps that led to your solution
4. CONFIDENCE ASSESSMENT: How certain you are of this solution (with percentage)
5. REMAINING UNCERTAINTIES: Any elements that cannot be fully explained
6. FOLLOW-UP RECOMMENDATIONS: Specific actions to verify your conclusion

Your solution should be elegant, accounting for all evidence while satisfying Occam's Razor.
"""
        desi_result = self.safe_run(self.desi_holmes, desi_prompt)

        # Extract the solution section
        solution_pattern = r"(?:THE SOLUTION|SOLUTION)(?:\s*:)?\s*((?:.+?\n)+)"
        solution_match = re.search(solution_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if solution_match:
            solution_text = solution_match.group(1).strip()
            self.case_data["conclusion"]["solution"] = solution_text
        else:
            self.case_data["conclusion"]["solution"] = "Solution could not be clearly extracted from the analysis."

        # Extract confidence level
        confidence_pattern = r"(?:CONFIDENCE ASSESSMENT|CONFIDENCE)(?:\s*:)?\s*.*?(\d+)%"
        confidence_match = re.search(confidence_pattern, desi_result["raw"], re.IGNORECASE)
        if confidence_match:
            self.case_data["conclusion"]["confidence_level"] = int(confidence_match.group(1))

        # Extract remaining uncertainties
        uncertainties_pattern = r"(?:REMAINING UNCERTAINTIES|UNCERTAINTIES)(?:\s*:)?\s*((?:.+?\n)+)"
        uncertainties_match = re.search(uncertainties_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if uncertainties_match:
            uncertainties_text = uncertainties_match.group(1)
            uncertainties = [u.strip() for u in re.findall(r"[•\-\*]\s*(.+)", uncertainties_text)]
            self.case_data["conclusion"]["remaining_uncertainties"] = uncertainties

        print("\n✅ Case analysis complete!")

        # Save case to file
        self.save_case_to_file()

        # Format the final solution for return
        final_solution = self.format_final_solution(desi_result["raw"])

        return {
            "solution": final_solution,
            "follow_up_questions": self.follow_up_questions,
            "case_id": self.case_data["metadata"]["case_id"]
        }

    def format_final_solution(self, raw_solution):
        """Format the final solution for better presentation"""
        sections = {}
        section_patterns = {
            "solution": r"(?:THE SOLUTION|SOLUTION)(?:\s*:)?\s*((?:.+?\n)+)",
            "evidence": r"(?:KEY EVIDENCE|EVIDENCE)(?:\s*:)?\s*((?:.+?\n)+)",
            "reasoning": r"(?:REASONING CHAIN|REASONING)(?:\s*:)?\s*((?:.+?\n)+)",
            "confidence": r"(?:CONFIDENCE ASSESSMENT|CONFIDENCE)(?:\s*:)?\s*((?:.+?\n)+)",
            "uncertainties": r"(?:REMAINING UNCERTAINTIES|UNCERTAINTIES)(?:\s*:)?\s*((?:.+?\n)+)",
            "recommendations": r"(?:FOLLOW-UP RECOMMENDATIONS|RECOMMENDATIONS)(?:\s*:)?\s*((?:.+?\n)+)"
        }

        for key, pattern in section_patterns.items():
            match = re.search(pattern, raw_solution, re.IGNORECASE | re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()

        # Format the solution with Markdown
        formatted_solution = "# Case Solution\n\n"

        if "solution" in sections:
            formatted_solution += f"## The Solution\n{sections['solution']}\n\n"

        if "evidence" in sections:
            formatted_solution += f"## Key Evidence\n{sections['evidence']}\n\n"

        if "reasoning" in sections:
            formatted_solution += f"## Reasoning Chain\n{sections['reasoning']}\n\n"

        if "confidence" in sections:
            formatted_solution += f"## Confidence Assessment\n{sections['confidence']}\n\n"

        if "uncertainties" in sections:
            formatted_solution += f"## Remaining Uncertainties\n{sections['uncertainties']}\n\n"

        if "recommendations" in sections:
            formatted_solution += f"## Follow-up Recommendations\n{sections['recommendations']}\n\n"

        if len(sections) == 0:
            return raw_solution

        return formatted_solution

    def save_case_to_file(self):
        """Save the case data to a JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        case_id = self.case_data["metadata"]["case_id"]
        filename = f"case_{case_id}_{timestamp}.json"

        os.makedirs("cases", exist_ok=True)
        filepath = os.path.join("cases", filename)

        with open(filepath, 'w') as f:
            json.dump(self.case_data, f, indent=2)

        self.last_saved_filename = filepath
        print(f"\nCase data saved to {filepath}")

        return filepath

    def add_follow_up_information(self, case_id: str, question_index: int, answer: str) -> Dict:
        """Add follow-up information and update the analysis"""
        if case_id != self.case_data["metadata"]["case_id"]:
            return {"error": "Case ID does not match current case"}

        if question_index < 0 or question_index >= len(self.follow_up_questions):
            return {"error": "Invalid question index"}

        question = self.follow_up_questions[question_index]
        self.case_data["inputs"]["follow_up_responses"][question] = answer

        print(f"\n➕ Adding follow-up information for question: {question}")

        original_case = self.case_data["inputs"]["description"]
        new_info = f"\nADDITIONAL INFORMATION:\nQ: {question}\nA: {answer}"

        desi_prompt = f"""As Desi Holmes, master detective, update your analysis with this new information:

ORIGINAL CASE:
{self.truncate_text(original_case, 800)}

PREVIOUS SOLUTION:
{self.truncate_text(self.case_data["conclusion"]["solution"], 500)}

NEW INFORMATION:
{new_info}

How does this new information affect your solution? Provide:
1. UPDATED SOLUTION: Your revised solution incorporating the new information
2. IMPACT ASSESSMENT: How this information changes the case (confirms, contradicts, or clarifies)
3. CONFIDENCE CHANGE: Whether your confidence level increased or decreased

#Only make changes if the new information meaningfully impacts your analysis.
"""       
        desi_result = self.safe_run(self.desi_holmes, desi_prompt)

        updated_solution_pattern = r"(?:UPDATED SOLUTION|SOLUTION)(?:\s*:)?\s*((?:.+?\n)+)"
        updated_match = re.search(updated_solution_pattern, desi_result["raw"], re.IGNORECASE | re.DOTALL)
        if updated_match:
            updated_solution = updated_match.group(1).strip()
            self.case_data["conclusion"]["solution"] = updated_solution

        confidence_pattern = r"(?:CONFIDENCE CHANGE|CONFIDENCE)(?:\s*:)?\s*.*?(\d+)%"
        confidence_match = re.search(confidence_pattern, desi_result["raw"], re.IGNORECASE)
        if confidence_match:
            self.case_data["conclusion"]["confidence_level"] = int(confidence_match.group(1))

        self.save_case_to_file()

        return {
            "updated_solution": self.format_final_solution(desi_result["raw"]),
            "case_id": case_id
        }

# Main execution block
if __name__ == "__main__":
    # Create an instance of the DesiHolmesAgency
    agency = DesiHolmesAgency()

    # Example case description
    case_description = """
    A wealthy businessman was found dead in his study, locked from the inside. 
    The police suspect foul play, but the only evidence is a broken window and a missing safe.
    """

    # Analyze the case
    result = agency.analyze_case(case_description)

    # Print the output
    print(json.dumps(result, indent=2))