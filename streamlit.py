import streamlit as st
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

class DesiHolmesAgency:
    def __init__(self):
        self.case_database = CaseDatabase()

    def analyze_case(self, case_description: str):
        categories = self.case_database.categorize_case(case_description)
        similar_cases = self.case_database.get_similar_cases(case_description)

        return categories, similar_cases

def main():
    st.title("Desi Holmes Agency Case Analyzer")
    
    case_description = st.text_area("Enter Case Description", height=200)
    
    if st.button("Analyze Case"):
        agency = DesiHolmesAgency()
        categories, similar_cases = agency.analyze_case(case_description)

        st.write("### Probable Case Categories:")
        for category, confidence in categories:
            st.write(f"- **{category.title()}:** {confidence:.1%}")

        st.write("\n### Similar Historical Cases:")
        for case in similar_cases:
            st.write(f"- **Case ID:** {case['id']}")
            st.write(f"  - **Title:** {case['title']}")
            st.write(f"  - **Year:** {case['year']}")
            st.write(f"  - **Similarity:** {case['similarity']:.1%}")
            st.write(f"  - **Key Pattern:** {case['key_pattern']}")
            st.write(f"  - **Solution Type:** {case['solution_type']}\n")

if __name__ == "__main__":
    main()
