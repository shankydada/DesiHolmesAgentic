from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional
import json
import time
import textwrap
import re

load_dotenv()

# Base model configuration for all agents
MODEL_ID = "llama3-70b-8192"  # Correct format for Groq models

class DesiHolmesAgency:
    def __init__(self):
        # Initialize all specialized agents
        self.perceptor = Agent(
            model=Groq(id=MODEL_ID),
            description="The Perceptor: Uses generative AI to analyze case descriptions and evidence, creating a comprehensive perception of the scene and facts."
        )
        
        self.connector = Agent(
            model=Groq(id=MODEL_ID),
            description="The Connector: Identifies data relationships and patterns in observations that humans might miss, using AI pattern recognition."
        )
        
        self.simulator = Agent(
            model=Groq(id=MODEL_ID),
            description="The Simulator: Uses AI simulation techniques to model different scenarios and investigate outcomes."
        )
        
        self.hypothesis_generator = Agent(
            model=Groq(id=MODEL_ID),
            description="The Hypothesis Generator: Creates multiple plausible explanations based on all available information, considering even non-obvious scenarios."
        )
        
        self.evaluator = Agent(
            model=Groq(id=MODEL_ID),
            description="The Evaluator: Uses AI reasoning to assess hypotheses against evidence and probability theory."
        )
        
        self.knowledge_base = Agent(
            model=Groq(id=MODEL_ID),
            description="The Knowledge Base: Accesses vast amounts of information to contextualize the case within historical, cultural and scientific precedents."
        )
        
        self.storyteller = Agent(
            model=Groq(id=MODEL_ID),
            description="The Storyteller: Generates a detailed narrative reconstruction of events based on the best hypothesis."
        )
        
        self.desi_holmes = Agent(
            model=Groq(id=MODEL_ID),
            description="Desi Holmes: The generative AI detective who synthesizes all information to deliver the final solution with creative flair."
        )
        
        # Storage for case information
        self.case_data = {
            "description": "",
            "perceptions": "",
            "connections": "",
            "simulations": "",
            "hypotheses": [],
            "best_hypothesis": "",
            "knowledge_context": "",
            "narrative": "",
            "solution": ""
        }
    
    def truncate_text(self, text, max_length=3000):
        """Truncate text to a maximum length while preserving meaning."""
        if not isinstance(text, str):
            text = str(text)  # Convert any non-string objects to strings
            
        if len(text) <= max_length:
            return text
        return text[:max_length] + "... [truncated for length]"
    
    def format_response(self, response):
        """Format the response for better readability."""
        if not isinstance(response, str):
            response = str(response)
            
        # Clean up the response
        response = response.strip()
        
        # Convert numbered lists to bullet points for consistency
        response = re.sub(r'^\d+\.\s', '• ', response, flags=re.MULTILINE)
        
        # Add proper spacing
        response = response.replace('\n\n', '\n')
        
        return response
    
    def safe_run(self, agent, prompt, max_retries=3, delay=5):
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
                return self.format_response(result)
            
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    print("Maximum retry attempts reached. Using fallback response.")
                    return f"Analysis could not be completed due to technical limitations. Error: {str(e)}"
    
    def analyze_case(self, case_description: str) -> Dict:
        """Process a case through the entire AI detective agency workflow."""
        print("🔍 Case received. Beginning AI analysis...")
        
        # Store case description
        self.case_data["description"] = case_description
        
        # Step 1: Perceptor analyzes the case and evidence
        print("\n🧠 The Perceptor is scanning the evidence...")
        perceptor_prompt = f"As an AI detective with enhanced perception capabilities, analyze this case description. Generate a detailed perception of the scene and facts, listing key observations as numbered points:\n\n{case_description}\n\nBe thorough and identify details a human might miss."
        self.case_data["perceptions"] = self.safe_run(self.perceptor, perceptor_prompt)
        # Add delay between API calls
        time.sleep(3)
        
        # Step 2: Connector identifies patterns and relationships
        print("\n🔎 The Connector is identifying patterns and relationships...")
        connector_prompt = f"Identify complex patterns, data relationships, and statistical anomalies in these observations as numbered points:\n\n{self.truncate_text(self.case_data['perceptions'])}\n\nUse your pattern recognition capabilities to find connections a human detective might overlook."
        self.case_data["connections"] = self.safe_run(self.connector, connector_prompt)
        time.sleep(3)
        
        # Step 3: Simulator runs different scenarios
        print("\n📊 The Simulator is modeling possible scenarios...")
        simulator_prompt = f"As an AI capable of simulating events, generate different scenarios for this case and evaluate their probability:\n\nCase: {self.truncate_text(case_description, 500)}\n\nPerceptions: {self.truncate_text(self.case_data['perceptions'], 600)}\n\nConnections: {self.truncate_text(self.case_data['connections'], 500)}\n\nSimulate at least 3 different scenarios with probability estimates."
        self.case_data["simulations"] = self.safe_run(self.simulator, simulator_prompt)
        time.sleep(3)
        
        # Step 4: Hypothesis Generator develops multiple hypotheses
        print("\n💡 The Hypothesis Generator is formulating potential explanations...")
        hypothesis_prompt = f"Generate 3 distinct hypotheses to explain the case, labeled as Hypothesis 1, Hypothesis 2, and Hypothesis 3:\n\nCase: {self.truncate_text(case_description, 500)}\n\nKey Perceptions: {self.truncate_text(self.case_data['perceptions'], 500)}\n\nConnections: {self.truncate_text(self.case_data['connections'], 300)}\n\nSimulations: {self.truncate_text(self.case_data['simulations'], 300)}\n\nFor each hypothesis, explain the reasoning and consider even non-obvious possibilities."
        hypotheses_raw = self.safe_run(self.hypothesis_generator, hypothesis_prompt)
        self.case_data["hypotheses"] = hypotheses_raw
        time.sleep(3)
        
        # Step 5: Evaluator assesses hypotheses to find the best one
        print("\n⚖️ The Evaluator is assessing hypotheses...")
        evaluator_prompt = f"Evaluate these hypotheses using Bayesian reasoning and determine which is most probable. Start with 'The most likely hypothesis is...' and then explain why:\n\n{self.truncate_text(hypotheses_raw, 1500)}\n\nExplain your evaluation criteria and why your chosen hypothesis best fits the evidence with statistical reasoning."
        self.case_data["best_hypothesis"] = self.safe_run(self.evaluator, evaluator_prompt)
        time.sleep(3)
        
        # Step 6: Knowledge Base provides context
        print("\n📚 The Knowledge Base is providing relevant context...")
        knowledge_prompt = f"Draw on your vast knowledge repository to provide context for this case in numbered points:\n\nCase Summary: {self.truncate_text(case_description, 500)}\n\nBest Hypothesis: {self.truncate_text(self.case_data['best_hypothesis'], 500)}\n\nIdentify relevant historical, scientific, cultural, or psychological patterns that help explain this case."
        self.case_data["knowledge_context"] = self.safe_run(self.knowledge_base, knowledge_prompt)
        time.sleep(3)
        
        # Step 7: Storyteller reconstructs the events
        print("\n🎬 The Storyteller is reconstructing the narrative...")
        storyteller_prompt = f"Create a detailed narrative reconstruction of the events based on the best hypothesis. Present this as a chronological story:\n\nCase: {self.truncate_text(case_description, 500)}\n\nBest Hypothesis: {self.truncate_text(self.case_data['best_hypothesis'], 800)}\n\nProvide a vivid, creative narrative of what happened, dividing it into clear phases with sensory details."
        self.case_data["narrative"] = self.safe_run(self.storyteller, storyteller_prompt)
        time.sleep(3)
        
        # Step 8: Desi Holmes delivers the final solution
        print("\n🎩 Desi Holmes is presenting the solution...")
        desi_holmes_prompt = (
            f"As Desi Holmes, the generative AI detective, deliver your final solution to this case with creative flair. Structure your response with these sections:\n\n"
            f"1. Opening statement that captures the essence of the case\n"
            f"2. Key perceptions that led to your solution\n"
            f"3. Your chain of AI-enhanced reasoning\n"
            f"4. The solution with probability confidence\n"
            f"5. Final statement on what humans might have missed\n\n"
            f"Case: {self.truncate_text(case_description, 400)}\n\n"
            f"Key Perceptions: {self.truncate_text(self.case_data['perceptions'], 400)}\n\n"
            f"Best Hypothesis: {self.truncate_text(self.case_data['best_hypothesis'], 400)}\n\n"
            f"Narrative: {self.truncate_text(self.case_data['narrative'], 400)}\n\n"
            f"Explain your solution with Desi Holmes' signature blend of logic and imagination."
        )
        self.case_data["solution"] = self.safe_run(self.desi_holmes, desi_holmes_prompt)
        
        print("\n✅ Analysis complete! Desi Holmes has solved the case.")
        return self.case_data
    
    def get_full_report(self) -> str:
        """Generate a comprehensive report of the entire case analysis."""
        report = f"""
# DESI HOLMES AI DETECTIVE AGENCY: CASE REPORT

## THE CASE
{self.case_data['description']}

## AI PERCEPTIONS
{self.case_data['perceptions']}

## PATTERN CONNECTIONS
{self.case_data['connections']}

## SCENARIO SIMULATIONS
{self.case_data['simulations']}

## HYPOTHESES CONSIDERED
{self.case_data['hypotheses']}

## MOST PROBABLE HYPOTHESIS
{self.case_data['best_hypothesis']}

## KNOWLEDGE CONTEXT
{self.case_data['knowledge_context']}

## NARRATIVE RECONSTRUCTION
{self.case_data['narrative']}

## DESI HOLMES' SOLUTION
{self.case_data['solution']}
"""
        return report
    
    def save_report(self, filename: str = "desi_holmes_case_report.md"):
        """Save the case report to a markdown file."""
        with open(filename, 'w') as f:
            f.write(self.get_full_report())
        print(f"Report saved to {filename}")
    
    def print_summary(self):
        """Print a concise summary of the case analysis to the console."""
        print("\n" + "="*80)
        print(" "*30 + "CASE SUMMARY")
        print("="*80)
        
        # Print key sections with proper formatting
        print("\n📜 THE CASE:")
        print("-" * 50)
        print(textwrap.fill(self.truncate_text(self.case_data['description'], 200), width=80))
        
        print("\n🔍 KEY PERCEPTIONS:")
        print("-" * 50)
        perceptions = self.case_data['perceptions'].split('\n')
        for i, perc in enumerate(perceptions[:5]):  # Limit to first 5 perceptions
            if perc.strip():
                print(f"• {perc.strip()}")
        if len(perceptions) > 5:
            print("• ...")
            
        print("\n💡 MOST PROBABLE HYPOTHESIS:")
        print("-" * 50)
        print(textwrap.fill(self.truncate_text(self.case_data['best_hypothesis'], 300), width=80))
        
        print("\n🎭 DESI HOLMES' SOLUTION:")
        print("-" * 50)
        print(self.truncate_text(self.case_data['solution'], 500))
        
        print("\n" + "="*80)
        print(f"Full report saved to file: {self.last_saved_filename}")
        print("="*80 + "\n")

# Input function for case description
def get_case_description():
    print("\n🎩 DESI HOLMES AI DETECTIVE AGENCY 🔎")
    print("\nWelcome! Please describe the mystery you'd like Desi Holmes to solve.")
    print("You can enter a detailed case description below (type 'END' on a new line when finished):\n")
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    
    return "\n".join(lines)

# Example with pre-defined cases
def get_sample_case_option():
    print("\n" + "="*80)
    print(" "*25 + "🎩 DESI HOLMES AI DETECTIVE AGENCY 🔎")
    print("="*80)
    print("\nWelcome! Would you like to:")
    print("1. Enter your own mystery case")
    print("2. Use a sample case (The Modern Locked Room Mystery)")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == '2':
        return """
        Tech entrepreneur Rajiv Mehta was found dead in his smart home office, which was locked from the inside with no signs of forced entry. The security system shows no one entered or left for 6 hours before his body was discovered. His smartwatch recorded an elevated heart rate at 11:42 PM, followed by a sudden flatline. His laptop was open to an encrypted messaging platform, and his cryptocurrency wallet was emptied minutes before his death. The home's AI assistant logged an unusual voice command at 11:40 PM that was later deleted from the system. Three of his business partners were at a conference in another city, but one had stepped away from the event during the estimated time of death. Rajiv had recently mentioned to friends that he was working on a breakthrough AI model that would "change everything."
        """
    else:
        return get_case_description()

# Main execution
if __name__ == "__main__":
    try:
        agency = DesiHolmesAgency()
        
        # Get case description from user
        case_description = get_sample_case_option()
        
        # Generate a filename based on the first few words of the case
        case_words = case_description.strip().split()[:3]
        filename = "_".join(case_words).lower().replace(",", "").replace(".", "") + "_case_report.md"
        agency.last_saved_filename = filename
        
        # Analyze the case
        result = agency.analyze_case(case_description)
        agency.save_report(filename)
        
        # Print summary to console
        agency.print_summary()
    
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try again with a shorter case description or contact support.")