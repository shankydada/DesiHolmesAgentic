from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools

load_dotenv()

agent=Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_too_calls=True,
    markdown=True,
    instructions=["Use tables to display data."]
)
agent.print_response("Summarize and compare analyst recommendation and fundamentals for TSLA and NVDA")