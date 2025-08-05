# agents/categorizer_agent.py

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from config import llm

@tool
def categorize(summary: str, existing_categories: str) -> str:
    """
    Assigns the summary to one of the provided existing categories.
    Output only the category name from the existing list.
    """
    categories = existing_categories.split(",") if existing_categories else []
    cat_str = ", ".join(categories)

    prompt = (
        f"Available categories: {cat_str}\n\n"
        "Read the case study summary below and output ONLY the most suitable category name from the existing list.\n"
        "- Be brief (1–2 words)\n"
        "- Match exactly one category from the list\n"
        "- Do NOT create new categories\n"
        "**IMPORTANT: Output only the category name — no project titles, no description, no explanation, no client name, no sentence, and no multi-line output.**\n\n"
        "Case Study Summary:\n"
        f"{summary}"
    )

    output = llm.invoke([
        {"role": "system", "content": "You are an expert case study categorizer."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()

tools = [categorize]

prompt_template = ChatPromptTemplate.from_template("""
You are a categorization agent. Choose a fitting category for the given case study summary.

Case Study Summary:
{text}
Existing Categories: {existing_categories}
{agent_scratchpad}
""")

agent = create_tool_calling_agent(llm, tools, prompt_template)
categorizer_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_categorization_agent(summary: str, existing_categories: list = None) -> str:
    existing_categories = existing_categories or [
        "Digital Transformation",
        "Data Migration",
        "Data Governance",
        "Data Quality",
        "Data Security",
        "Data Analytics", 
        "Data Integration",
        "Master Data Management",
    ]
    
    existing_categories_str = ", ".join(existing_categories)
    
    return categorizer_executor.invoke({
        "text": summary,
        "existing_categories": existing_categories_str
    })["output"]
