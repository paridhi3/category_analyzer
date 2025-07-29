# agents/categorizer_agent.py

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from llm_config import llm

@tool
def categorize(summary: str, existing_categories: str) -> str:
    """
    Assigns the summary to an existing category or creates a new one.
    Output only the category name.
    """
    categories = existing_categories.split(",") if existing_categories else []
    cat_str = ", ".join(categories)

    prompt = (
        f"Existing categories: {cat_str}\n\n"
        "Read the case study summary and output ONLY the most suitable category name.\n"
        "- Be brief (1–2 words)\n"
        "- Be generic (not client-specific)\n"
        "- Match existing if similar exists\n"
        "- Otherwise create a new generic category name\n\n"
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

def run_categorization_agent(summary: str, existing_categories: list) -> str:
    return categorizer_executor.invoke({
        "text": summary,
        "existing_categories": ",".join(existing_categories)
    })["output"]
