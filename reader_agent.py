# reader_agent.py

from llm_config import llm
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

@tool
def summarize_case_study(text: str) -> str:
    """
    Reads the case study and summarizes the key subject area and topic.
    Output should be concise (3–5 sentences max).
    """
    prompt = (
        "Summarize the subject area and key topic of the following case study in 3–5 sentences.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You are a summarization agent for case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()

# ... rest remains the same
