# agents/validation_agent.py

import json
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from config import llm

# === Tool ===
@tool
def estimate_confidence_score(field_name: str, field_value: str, case_text: str) -> str:
    """
    Estimate confidence score (0.0 to 1.0) for the extracted field.
    """
    prompt = (
        f"You are a metadata validation agent.\n\n"
        f"The field you are validating is: '{field_name}'\n"
        f"Extracted value: '{field_value}'\n\n"
        "Given the case study text below, how confident are you that the value is accurate and relevant?\n"
        "- Output ONLY a float score between 0.0 and 1.0\n"
        "- Do NOT include explanations or text\n\n"
        f"Case Study Text:\n{case_text[:3500]}"
    )

    response = llm.invoke([
        {"role": "system", "content": "You are a strict and concise validator."},
        {"role": "user", "content": prompt}
    ])

    try:
        score = float(response.content.strip())
        return str(min(max(score, 0.0), 1.0))  # Clamp score between 0 and 1
    except:
        return "0.5"  # Default fallback confidence

# === Tool List ===
tools = [estimate_confidence_score]

# === Prompt Template ===
prompt_template = ChatPromptTemplate.from_template("""
You are a validation agent. Use tools to estimate confidence scores for metadata extracted from case studies.

Inputs:
- Field name
- Field value
- Full case study text

Return a float score between 0.0 and 1.0.
{agent_scratchpad}
""")

# === Create AgentExecutor ===
agent = create_tool_calling_agent(llm, tools, prompt_template)
validation_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# === Utility Function ===
def run_validation_agent(metadata: dict, case_text: str) -> dict:
    """
    Returns confidence scores for each metadata field.
    Example Output:
    {
        "project_title": 0.91,
        "client_name": 0.67,
        ...
    }
    """
    fields_to_check = ["project_title", "client_name", "domain", "technology_used", "summary"]
    scores = {}

    for field in fields_to_check:
        value = metadata.get(field, "")
        result = validation_executor.invoke({
            "field_name": field.replace("_", " ").title(),
            "field_value": value,
            "case_text": case_text
        })
        try:
            scores[field] = float(result["output"])
        except:
            scores[field] = 0.5  # fallback

    return scores
