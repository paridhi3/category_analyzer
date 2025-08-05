# agents/self_healing_agent.py

import json
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from config import llm

# === Tool ===
@tool
def heal_metadata_field(field_name: str, case_text: str) -> str:
    """
    Attempts to regenerate a more accurate metadata value for a given field using the case study.
    """
    prompt = (
        f"You are a metadata correction agent.\n\n"
        f"Field to regenerate: '{field_name}'\n\n"
        "Analyze the following case study and return the best possible value for the field.\n"
        "- If unsure, return 'Not Available'\n"
        "- Do NOT explain. Just output the value.\n\n"
        f"Case Study:\n{case_text[:3500]}"
    )

    response = llm.invoke([
        {"role": "system", "content": "You are a precise extractor for metadata fields."},
        {"role": "user", "content": prompt}
    ])

    return response.content.strip()

# === Tool List ===
tools = [heal_metadata_field]

# === Prompt Template ===
prompt_template = ChatPromptTemplate.from_template("""
You are a self-healing metadata agent. Your task is to re-extract low-confidence metadata fields from a case study.

Inputs:
- Field name
- Case study text

Return a corrected field value.
{agent_scratchpad}
""")

# === Create AgentExecutor ===
agent = create_tool_calling_agent(llm, tools, prompt_template)
healing_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# === Utility Function ===
def run_self_healing_agent(metadata: dict, confidence_scores: dict, case_text: str, threshold: float = 0.75) -> dict:
    """
    For any metadata field with confidence < threshold, regenerate it using LLM agent.
    Returns the updated metadata dictionary.
    """
    healed_metadata = metadata.copy()

    for field, score in confidence_scores.items():
        if score < threshold:
            result = healing_executor.invoke({
                "field_name": field.replace("_", " ").title(),
                "case_text": case_text
            })
            healed_metadata[field] = result["output"]

    return healed_metadata
