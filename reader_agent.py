# agents/reader_agent.py

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from llm_config import llm

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

tools = [summarize_case_study]

prompt_template = ChatPromptTemplate.from_template("""
You are a summarization agent. Your task is to extract the main subject area and key points from the case study.

Case Study:
{text}
{agent_scratchpad}
""")

agent = create_tool_calling_agent(llm, tools, prompt_template)
reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_reader_agent(case_text: str) -> str:
    return reader_executor.invoke({"text": case_text})["output"]
