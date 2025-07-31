# agents/reader_agent.py

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from config import llm

@tool
def review_case_study(text: str) -> str:
    """
    Thoroughly reviews the case study and delivers a detailed write-up.
    """
    prompt = (
        "Carefully read and analyze the provided case study. \n\n"
        "Identify the main themes, key issues, and relevant context. \n\n"
        "Then, write a comprehensive and well-structured summary that conveys all critical insights\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You are a summarization agent for case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()

tools = [review_case_study]

prompt_template = ChatPromptTemplate.from_template("""
You are tasked with reading the provided case study thoroughly and performing the following steps:
1. Identify the Main Subject Area
2. Extract Key Points
3. Present a Detailed Write-up about the case study

Case Study:
{text}
{agent_scratchpad}
""")

agent = create_tool_calling_agent(llm, tools, prompt_template)
reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_reader_agent(case_text: str) -> str:
    return reader_executor.invoke({"text": case_text})["output"]
