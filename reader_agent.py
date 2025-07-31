# # agents/reader_agent.py

# from langchain_core.tools import tool
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from config import llm

# @tool
# def review_case_study(text: str) -> str:
#     """
#     Thoroughly reviews the case study and delivers a detailed write-up.
#     """
#     prompt = (
#         "Carefully read and analyze the provided case study. \n\n"
#         "Identify the main themes, key issues, and relevant context. \n\n"
#         "Then, write a comprehensive and well-structured summary that conveys all critical insights\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You are a summarization agent for case studies."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()

# tools = [review_case_study]

# prompt_template = ChatPromptTemplate.from_template("""
# You are tasked with reading the provided case study thoroughly and performing the following steps:
# 1. Identify the Main Subject Area
# 2. Extract Key Points
# 3. Present a Detailed Write-up about the case study

# Case Study:
# {text}
# {agent_scratchpad}
# """)

# agent = create_tool_calling_agent(llm, tools, prompt_template)
# reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# def run_reader_agent(case_text: str) -> str:
#     return reader_executor.invoke({"text": case_text})["output"]
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
        "Carefully read and analyze the provided case study.\n\n"
        "Identify the main themes, key issues, and relevant context.\n\n"
        "Then, write a comprehensive and well-structured summary that conveys all critical insights.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You are a summarization agent for case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()

@tool
def detect_domain(text: str) -> str:
    """
    Detects the business domain (like Finance, Healthcare, HR, etc.) from a case study.
    """
    prompt = (
        "Read the case study below and identify its business domain.\n"
        "Examples: Finance, Healthcare, HR, Retail, Manufacturing, Insurance, Legal, Education, Energy, Government, Telecom, Logistics, Technology\n\n"
        "IMPORTANT: Output ONLY the domain name (1–2 words) with no explanation.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You are a domain classification expert."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()

tools = [review_case_study, detect_domain]

prompt_template = ChatPromptTemplate.from_template("""
You are tasked with reading the case study thoroughly and doing:
1. Summary writing
2. Domain detection
Use the appropriate tools to complete the tasks.

Case Study:
{text}
{agent_scratchpad}
""")

agent = create_tool_calling_agent(llm, tools, prompt_template)
reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_reader_agent(case_text: str) -> dict:
    """
    Runs both summary and domain detection for a given case study.
    Returns a dictionary with both results.
    """
    summary = review_case_study.invoke(case_text)
    domain = detect_domain.invoke(case_text)
    return {"summary": summary, "domain": domain}
