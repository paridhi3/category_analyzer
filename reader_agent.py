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
#         "Carefully read and analyze the provided case study.\n\n"
#         "Identify the main themes, key issues, and relevant context.\n\n"
#         "Then, write a comprehensive and well-structured summary that conveys all critical insights.\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You are a summarization agent for case studies."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()

# @tool
# def detect_domain(text: str) -> str:
#     """
#     Detects the business domain (like Finance, Healthcare, HR, etc.) from a case study.
#     """
#     prompt = (
#         "Read the case study below and identify its business domain.\n"
#         "Examples: Finance, Healthcare, HR, Retail, Manufacturing, Insurance, Legal, Education, Energy, Government, Telecom, Logistics, Technology\n\n"
#         "IMPORTANT: Output ONLY the domain name (1–2 words) with no explanation.\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You are a domain classification expert."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()

# tools = [review_case_study, detect_domain]

# prompt_template = ChatPromptTemplate.from_template("""
# You are tasked with reading the case study thoroughly and doing:
# 1. Summary writing
# 2. Domain detection
# Use the appropriate tools to complete the tasks.

# Case Study:
# {text}
# {agent_scratchpad}
# """)

# agent = create_tool_calling_agent(llm, tools, prompt_template)
# reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# def run_reader_agent(case_text: str) -> dict:
#     """
#     Runs both summary and domain detection for a given case study.
#     Returns a dictionary with both results.
#     """
#     summary = review_case_study.invoke(case_text)
#     domain = detect_domain.invoke(case_text)
#     return {"summary": summary, "domain": domain}


# agents/reader_agent.py

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from config import llm


@tool
def review_case_study(text: str) -> str:
    """
    Reviews the case study and returns a detailed summary.
    """
    prompt = (
        "Carefully read the case study below.\n\n"
        "Then write a comprehensive and well-structured summary that captures all important details:\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You are a summarization expert."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def detect_domain(text: str) -> str:
    """
    Detects the business domain (like Finance, Healthcare, HR, etc.) from the case study.
    """
    prompt = (
        "Identify the business domain of the following case study.\n"
        "Output ONLY the domain name (1–2 words), e.g., Finance, HR, Retail.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You are a domain classification expert."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def extract_client_name(text: str) -> str:
    """
    Extracts the client name mentioned in the case study.
    """
    prompt = (
        "Read the case study and extract the name of the client company.\n"
        "If not clearly mentioned, return 'Unknown'.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You extract client names from business documents."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def extract_project_name(text: str) -> str:
    """
    Extracts the project name or a suitable title that describes the initiative.
    """
    prompt = (
        "Read the case study and generate a concise project name or title.\n"
        "Avoid long sentences, just 3–6 words. If unavailable, return 'Untitled Project'.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You generate project names for case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def extract_technology_used(text: str) -> str:
    """
    Extracts a list of technologies, platforms, or tools used in the case study.
    """
    prompt = (
        "Identify the technologies, platforms, or tools used in the following case study.\n"
        "Return a comma-separated list (e.g., Azure, Power BI, Snowflake). If unknown, return 'Not Mentioned'.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You extract technologies from case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


tools = [
    review_case_study,
    detect_domain,
    extract_client_name,
    extract_project_name,
    extract_technology_used,
]

prompt_template = ChatPromptTemplate.from_template("""
You are a metadata extraction agent for case studies. Use tools to extract:

1. Summary
2. Domain
3. Client Name
4. Project Name
5. Technology Used

Case Study:
{text}
{agent_scratchpad}
""")

agent = create_tool_calling_agent(llm, tools, prompt_template)
reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_reader_agent(case_text: str, file_name: str) -> dict:
    """
    Runs metadata extraction agent on case study and returns a JSON dict.
    """
    summary = review_case_study.invoke(case_text)
    domain = detect_domain.invoke(case_text)
    client = extract_client_name.invoke(case_text)
    project = extract_project_name.invoke(case_text)
    tech = extract_technology_used.invoke(case_text)

    return {
        "file_name": file_name,
        "summary": summary,
        "domain": domain,
        "client_name": client,
        "project_name": project,
        "technology_used": tech,
    }
