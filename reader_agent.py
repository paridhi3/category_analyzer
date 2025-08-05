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
        "Then write a detailed, comprehensive and well-structured summary that captures all important details:\n\n"
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
def extract_project_title(text: str) -> str:
    """
    Extracts the project title that describes the initiative.
    """
    prompt = (
        "Read the case study and extract the project title.\n"
        "If unavailable, return 'Untitled Project'.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You extract project titles for case studies."},
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
        "Return a comma-separated list (e.g., AWS, Azure, Power BI, Snowflake). If unknown, return 'Not Mentioned'.\n\n"
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
    extract_project_title,
    extract_technology_used,
]

prompt_template = ChatPromptTemplate.from_template("""
You are a metadata extraction agent for case studies. Use tools to extract:

1. Project Title
2. Domain
3. Client name 
4. Technologies used
5. Detailed Summary

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
    project_title = extract_project_title.invoke(case_text)
    domain = detect_domain.invoke(case_text)
    client = extract_client_name.invoke(case_text)
    tech = extract_technology_used.invoke(case_text)
    summary = review_case_study.invoke(case_text)    

    return {
        "file_name": file_name,
        "project_title": project_title,
        "domain": domain,
        "client_name": client,
        "technology_used": tech,
        "summary": summary,
    }
