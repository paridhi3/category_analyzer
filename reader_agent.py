# # agents/reader_agent.py

# from langchain_core.tools import tool
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from config import llm


# @tool
# def review_case_study(text: str) -> str:
#     """
#     Reviews the case study and returns a detailed summary.
#     """
#     prompt = (
#         "Carefully read the case study below.\n\n"
#         "Then write a detailed, comprehensive and well-structured summary that captures all important details:\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You are a summarization expert."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()


# @tool
# def detect_domain(text: str) -> str:
#     """
#     Detects the business domain (like Finance, Healthcare, HR, etc.) from the case study.
#     """
#     prompt = (
#         "Identify the business domain of the following case study.\n"
#         "Output ONLY the domain name (1–2 words), e.g., Finance, HR, Retail.\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You are a domain classification expert."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()


# @tool
# def extract_client_name(text: str) -> str:
#     """
#     Extracts the client name mentioned in the case study.
#     """
#     prompt = (
#         "Read the case study and extract the name of the client company.\n"
#         "If not clearly mentioned, return 'Unknown'.\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You extract client names from business documents."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()


# @tool
# def extract_project_title(text: str) -> str:
#     """
#     Extracts the project title that describes the initiative.
#     """
#     prompt = (
#         "Read the case study and extract the project title.\n"
#         "If unavailable, return 'Untitled Project'.\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You extract project titles for case studies."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()


# @tool
# def extract_technology_used(text: str) -> str:
#     """
#     Extracts a list of technologies, platforms, or tools used in the case study.
#     """
#     prompt = (
#         "Identify the technologies, platforms, or tools used in the following case study.\n"
#         "Return a comma-separated list (e.g., AWS, Azure, Power BI, Snowflake). If unknown, return 'Not Mentioned'.\n\n"
#         f"{text[:3500]}"
#     )
#     output = llm.invoke([
#         {"role": "system", "content": "You extract technologies from case studies."},
#         {"role": "user", "content": prompt}
#     ])
#     return output.content.strip()


# tools = [
#     review_case_study,
#     detect_domain,
#     extract_client_name,
#     extract_project_title,
#     extract_technology_used,
# ]

# prompt_template = ChatPromptTemplate.from_template("""
# You are a metadata extraction agent for case studies. Use tools to extract:

# 1. Project Title
# 2. Domain
# 3. Client name 
# 4. Technologies used
# 5. Detailed Summary

# Case Study:
# {text}
# {agent_scratchpad}
# """)

# agent = create_tool_calling_agent(llm, tools, prompt_template)
# reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# def run_reader_agent(case_text: str, file_name: str) -> dict:
#     """
#     Runs metadata extraction agent on case study and returns a JSON dict.
#     """
#     project_title = extract_project_title.invoke(case_text)
#     domain = detect_domain.invoke(case_text)
#     client = extract_client_name.invoke(case_text)
#     tech = extract_technology_used.invoke(case_text)
#     summary = review_case_study.invoke(case_text)    

#     return {
#         "file_name": file_name,
#         "project_title": project_title,
#         "domain": domain,
#         "client_name": client,
#         "technology_used": tech,
#         "summary": summary,
#     }


# agents/reader_agent.py

import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from config import llm


def extract_with_confidence_prompt(field_name: str, instructions: str, text: str) -> str:
    """
    Helper to generate prompt asking for field extraction with confidence score.
    """
    return (
        f"Extract the {field_name} from the following case study.\n"
        f"{instructions}\n"
        "Output a JSON object with keys:\n"
        f"{{ \"{field_name}\": <string>, \"confidence\": <number between 0 and 1> }}\n\n"
        f"Case Study:\n{text[:3500]}"
    )


@tool
def extract_project_title_with_confidence(text: str) -> str:
    prompt = extract_with_confidence_prompt(
        "project_title",
        "If unavailable, return \"Untitled Project\" with confidence 0.",
        text,
    )
    output = llm.invoke([
        {"role": "system", "content": "You extract project titles with confidence from case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def detect_domain_with_confidence(text: str) -> str:
    prompt = extract_with_confidence_prompt(
        "domain",
        "If unknown, return \"Unknown\" with confidence 0.",
        text,
    )
    output = llm.invoke([
        {"role": "system", "content": "You identify business domains with confidence."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def extract_client_name_with_confidence(text: str) -> str:
    prompt = extract_with_confidence_prompt(
        "client_name",
        "If not clearly mentioned, return \"Unknown\" with confidence 0.",
        text,
    )
    output = llm.invoke([
        {"role": "system", "content": "You extract client names with confidence."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def extract_technology_used_with_confidence(text: str) -> str:
    prompt = extract_with_confidence_prompt(
        "technology_used",
        "Return a comma-separated list (e.g., AWS, Azure, Power BI). If unknown, return \"Not Mentioned\" with confidence 0.",
        text,
    )
    output = llm.invoke([
        {"role": "system", "content": "You extract technologies with confidence."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()


@tool
def review_case_study(text: str) -> str:
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


tools = [
    extract_project_title_with_confidence,
    detect_domain_with_confidence,
    extract_client_name_with_confidence,
    extract_technology_used_with_confidence,
    review_case_study,
]

prompt_template = ChatPromptTemplate.from_template("""
You are a metadata extraction agent for case studies. Use tools to extract:

1. Project Title with confidence
2. Domain with confidence
3. Client name with confidence
4. Technologies used with confidence
5. Detailed Summary

Case Study:
{text}
{agent_scratchpad}
""")

agent = create_tool_calling_agent(llm, tools, prompt_template)
reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_reader_agent(case_text: str, file_name: str) -> dict:
    """
    Runs metadata extraction agent on case study and returns a JSON dict with values and confidence.
    """
    def parse_json_or_default(raw_str, default_key, default_value, default_conf=0):
        try:
            parsed = json.loads(raw_str)
            value = parsed.get(default_key, default_value)
            confidence = parsed.get("confidence", default_conf)
            return {"value": value, "confidence": confidence}
        except Exception:
            return {"value": default_value, "confidence": 0}

    project = parse_json_or_default(
        extract_project_title_with_confidence.invoke(case_text),
        "project_title", "Untitled Project"
    )
    domain = parse_json_or_default(
        detect_domain_with_confidence.invoke(case_text),
        "domain", "Unknown"
    )
    client_name = parse_json_or_default(
        extract_client_name_with_confidence.invoke(case_text),
        "client_name", "Unknown"
    )
    technology_used = parse_json_or_default(
        extract_technology_used_with_confidence.invoke(case_text),
        "technology_used", "Not Mentioned"
    )
    summary = review_case_study.invoke(case_text)  # summary has no confidence here

    return {
        "file_name": file_name,
        "project_title": project,
        "domain": domain,
        "client_name": client_name,
        "technology_used": technology_used,
        "summary": summary,
    }
