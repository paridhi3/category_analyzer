import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from llmconfig import llm
 
@tool
def review_case_study(text: str) -> str:
    """
    Extracts and summarizes a document into structured sections including use cases.
    If a section is missing, it will be noted and a suggestion will be provided.
    """
    prompt = (
        "Carefully read and analyze the provided document.\n\n"
        "Extract and summarize the document into these sections:\n"
        "0. Title\n"
        "1. Introduction\n"
        "2. Problem Statement\n"
        "3. Objectives\n"
        "4. Proposed Solution\n"
        "5. Implementation\n"
        "6. Results\n"
        "7. Use Cases\n"
        "8. Conclusion\n"
        "9. Mention the Subject Area\n"
        "10. Tags\n\n"
        "If any section is missing, clearly write 'NA' and suggest what could be included.\n"
        "Return the output in a structured format with section labels.\n\n"
        "Finally make sure you're not missing any of the key information, data insights, statistics or any of the information from the document.\n\n"
        f"{text[:3500]}"
    )
    output = llm.invoke([
        {"role": "system", "content": "You are a specialized document summarization agent. Your task is to extract and organize key insights from business case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()
 
tools = [review_case_study]
 
prompt_template = ChatPromptTemplate.from_template("""
You are tasked with reading the provided document thoroughly and performing the following steps:
1. Extract and summarize the following sections if present:
    - Title (Give Title)\n
    - Subject Area\n
    - Introduction\n
    - Problem Statement\n
    - Objectives\n
    - Proposed Solution\n
    - Implementation\n
    - Results\n
    - Use Cases\n
    - tags\n\n
 
"If any section is missing, clearly write 'NA' and suggest what could be included.\n"
Return the output in a JSON structured format with section labels, NOTE: your output will be directly writen to files and Data Structures, so create accordingly.
 
document:
{text}
{agent_scratchpad}
""")
 
agent = create_tool_calling_agent(llm, tools, prompt_template)
reader_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
 
def run_reader_agent(case_text: str, filename: str) -> str:
    result = reader_executor.invoke({"text": case_text})["output"]
 
    try:
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        parsed_result = {"raw_summary": result}
 
    parsed_result["filename"] = filename  # ✅ Add filename
 
    with open("reader_agent.json", "a", encoding="utf-8") as f:
        json.dump(parsed_result, f)
        f.write("\n")
 
    return json.dumps(parsed_result)
 
 
def run_reader_agent_json(case_text: str, filename: str = "reader_agent.json") -> str:
    result = reader_executor.invoke({"text": case_text})["output"]
 
    try:
        # Parse the JSON string into a dictionary
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        # If parsing fails, store the raw string
        parsed_result = {"raw_summary": result}
 
    
    # Append to reader_agent.json
    filename ="reader_agent.json"
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(parsed_result, f)
        f.write("\n")
 
    return result
