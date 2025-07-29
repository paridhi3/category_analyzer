import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Step 1: Setup LLM
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_END_POINT"),
    openai_api_version=os.getenv("API_VERSION"),
    deployment_name=os.getenv("MODEL_NAME"),
    temperature=0
)

# Step 2: Define a tool the agent can use
@tool
def categorize(text: str, existing_categories: str) -> str:
    """
    Assigns the given case study text to an existing category if it matches.
    If it doesn't match, generates a new, brief, clear category name that describes the subject area.
    Gives only the category name and no description or explanation.
    """
    categories = existing_categories.split(",") if existing_categories else []
    cat_str = ", ".join(categories)
    instruction = (
        f"Existing categories: {cat_str}\n"
        "Read the case study and generate a generic, brief, clear, category name that describes its subject area."
        "The category name should not be more than 2 words."
        "Put similar case studies under one category."
        "Respond with ONLY the category name." 
        "DO NOT give any description or explanation.\n\nCase Study:\n"
    )
    prompt = instruction + text[:3500]

    output = llm.invoke([
        {"role": "system", "content": "You are an expert in categorizing case studies."},
        {"role": "user", "content": prompt}
    ])
    return output.content.strip()

# Step 3: Wrap tool
tools = [categorize]

# Step 4: Create prompt
prompt_template = ChatPromptTemplate.from_template("""
You are a categorization agent. Your goal is to process case study text and choose a fitting category that describes the subject area.
                                                   
Case Study: {text}
Existing Categories: {existing_categories}                             
{agent_scratchpad}
""")

# Step 5: Create agent + executor
agent = create_tool_calling_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 6: Function to run agent
def run_categorization_agent(case_text, existing_categories):
    inputs = {
        "text": case_text,
        "existing_categories": ",".join(existing_categories)
    }
    return agent_executor.invoke(inputs)["output"]



# # categorizer.py

# from openai import AzureOpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def get_aoai_client():
#     return AzureOpenAI(
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version=os.getenv("API_VERSION"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_END_POINT")
#     )

# def categorize_case_study(text, existing_categories, client):
#     if existing_categories:
#         categories_str = ", ".join(existing_categories)
#         instruction = (
#             f"Existing categories: {categories_str}\n"
#             "Given the following case study, assign it to ONE of these existing categories if it fits."
#             "If not, create a NEW, generic, brief, clear category name and assign it to that. The category name should not be more than two words. "
#             "Respond with only the category name you choose.\n\nCase Study Text:\n"
#         )
#     else:
#         instruction = (
#             "There are currently no categories. Please read the case study and create a generic, brief, clear category name "
#             "that best describes its subject area. The category name should not be more than two words. Respond with ONLY the category name.\n\nCase Study Text:\n"
#         )
#     prompt = instruction + text[:3500]  # Truncate to avoid model limits

#     response = client.chat.completions.create(
#         model=os.getenv("MODEL_NAME"),
#         messages=[
#             {"role": "system", "content": "You are an expert at creating and assigning generic categories for case studies."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0
#     )
#     return response.choices[0].message.content.strip()
