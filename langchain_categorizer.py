import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
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
    If it doesn't match, generates a new, clear, 2-word category.
    """
    categories = existing_categories.split(",") if existing_categories else []
    if categories:
        cat_str = ", ".join(categories)
        instruction = (
            f"Existing categories: {cat_str}\n"
            "Assign the following case study to ONE of these existing categories if it fits. "
            "If not, create a NEW, brief, clear category name (max 2 words). "
            "Respond with ONLY the category name.\n\nCase Study:\n"
        )
    else:
        instruction = (
            "No categories exist yet. Read the case study and generate a clear, 2-word category name "
            "that describes its subject. Respond with ONLY the category name.\n\nCase Study:\n"
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
prompt_template = PromptTemplate.from_template("""
You are a categorization agent. Your goal is to process case study text and choose a fitting category from a provided list,
or create a new one if no good fit exists.
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
