# categorizer.py

from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_BASE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    DEPLOYMENT_NAME,
)

def get_aoai_client():
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_API_BASE
    )

def categorize_case_study(text, existing_categories, client):
    if existing_categories:
        categories_str = ", ".join(existing_categories)
        instruction = (
            f"Existing categories: {categories_str}\n"
            "Given the following case study, assign it to ONE of these existing categories "
            "if it fits. If not, create a NEW, brief, clear category name and assign it to that. "
            "Respond with only the category name you choose.\n\nCase Study Text:\n"
        )
    else:
        instruction = (
            "There are currently no categories. Please read the case study and create a brief, clear category name "
            "that best describes its subject area. Respond with ONLY the category name.\n\nCase Study Text:\n"
        )
    prompt = instruction + text[:3500]  # Truncate to avoid model limits

    response = client.chat.completions.create(
        deployment_id=DEPLOYMENT_NAME,
        model=None,
        messages=[
            {"role": "system", "content": "You are an expert at creating and assigning categories for case studies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
