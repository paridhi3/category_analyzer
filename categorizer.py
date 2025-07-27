# categorizer.py

from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_aoai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_END_POINT")
    )

def categorize_case_study(text, existing_categories, client):
    if existing_categories:
        categories_str = ", ".join(existing_categories)
        instruction = (
            f"Existing categories: {categories_str}\n"
            "Given the following case study, assign it to ONE of these existing categories if it fits."
            "If not, create a NEW, generic, brief, clear category name and assign it to that. The category name should not be more than two words. "
            "Respond with only the category name you choose.\n\nCase Study Text:\n"
        )
    else:
        instruction = (
            "There are currently no categories. Please read the case study and create a generic, brief, clear category name "
            "that best describes its subject area. The category name should not be more than two words. Respond with ONLY the category name.\n\nCase Study Text:\n"
        )
    prompt = instruction + text[:3500]  # Truncate to avoid model limits

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": "You are an expert at creating and assigning generic categories for case studies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
