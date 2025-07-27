# main.py

import os
from tqdm import tqdm

from extractor import extract_text
from agent import get_categorizer_agent  # New: LangChain agent

def process_files(input_folder):
    agent = get_categorizer_agent()  # Get the LangChain agent
    results = []
    categories = []  # Empty at start

    for fname in tqdm(os.listdir(input_folder)):
        if fname.lower().endswith(('.pdf', '.pptx')):
            fpath = os.path.join(input_folder, fname)
            text = extract_text(fpath)

            # Create the input prompt for the agent
            if categories:
                category_prompt = (
                    f"Existing categories: {', '.join(categories)}\n"
                    "Given this case study, assign it to one of the existing categories "
                    "or create a new brief, generic two-word category:\n\n"
                )
            else:
                category_prompt = (
                    "There are currently no categories.\n"
                    "Please assign a brief, generic two-word category to this case study:\n\n"
                )

            prompt = category_prompt + text[:3500]  # Truncate if needed
            category = agent.invoke(prompt).strip()

            if category not in categories:
                categories.append(category)

            results.append({"filename": fname, "category": category})
    return results, categories

if __name__ == "__main__":
    results, all_categories = process_files("case_studies")
    print("\nCategorization Results:")
    for item in results:
        print(f"{item['filename']}: {item['category']}")
    print("\nFinal Categories Identified:")
    for c in all_categories:
        print("-", c)
