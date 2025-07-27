# main.py

import os
from tqdm import tqdm

from extractor import extract_text
from categorizer import get_aoai_client, categorize_case_study

def process_files(input_folder):
    client = get_aoai_client()
    results = []
    categories = []  # Empty at start
    for fname in tqdm(os.listdir(input_folder)):
        if fname.lower().endswith(('.pdf', '.pptx')):
            fpath = os.path.join(input_folder, fname)
            text = extract_text(fpath)
            category = categorize_case_study(text, categories, client)
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
