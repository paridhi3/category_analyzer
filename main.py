# main.py

import os
from tqdm import tqdm
from extractor import extract_text
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from agents.reader_agent import run_reader_agent
from agents.categorizer_agent import run_categorization_agent

# Load documents from Azure Blob Storage
loader = AzureBlobStorageContainerLoader(
    conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    container=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
)
documents = loader.load()

def process_documents(documents):
    results = []
    categories = []

    for doc in tqdm(documents):
        try:
            source_path = doc.metadata.get("source")
            file_name = os.path.basename(source_path)
            if file_name.lower().endswith(('.pdf', '.pptx')):
                text = doc.page_content

                # Agent 1: Summarize
                summary = run_reader_agent(text)
                print(f"\nSummary of {file_name}:\n{summary}")

                # Agent 2: Categorize
                category = run_categorization_agent(summary, categories)
                if category not in categories:
                    categories.append(category)

                results.append({"filename": file_name, "category": category})
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return results, categories

if __name__ == "__main__":
    results, all_categories = process_documents(documents)

    print("\n✅ Categorization Results:")
    for item in results:
        print(f"{item['filename']}: {item['category']}")

    print("\n📚 Final Categories Identified:")
    for c in all_categories:
        print("-", c)
