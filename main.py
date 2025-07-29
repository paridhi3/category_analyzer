import os
from tqdm import tqdm
from extractor import extract_text
from categorizer import run_categorization_agent
from langchain_community.document_loaders import AzureBlobStorageContainerLoader

loader = AzureBlobStorageContainerLoader(
    conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    container=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
)
documents = loader.load()

def process_documents(documents):
    # print(documents)
    results = []
    categories = []
    
    for doc in tqdm(documents):
        # print(doc.metadata)
        try:
            source_path = doc.metadata.get("source")
            file_name = os.path.basename(source_path)
            # print(file_name)
            # Only process supported formats
            if file_name.lower().endswith(('.pdf', '.pptx')):
                text = doc.page_content or extract_text(doc)
                print(text)
                category = run_categorization_agent(text, categories)
                if category not in categories:
                    categories.append(category)
                results.append({"filename": file_name, "category": category})
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return results, categories

if __name__ == "__main__":
    results, all_categories = process_documents(documents)
    print("\nCategorization Results:")
    for item in results:
        print(f"{item['filename']}: {item['category']}")
    print("\nFinal Categories Identified:")
    for c in all_categories:
        print("-", c)
