# config.py

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.embeddings import AzureOpenAIEmbeddings

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_END_POINT"),
    openai_api_version=os.getenv("API_VERSION"),
    deployment_name=os.getenv("MODEL_NAME"),
    temperature=0
)

loader = AzureBlobStorageContainerLoader(
    conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    container=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
)

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version=os.getenv("API_VERSION"),
    azure_endpoint="https://gen-cim-eas-dep-genai-train-openai.openai.azure.com/",
    chunk_size=500
)
