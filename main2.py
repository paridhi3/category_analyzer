# main.py

import os
import streamlit as st
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from agents.reader_agent import run_reader_agent
from agents.categorizer_agent import run_categorization_agent
import pandas as pd

# === Streamlit Setup ===
st.set_page_config(page_title="Case Study Categorizer", layout="wide")
st.title("📁 Case Study Categorizer")

# === Load Documents from Azure Blob Storage ===
@st.cache_resource
def load_documents():
    loader = AzureBlobStorageContainerLoader(
        conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        container=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    )
    return loader.load()

documents = load_documents()

# === Prepare supported files ===
supported_files = {}
for doc in documents:
    source_path = doc.metadata.get("source", "Unknown source")
    file_name = os.path.basename(source_path)
    if file_name.lower().endswith(('.pdf', '.pptx')):
        supported_files[file_name] = doc.page_content

# === Embed Config ===
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_END_POINT")
api_version = os.getenv("API_VERSION")
embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# === create vector store per file after sanitizing file name ===
import re

def sanitize_name(file_name: str) -> str:
    # Remove extension and replace invalid characters with underscores
    name = os.path.splitext(file_name)[0]
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    name = name.strip("_")  # Remove leading/trailing underscores
    return f"ragstore_{name}"  # Ensure name starts with valid char

def create_vector_store(text, file_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version=os.getenv("API_VERSION"),
        azure_endpoint="https://gen-cim-eas-dep-genai-train-openai.openai.azure.com/",
        # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        chunk_size=500
    )

    safe_name = sanitize_name(file_name)

    vector_store = Chroma.from_texts(
        chunks,
        embedding=embedding_model,
        collection_name=safe_name,
        persist_directory=f".chromadb_{safe_name}"
    )
    vector_store.persist()
    return vector_store


def get_qa_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        k=3
    )
    return RetrievalQA.from_chain_type(
        llm=AzureChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            openai_api_version=api_version,
            deployment_name=chat_deployment,
            temperature=0
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False
    )

# === Caching agent outputs ===
@st.cache_data(show_spinner="Running categorization...")
def get_categories_and_summaries():
    results = []
    for file, text in supported_files.items():
        summary = run_reader_agent(text)
        category = run_categorization_agent(summary, [])
        results.append({"File Name": file, "Category": category, "Summary": summary})
    return results

results = get_categories_and_summaries()

# === Show File-Category Table ===
st.subheader("🗂 File Categories")
df = pd.DataFrame(results)[["File Name", "Category"]]
st.dataframe(df, use_container_width=True)

# === Summary Viewer ===
st.subheader("📄 View File Summary")
summary_files = [r["File Name"] for r in results]
selected_summary_file = st.selectbox("Choose a file to view summary:", summary_files)

if selected_summary_file:
    summary = next((r["Summary"] for r in results if r["File Name"] == selected_summary_file), None)
    if summary:
        st.markdown(f"### Summary of `{selected_summary_file}`")
        st.markdown(summary)

# === Chat Section ===
st.subheader("💬 Ask Questions About a File")
selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

if selected_chat_file:
    # Initialize RAG (vector store + QA chain) if not already
    if f"qa_chain_{selected_chat_file}" not in st.session_state:
        file_text = supported_files[selected_chat_file]
        vectorstore = create_vector_store(file_text, selected_chat_file.replace(".", "_"))
        qa_chain = get_qa_chain(vectorstore)
        st.session_state[f"qa_chain_{selected_chat_file}"] = qa_chain
        st.session_state[f"chat_history_{selected_chat_file}"] = []

    user_question = st.chat_input("Ask a question about the selected file:", key="user_question")
    if user_question:
        qa_chain = st.session_state[f"qa_chain_{selected_chat_file}"]
        response = qa_chain({"query": user_question})
        answer = response.get("result", "❌ No answer found.")
        st.session_state[f"chat_history_{selected_chat_file}"].append(
            {"user": user_question, "bot": answer}
        )
        # st.markdown(f"**You asked:** {user_question}")
        # st.markdown(f"**Answer:** {answer}")

    # Show chat history
    with st.expander("📜 Chat History"):
        for i, chat in enumerate(st.session_state[f"chat_history_{selected_chat_file}"], 1):
            st.markdown(f"**You {i}:** {chat['user']}")
            st.markdown(f"**Bot {i}:** {chat['bot']}")
