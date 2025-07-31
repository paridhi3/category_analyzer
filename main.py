# main.py
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from agents.reader_agent import run_reader_agent
from agents.categorizer_agent import run_categorization_agent
import pandas as pd

from config import llm, loader, embedding_model

# === Streamlit Setup ===
st.set_page_config(page_title="Case Study Categorizer", layout="wide")
st.title("📁 Case Study Categorizer")

# === Load Documents from Azure Blob Storage ===
@st.cache_resource
def load_documents():
    return loader.load()

documents = load_documents()

# === Prepare supported files ===
supported_files = {}
for doc in documents:
    source_path = doc.metadata.get("source", "Unknown source")
    file_name = os.path.basename(source_path)
    if file_name.lower().endswith(('.pdf', '.pptx')):
        supported_files[file_name] = doc.page_content

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
        llm=llm,
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

# === Tabs for UI ===
tab1, tab2, tab3 = st.tabs(["🗂 File Categories", "📄 File Summary", "💬 Chat with File"])

# === Tab 1: File Categories Table ===
with tab1:
    st.subheader("🗂 File Categories")
    df = pd.DataFrame(results)[["File Name", "Category"]]
    st.dataframe(df, use_container_width=True)

# === Tab 2: File Summary Viewer ===
with tab2:
    st.subheader("📄 View File Summary")
    summary_files = [r["File Name"] for r in results]
    selected_summary_file = st.selectbox("Choose a file to view summary:", summary_files, key="summary_file")

    if selected_summary_file:
        summary = next((r["Summary"] for r in results if r["File Name"] == selected_summary_file), None)
        if summary:
            st.markdown(f"### Summary of `{selected_summary_file}`")
            st.markdown(summary)

# === Tab 3: File Chatbot ===
with tab3:
    st.subheader("💬 Ask Questions About a File")
    selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

    if selected_chat_file:
        file_key = selected_chat_file.replace(".", "_")

        # Initialize session state
        if f"qa_chain_{file_key}" not in st.session_state:
            file_text = supported_files[selected_chat_file]
            vectorstore = create_vector_store(file_text, file_key)
            qa_chain = get_qa_chain(vectorstore)
            st.session_state[f"qa_chain_{file_key}"] = qa_chain
            st.session_state[f"chat_history_{file_key}"] = []

        qa_chain = st.session_state[f"qa_chain_{file_key}"]
        chat_history = st.session_state[f"chat_history_{file_key}"]

        # Reset Button
        if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
            chat_history.clear()
            st.rerun()

        # Show chat history
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        user_input = st.chat_input("Type your question:")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            response = qa_chain({"query": user_input})
            answer = response.get("result", "❌ No answer found.")

            with st.chat_message("assistant"):
                st.markdown(answer)

            # Save to session
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})
