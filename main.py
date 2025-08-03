# # main.py
# import os
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import RetrievalQA
# from agents.reader_agent import run_reader_agent
# from agents.categorizer_agent import run_categorization_agent
# import pandas as pd

# from config import llm, loader, embedding_model

# # === Streamlit Setup ===
# st.set_page_config(page_title="Case Study Categorizer", layout="wide")
# st.title("📁 Case Study Categorizer")

# # === Load Documents from Azure Blob Storage ===
# @st.cache_resource
# def load_documents():
#     return loader.load()

# documents = load_documents()

# # === Prepare supported files ===
# supported_files = {}
# for doc in documents:
#     source_path = doc.metadata.get("source", "Unknown source")
#     file_name = os.path.basename(source_path)
#     if file_name.lower().endswith(('.pdf', '.pptx')):
#         supported_files[file_name] = doc.page_content

# # === create vector store per file after sanitizing file name ===
# import re

# def sanitize_name(file_name: str) -> str:
#     # Remove extension and replace invalid characters with underscores
#     name = os.path.splitext(file_name)[0]
#     name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
#     name = name.strip("_")  # Remove leading/trailing underscores
#     return f"ragstore_{name}"  # Ensure name starts with valid char

# def create_vector_store(text, file_name):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_text(text)

#     safe_name = sanitize_name(file_name)

#     vector_store = Chroma.from_texts(
#         chunks,
#         embedding=embedding_model,
#         collection_name=safe_name,
#         persist_directory=f".chromadb_{safe_name}"
#     )
#     vector_store.persist()
#     return vector_store


# def get_qa_chain(vectorstore):
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         input_key="query",
#         k=3
#     )
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(),
#         memory=memory,
#         return_source_documents=False
#     )

# # === Caching per-file processing ===
# @st.cache_data(show_spinner="🔍 Processing file...", ttl=3600)
# def process_file(file_name, text):
#     agent_output = run_reader_agent(text)
#     category = run_categorization_agent(agent_output["summary"])
#     return {
#         "File Name": file_name,
#         "Category": category,
#         "Domain": agent_output["domain"],
#         "Summary": agent_output["summary"]
#     }

# results_cache = {}

# # === Tabs for UI ===
# tab1, tab2, tab3, tab4 = st.tabs(["🗂 File Categories", "📄 File Summary", "💬 Chat with File", "📊 Cross-File Chat"])

# # === Tab 1: File Categories Table ===
# with tab1:
#     st.subheader("🗂 File Categories")
#     results_cache.update({file: process_file(file, text) for file, text in supported_files.items()})
#     if results_cache:
#         df = pd.DataFrame(results_cache.values())[
#             ["File Name", "Category", "Domain"]
#         ]
#         st.dataframe(df, use_container_width=True)
#     else:
#         st.info("Click the button above to process files.")


# # === Tab 2: File Summary Viewer ===
# with tab2:
#     st.subheader("📄 View File Summary")
#     summary_files = list(supported_files.keys())
#     selected_summary_file = st.selectbox("Choose a file to view summary:", summary_files, key="summary_file")

#     if selected_summary_file:
#         # Run agent only if not already cached
#         if selected_summary_file not in results_cache:
#             results_cache[selected_summary_file] = process_file(selected_summary_file, supported_files[selected_summary_file])

#         summary = results_cache[selected_summary_file]["Summary"]
#         domain = results_cache[selected_summary_file]["Domain"]
#         category = results_cache[selected_summary_file]["Category"]

#         st.markdown(f"### Summary of `{selected_summary_file}`")
#         st.markdown(summary)
#         st.info(f"**Category:** {category} | **Domain:** {domain}")


# # === Tab 3: File Chatbot ===
# with tab3:
#     st.subheader("💬 Ask Questions About a File")
#     selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

#     if selected_chat_file:
#         file_key = selected_chat_file.replace(".", "_")

#         # Initialize session state
#         if f"qa_chain_{file_key}" not in st.session_state:
#             file_text = supported_files[selected_chat_file]
#             vectorstore = create_vector_store(file_text, file_key)
#             qa_chain = get_qa_chain(vectorstore)
#             st.session_state[f"qa_chain_{file_key}"] = qa_chain
#             st.session_state[f"chat_history_{file_key}"] = []

#         qa_chain = st.session_state[f"qa_chain_{file_key}"]
#         chat_history = st.session_state[f"chat_history_{file_key}"]

#         # Reset Button
#         if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
#             chat_history.clear()
#             st.rerun()

#         # Show chat history
#         for msg in chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         # Chat Input
#         user_input = st.chat_input("Type your question:")
#         if user_input:
#             with st.chat_message("user"):
#                 st.markdown(user_input)

#             response = qa_chain({"query": user_input})
#             answer = response.get("result", "❌ No answer found.")

#             with st.chat_message("assistant"):
#                 st.markdown(answer)

#             # Save to session
#             chat_history.append({"role": "user", "content": user_input})
#             chat_history.append({"role": "assistant", "content": answer})

# with tab4:
#     st.subheader("📊 Ask Questions Across All Files")
#     if not results_cache:
#         st.warning("Please run the categorization first.")
#     else:
#         df = pd.DataFrame(results_cache.values())
#         meta_texts = [
#             f"{row['Summary']}\nFile: {row['File Name']}\nCategory: {row['Category']}\nDomain: {row['Domain']}"
#             for _, row in df.iterrows()
#         ]
#         vs = Chroma.from_texts(meta_texts, embedding=embedding_model, collection_name="case_meta", persist_directory=".chromadb_case_meta")
#         cross_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever())
#         cross_query = st.text_input("Ask a question across all case studies:")
#         if cross_query:
#             with st.spinner("Thinking..."):
#                 result = cross_qa.run(cross_query)
#                 st.markdown(result)

# main.py
import os
import re
import json
import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

from agents.reader_agent import run_reader_agent
from agents.categorizer_agent import run_categorization_agent
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

# === Create vector store per file ===
def sanitize_name(file_name: str) -> str:
    name = os.path.splitext(file_name)[0]
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name).strip("_")
    return f"ragstore_{name}"

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

# === Load & Save Metadata JSON ===
METADATA_FILE = "metadata.json"

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# === Process File with Agents (only if not in cache) ===
metadata_cache = load_metadata()

@st.cache_data(show_spinner="🔍 Processing file...", ttl=3600)
def process_file(file_name, text):
    if file_name in metadata_cache:
        return metadata_cache[file_name]

    agent_output = run_reader_agent(text)
    category = run_categorization_agent(agent_output["summary"])
    result = {
        "File Name": file_name,
        "Category": category,
        "Domain": agent_output["domain"],
        "Summary": agent_output["summary"]
    }
    metadata_cache[file_name] = result
    save_metadata(metadata_cache)
    return result

results_cache = {}

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["🗂 File Categories", "📄 File Summary", "💬 Chat with File", "📊 Cross-File Chat"])

# === Tab 1: File Categories Table ===
with tab1:
    st.subheader("🗂 File Categories")
    results_cache.update({file: process_file(file, text) for file, text in supported_files.items()})
    if results_cache:
        df = pd.DataFrame(results_cache.values())[["file_name", "category", "domain"]]
        df = df.rename(columns={
            "file_name": "File Name",
            "category": "Category",
            "domain": "Domain"
        })
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Click the button above to process files.")

# === Tab 2: File Summary Viewer ===
with tab2:
    st.subheader("📄 View File Summary")
    summary_files = list(supported_files.keys())
    selected_summary_file = st.selectbox("Choose a file to view summary:", summary_files, key="summary_file")

    if selected_summary_file:
        if selected_summary_file not in results_cache:
            results_cache[selected_summary_file] = process_file(selected_summary_file, supported_files[selected_summary_file])

        summary = results_cache[selected_summary_file]["summary"]
        domain = results_cache[selected_summary_file]["domain"]
        category = results_cache[selected_summary_file]["category"]

        st.markdown(f"### Summary of `{selected_summary_file}`")
        st.info(f"**Category:** {category} | **Domain:** {domain}")
        st.markdown(summary)

# === Tab 3: File Chatbot ===
with tab3:
    st.subheader("💬 Ask Questions About a File")
    selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

    if selected_chat_file:
        file_key = selected_chat_file.replace(".", "_")

        if f"qa_chain_{file_key}" not in st.session_state:
            file_text = supported_files[selected_chat_file]
            vectorstore = create_vector_store(file_text, file_key)
            qa_chain = get_qa_chain(vectorstore)
            st.session_state[f"qa_chain_{file_key}"] = qa_chain
            st.session_state[f"chat_history_{file_key}"] = []

        qa_chain = st.session_state[f"qa_chain_{file_key}"]
        chat_history = st.session_state[f"chat_history_{file_key}"]

        if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
            chat_history.clear()
            st.rerun()

        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Type your question:")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            response = qa_chain({"query": user_input})
            answer = response.get("result", "❌ No answer found.")

            with st.chat_message("assistant"):
                st.markdown(answer)

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})

# === Tab 4: Cross-File Chat ===
with tab4:
    st.subheader("📊 Ask Questions Across All Files")
    if not results_cache:
        st.warning("Please run the categorization first.")
    else:
        df = pd.DataFrame(results_cache.values())
        meta_texts = [
            f"{row['summary']}\nFile: {row['file_name']}\nCategory: {row['category']}\nDomain: {row['domain']}"
            for _, row in df.iterrows()
        ]
        vs = Chroma.from_texts(meta_texts, embedding=embedding_model, collection_name="case_meta", persist_directory=".chromadb_case_meta")
        cross_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever())
        cross_query = st.text_input("Ask a question across all case studies:")
        if cross_query:
            with st.spinner("Thinking..."):
                result = cross_qa.run(cross_query)
                st.markdown(result)
