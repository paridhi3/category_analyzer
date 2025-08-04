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
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import shutil
from langchain_core.documents import Document

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

def load_or_create_vector_store(text, file_name):
    safe_name = sanitize_name(file_name)
    persist_dir = f".chromadb_{safe_name}"

    # If the vector store already exists, just load it
    if os.path.exists(persist_dir):
        return Chroma(
            collection_name=safe_name,
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
    # Otherwise, create it from text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    vector_store = Chroma.from_texts(
        chunks,
        embedding=embedding_model,
        collection_name=safe_name,
        persist_directory=persist_dir
    )
    vector_store.persist()
    return vector_store

def load_or_create_metadata_vectorstore(metadata_cache):
    safe_name = "metadata_vectorstore"
    persist_dir = f".chromadb_{safe_name}"
    
    # If vector store already exists
    if os.path.exists(persist_dir):
        vs = Chroma(
            collection_name=safe_name,
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
        # Check if the document count matches metadata cache
        if len(vs.get()["documents"]) == len(metadata_cache):
            return vs
        else:
            # Rebuild vectorstore if new/updated metadata is detected
            shutil.rmtree(persist_dir)

    # Create new Chroma vectorstore from metadata
    documents = []
    for metadata in metadata_cache.values():
        text_for_embedding = (
            f"File Name: {metadata.get('file_name', '')}\n"
            f"Category: {metadata.get('category', '')}\n"
            f"Domain: {metadata.get('domain', '')}\n"
            f"Project Title: {metadata.get('project_title', '')}\n"
            f"Technologies used: {metadata.get('technology_used', '')}\n"
            f"Client Name: {metadata.get('client_name', '')}\n"
            f"Summary: {metadata.get('summary', '')}"
        )
        doc = Document(page_content=text_for_embedding, metadata=metadata)
        documents.append(doc)

    vs = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=safe_name,
        persist_directory=persist_dir,
    )
    vs.persist()
    return vs


# === Prompt for Tab 3: File-specific Q&A ===
file_chat_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are an assistant answering questions based on extracted file metadata.
Answer the questions based on the selected file name.

Context:
{context}

Question:
{query}

Answer as clearly and concisely as possible. If the answer isn't in the context, say so explicitly.
""",
)

# === Prompt for Tab 4: Cross-file metadata Q&A ===
cross_file_chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant with access to metadata summaries of case study files. "
        "You should base your answers only on the summaries, categories, domains, and technologies used."
    ),
    HumanMessagePromptTemplate.from_template(
        "Here is some metadata information:\n{context}\n\nQuestion: {query}"
    )
])

# === Function to create QA chain with prompt ===
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
        return_source_documents=False,
        chain_type_kwargs={"prompt": file_chat_prompt}
    )

def get_cross_file_chain(vectorstore):
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
        return_source_documents=False,
        chain_type_kwargs={"prompt": cross_file_chat_prompt}
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
        "file_name": agent_output["file_name"],
        "category": category,
        "domain": agent_output["domain"],
        "technology_used": agent_output["technology_used"],
        "project_title": agent_output["project_title"],
        "client_name": agent_output["client_name"],
        "summary": agent_output["summary"],
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
    metadata_cache.update({file: process_file(file, text) for file, text in supported_files.items()})
    if metadata_cache:
        df = pd.DataFrame(metadata_cache.values())[["file_name", "category", "domain", "client_name", "technology_used"]]
        df = df.rename(columns={
            "file_name": "File Name",
            "category": "Category",
            "domain": "Domain",
            "client_name": "Client Name",
            "technology_used": "Technologies Used",  
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
        if selected_summary_file not in metadata_cache:
            metadata_cache[selected_summary_file] = process_file(selected_summary_file, supported_files[selected_summary_file])

        summary = metadata_cache[selected_summary_file]["summary"]
        domain = metadata_cache[selected_summary_file]["domain"]
        category = metadata_cache[selected_summary_file]["category"]

        st.markdown(f"### Summary of `{selected_summary_file}`")
        st.info(f"**Category:** {category} | **Domain:** {domain}")
        st.markdown(summary)

# === Tab 3: File Chatbot ===
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

with tab3:
    st.subheader("💬 Ask Questions About a File")
    summary_files = list(supported_files.keys())
    selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

    if selected_chat_file:
        file_key = selected_chat_file.replace(".", "_")

        if f"qa_chain_{file_key}" not in st.session_state:
            meta = metadata_cache.get(selected_chat_file)
            if not meta:
                st.warning(f"No metadata found for {selected_chat_file}. Please process the file first.")
                st.stop()

            # Prepare text for embedding
            text_for_embedding = (
                f"File Name: {meta.get('file_name', '')}\n"
                f"Category: {meta.get('category', '')}\n"
                f"Domain: {meta.get('domain', '')}\n"
                f"Project Title: {meta.get('project_title', '')}\n"
                f"Technologies used: {meta.get('technology_used', '')}\n"
                f"Client Name: {meta.get('client_name', '')}\n"
                f"Summary:\n{meta.get('summary', '')}"
            )

            # Create a one-file vectorstore in memory
            doc = Document(
                page_content=text_for_embedding,
                metadata={"file_name": selected_chat_file}
            )
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents([doc])

            vectorstore = Chroma.from_documents(
                docs,
                embedding=embedding_model,
                collection_name=f"metadata_{file_key}",
                persist_directory=None  # In-memory only
            )

            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False
            )

            st.session_state[f"qa_chain_{file_key}"] = qa_chain
            st.session_state[f"chat_history_{file_key}"] = []

        qa_chain = st.session_state[f"qa_chain_{file_key}"]
        chat_history = st.session_state[f"chat_history_{file_key}"]

        if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
            chat_history.clear()
            st.experimental_rerun()

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
    if not metadata_cache:
        st.warning("Please run the categorization first.")
    else:
        df = pd.DataFrame(metadata_cache.values())
        meta_texts = [
            f"File: {row['file_name']}\n"
            f"Project Title: {row['project_title']}"
            f"\nCategory: {row['category']}\n"
            f"Domain: {row['domain']}\n"
            f"Technologies used: {row['technology_used']}\n"
            f"Summary: {row['summary']}"
            
            for _, row in df.iterrows()
        ]
        vs = load_or_create_metadata_vectorstore(metadata_cache)
        cross_qa = get_cross_file_chain(vs)

        cross_query = st.chat_input("Ask a question across all case studies:")
        if cross_query:
            with st.spinner("Thinking..."):
                result = cross_qa.run(cross_query)
                st.markdown(result)
