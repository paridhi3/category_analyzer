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
from agents.validation_agent import agent_run_validation
 
from config import llm, loader, embedding_model
 
# === Streamlit Setup ===
import streamlit as st
 
st.set_page_config(page_title="Case Study Categorizer", layout="wide", page_icon="📁")
# === Top Bar: Title + Dark Mode Toggle ===
col1, col2 = st.columns([3, 1])  # Adjust width ratio as needed
 
with col1:
    st.title("📁 Case Study Categorizer")
 
with col2:
    dark_mode = st.toggle("🌙 Dark Mode", key="dark_mode_toggle", label_visibility="visible")
 
if dark_mode:
    st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
 
    /* General text elements */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #ffffff !important;
    }
 
    /* Chat message containers — white box with BLACK text */
    [data-testid="stChatMessage"] {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        font-size: 1rem;
    }
 
    /* Force all nested text inside chat message to be black */
    [data-testid="stChatMessage"] * {
        color: black !important;
    }
 
    /* Dropdown options */
    div[role="option"] {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    /* Override info box styles in dark mode */
    div.stAlert.stAlertInfo {
        background-color: #2a3a4a !important;
        color: #ffffff !important;
        border-left: 4px solid #4CAF50 !important;
        padding: 1rem;
    }
    /* Hover effect for dropdown options */
    div[role="option"]:hover {
        background-color: #4a90e2 !important;  /* nice blue */
        color: #000000 !important;             /* black text */
        cursor: pointer;
    }
 
    /* Chat input box */
    textarea {
        background-color: #1a1e26 !important;
        color: #ffffff !important;
        border: 1px solid #555 !important;
    }
 
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
 
    .stButton > button:hover {
        background-color: #ADD8E6;
    }
 
    /* Scrollbar in dropdown */
    div[data-baseweb="popover"]::-webkit-scrollbar {
        width: 8px;
    }
 
    div[data-baseweb="popover"]::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 4px;
    }
 
    /* Alerts */
    .stAlert {
        background-color: #20232a;
        color: white;
    }
</style>
    """, unsafe_allow_html=True)

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
    if file_name.lower().endswith(('.pdf', '.pptx')):  # Supported file types
        supported_files[file_name] = doc.page_content
 
# === Create vector store per file ===
def sanitize_name(file_name: str) -> str:
    name = os.path.splitext(file_name)[0]
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name).strip("_")
    return f"ragstore_{name}"
 
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
 
 
# === Fixed Prompts ===
file_chat_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
You are an assistant answering questions based on extracted file metadata.
Answer the questions based on the selected file name.
 
Context:
{context}
 
Question:
{question}
 
Answer as clearly and concisely as possible. If the answer isn't in the context, say so explicitly.
""",
)
 
# === Cross-file prompt also needs to use 'question' ===
cross_file_chat_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
You are a helpful assistant with access to metadata summaries of case study files.
You should base your answers only on the summaries, categories, domains, and technologies used.
 
Here is some metadata information:
{context}
 
Question: {question}
 
Answer based on the provided metadata information.
""",
)
 
# === Function to create QA chain with prompt ===
def get_qa_chain(vectorstore, file_key):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        k=3
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"filter": {"file_name": file_key}}),
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
VALIDATION_FILE = "validation_results.json"
MAX_RETRIES = 3
RETRY_DELAY = 2  
 
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
    # Check if the file has been processed before
    if file_name in metadata_cache:
        return metadata_cache[file_name]
 
    # Run the reader and categorization agents
    agent_output = run_reader_agent(text)
    category = run_categorization_agent(agent_output["summary"])["category"]
    domain = agent_output["domain"]
# agent_run_validation(metadata_path=METADATA_FILE, output_path=VALIDATION_FILE)
 
# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗂 File Categories", "📄 File Summary", "💬 Chat with File", "📊 Cross-File Chat", "🔍 Confidence Scores"]
)
 
# === Tab 1: File Categories Table ===
with tab1:
    st.subheader("🗂 File Categories")
    metadata_cache.update({file: process_file(file, text) for file, text in supported_files.items()})
    if metadata_cache:
        df = pd.DataFrame(metadata_cache.values())[["file_name", "category", "domain", "technology_used"]]
        df = df.rename(columns={
            "file_name": "File Name",
            "category": "Category",
            "domain": "Domain",
            "technology_used": "Technologies Used",  
        })
        st.dataframe(
            df, 
            use_container_width=True,
        )
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
            vs = load_or_create_metadata_vectorstore(metadata_cache)
            qa_chain = get_qa_chain(vs, selected_chat_file)
 
            st.session_state[f"qa_chain_{file_key}"] = qa_chain
            st.session_state[f"chat_history_{file_key}"] = []
 
        qa_chain = st.session_state[f"qa_chain_{file_key}"]
        chat_history = st.session_state[f"chat_history_{file_key}"]
 
        if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
            chat_history.clear()
            st.rerun()  # Updated deprecated method
 
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
        if "cross_qa_chain" not in st.session_state:
            vs = load_or_create_metadata_vectorstore(metadata_cache)
            st.session_state["cross_qa_chain"] = get_cross_file_chain(vs)
            st.session_state["cross_chat_history"] = []
 
        cross_qa = st.session_state["cross_qa_chain"]
        cross_chat_history = st.session_state["cross_chat_history"]
 
        if st.button("🔄 Reset Cross-File Chat", help="Reset cross-file chat history"):
            cross_chat_history.clear()
            st.rerun()
        for msg in cross_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
 
        cross_query = st.chat_input("Ask a question across all case studies:")
        if cross_query:
            with st.chat_message("user"):
                st.markdown(cross_query)
            with st.spinner("Thinking..."):
                result = cross_qa({"query": cross_query})
                answer = result.get("result", "❌ No answer found.")
            with st.chat_message("assistant"):
                st.markdown(answer)
            cross_chat_history.append({"role": "user", "content": cross_query})

 
 
agent_run_validation(metadata_path=METADATA_FILE, output_path=VALIDATION_FILE)
# === Tab 5: Confidence Scores ===
with tab5:
    st.subheader("🔍 Confidence Scores")
    # Read data from the validation results JSON file
    try:
        with open(VALIDATION_FILE, "r") as f:
            validation_results = json.load(f)
    except FileNotFoundError:
        st.error(f"Could not find the file: {VALIDATION_FILE}")
        validation_results = {}
    except json.JSONDecodeError:
        st.error("Error decoding JSON file.")
        validation_results = {}
    # Create a dictionary to hold the confidence data
    confidence_data = {
        "File Name": [],
        "Category Confidence": [],
        "Domain Confidence": [],
        "Technology Confidence": [],
        "Category Status": [],
        "Domain Status": [],
        "Technology Status": [],
    }
 
    # Process each entry in the JSON data
    for file_name, details in validation_results.items():
        # Extract confidence values from the JSON
        category_confidence = details["category"]["confidence"]
        domain_confidence = details["domain"]["confidence"]
        technology_confidence = details["technology_used"]["confidence"]
        # Determine the status of each confidence value
        category_status = "Valid" if category_confidence >= 0.9 else "Invalid"
        domain_status = "Valid" if domain_confidence >= 0.9 else "Invalid"
        technology_status = "Valid" if technology_confidence >= 0.9 else "Invalid"
        # Add the data to the dictionary
        confidence_data["File Name"].append(file_name)
        confidence_data["Category Confidence"].append(category_confidence)
        confidence_data["Domain Confidence"].append(domain_confidence)
        confidence_data["Technology Confidence"].append(technology_confidence)
        confidence_data["Category Status"].append(category_status)
        confidence_data["Domain Status"].append(domain_status)
        confidence_data["Technology Status"].append(technology_status)
 
    # Display the results in a table format
    st.write("**Confidence Evaluation Results:**")
    # Convert the dictionary into a pandas DataFrame for display
    confidence_df = pd.DataFrame(confidence_data)

 
    # Optionally, display a summary with counts of valid and invalid entries
    valid_count_c = sum(1 for status in confidence_data["Category Status"] if status == "Valid")
    invalid_count_c = len(confidence_data["Category Status"]) - valid_count_c
    valid_count_d = sum(1 for status in confidence_data["Domain Status"] if status == "Valid")
    invalid_count_d = len(confidence_data["Domain Status"]) - valid_count_d 
    valid_count_t = sum(1 for status in confidence_data["Technology Status"] if status == "Valid")
    invalid_count_t = len(confidence_data["Technology Status"]) - valid_count_t
    col_valid, col_invalid = st.columns(2)
 
    if dark_mode:
    # Dark mode colors
        valid_bg = "#1f3b2e"
        invalid_bg = "#3b1f1f"
        text_color = "white"
    else:
    # Light mode colors
        valid_bg = "#f0fff4"
        invalid_bg = "#fff0f0"
        text_color = "black"
 
    with col_valid:
     st.markdown(f"""
<div style='border: 1px solid #4CAF50; padding: 1em; border-radius: 10px;
                background-color: {valid_bg}; color: {text_color};'>
<h4>✅ Valid Entries</h4>
<p><b>Category:</b> {valid_count_c}</p>
<p><b>Domain:</b> {valid_count_d}</p>
<p><b>Technology:</b> {valid_count_t}</p>
</div>
    """, unsafe_allow_html=True)
 
    with col_invalid:
      st.markdown(f"""
<div style='border: 1px solid #f44336; padding: 1em; border-radius: 10px;
                background-color: {invalid_bg}; color: {text_color};'>
<h4>❌ Invalid Entries</h4>
<p><b>Category:</b> {invalid_count_c}</p>
<p><b>Domain:</b> {invalid_count_d}</p>
<p><b>Technology:</b> {invalid_count_t}</p>
</div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)  # Two lines space
 
    st.write("**confidence Dataframe:**")   
    st.dataframe(confidence_df)

# # main.py
# import os
# import re
# import json
# import streamlit as st
# import pandas as pd
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# import shutil
# from langchain_core.documents import Document

# from agents.reader_agent import run_reader_agent
# from agents.categorizer_agent import run_categorization_agent
# from agents.validation_agent import run_validation
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

# # === Create vector store per file ===
# def sanitize_name(file_name: str) -> str:
#     name = os.path.splitext(file_name)[0]
#     name = re.sub(r'[^a-zA-Z0-9._-]', '_', name).strip("_")
#     return f"ragstore_{name}"

# def load_or_create_metadata_vectorstore(metadata_cache):
#     safe_name = "metadata_vectorstore"
#     persist_dir = f".chromadb_{safe_name}"
    
#     # If vector store already exists
#     if os.path.exists(persist_dir):
#         vs = Chroma(
#             collection_name=safe_name,
#             embedding_function=embedding_model,
#             persist_directory=persist_dir,
#         )
#         # Check if the document count matches metadata cache
#         if len(vs.get()["documents"]) == len(metadata_cache):
#             return vs
#         else:
#             # Rebuild vectorstore if new/updated metadata is detected
#             shutil.rmtree(persist_dir)

#     # Create new Chroma vectorstore from metadata
#     documents = []
#     for metadata in metadata_cache.values():
#         text_for_embedding = (
#             f"File Name: {metadata.get('file_name', '')}\n"
#             f"Category: {metadata.get('category', '')}\n"
#             f"Domain: {metadata.get('domain', '')}\n"
#             f"Project Title: {metadata.get('project_title', '')}\n"
#             f"Technologies used: {metadata.get('technology_used', '')}\n"
#             f"Client Name: {metadata.get('client_name', '')}\n"
#             f"Summary: {metadata.get('summary', '')}"
#         )
#         doc = Document(page_content=text_for_embedding, metadata=metadata)
#         documents.append(doc)

#     vs = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         collection_name=safe_name,
#         persist_directory=persist_dir,
#     )
#     vs.persist()
#     return vs


# # === Fixed Prompts ===
# file_chat_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an assistant answering questions based on extracted file metadata.
# Answer the questions based on the selected file name.

# Context:
# {context}

# Question:
# {question}

# Answer as clearly and concisely as possible. If the answer isn't in the context, say so explicitly.
# """,
# )

# # === Cross-file prompt also needs to use 'question' ===
# cross_file_chat_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a helpful assistant with access to metadata summaries of case study files.
# You should base your answers only on the summaries, categories, domains, and technologies used.

# Here is some metadata information:
# {context}

# Question: {question}

# Answer based on the provided metadata information.
# """,
# )

# # === Function to create QA chain with prompt ===
# def get_qa_chain(vectorstore, file_key):
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         input_key="query",
#         k=3
#     )
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(search_kwargs={"filter": {"file_name": file_key}}),
#         memory=memory,
#         return_source_documents=False,
#         chain_type_kwargs={"prompt": file_chat_prompt}
#     )

# def get_cross_file_chain(vectorstore):
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
#         return_source_documents=False,
#         chain_type_kwargs={"prompt": cross_file_chat_prompt}
#     )


# # === Load & Save Metadata JSON ===
# METADATA_FILE = "metadata.json"
# VALIDATION_FILE = "validation_results.json"

# def load_metadata():
#     if os.path.exists(METADATA_FILE):
#         with open(METADATA_FILE, "r") as f:
#             return json.load(f)
#     return {}

# def save_metadata(metadata):
#     with open(METADATA_FILE, "w") as f:
#         json.dump(metadata, f, indent=2)

# # === Process File with Agents (only if not in cache) ===
# metadata_cache = load_metadata()

# @st.cache_data(show_spinner="🔍 Processing file...", ttl=3600)
# def process_file(file_name, text):
#     if file_name in metadata_cache:
#         return metadata_cache[file_name]

#     agent_output = run_reader_agent(text, file_name)
#     category = run_categorization_agent(agent_output["summary"])
#     result = {
#         "file_name": agent_output["file_name"],
#         "category": category,
#         "domain": agent_output["domain"],
#         "technology_used": agent_output["technology_used"],
#         "project_title": agent_output["project_title"],
#         "client_name": agent_output["client_name"],
#         "summary": agent_output["summary"],
#     }
#     metadata_cache[file_name] = result
#     save_metadata(metadata_cache)
#     # run_validation(metadata_path=METADATA_FILE, output_path=VALIDATION_FILE)
#     return result

# # === Tabs ===
# tab1, tab2, tab3, tab4 = st.tabs(["🗂 File Categories", "📄 File Summary", "💬 Chat with File", "📊 Cross-File Chat"])

# # === Tab 1: File Categories Table ===
# with tab1:
#     st.subheader("🗂 File Categories")
#     metadata_cache.update({file: process_file(file, text) for file, text in supported_files.items()})
#     if metadata_cache:
#         df = pd.DataFrame(metadata_cache.values())[["file_name", "category", "domain", "technology_used"]]
#         df = df.rename(columns={
#             "file_name": "File Name",
#             "category": "Category",
#             "domain": "Domain",
#             # "client_name": "Client Name",
#             "technology_used": "Technologies Used",  
#         })
        
#         st.dataframe(
#             df, 
#             use_container_width=True,
#         )
#     else:
#         st.info("Click the button above to process files.")

# # === Tab 2: File Summary Viewer ===
# with tab2:
#     st.subheader("📄 View File Summary")
#     summary_files = list(supported_files.keys())
#     selected_summary_file = st.selectbox("Choose a file to view summary:", summary_files, key="summary_file")

#     if selected_summary_file:
#         if selected_summary_file not in metadata_cache:
#             metadata_cache[selected_summary_file] = process_file(selected_summary_file, supported_files[selected_summary_file])

#         summary = metadata_cache[selected_summary_file]["summary"]
#         domain = metadata_cache[selected_summary_file]["domain"]
#         category = metadata_cache[selected_summary_file]["category"]

#         st.markdown(f"### Summary of `{selected_summary_file}`")
#         st.info(f"**Category:** {category} | **Domain:** {domain}")
#         st.markdown(summary)

# # === Tab 3: File Chatbot ===
# with tab3:
#     st.subheader("💬 Ask Questions About a File")
#     summary_files = list(supported_files.keys())
#     selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

#     if selected_chat_file:
#         file_key = selected_chat_file.replace(".", "_")

#         if f"qa_chain_{file_key}" not in st.session_state:
#             meta = metadata_cache.get(selected_chat_file)
#             if not meta:
#                 st.warning(f"No metadata found for {selected_chat_file}. Please process the file first.")
#                 st.stop()
                
#             vs = load_or_create_metadata_vectorstore(metadata_cache)
#             qa_chain = get_qa_chain(vs, selected_chat_file)

#             st.session_state[f"qa_chain_{file_key}"] = qa_chain
#             st.session_state[f"chat_history_{file_key}"] = []

#         qa_chain = st.session_state[f"qa_chain_{file_key}"]
#         chat_history = st.session_state[f"chat_history_{file_key}"]

#         if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
#             chat_history.clear()
#             st.rerun()  # Updated deprecated method

#         for msg in chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         user_input = st.chat_input("Type your question:")
#         if user_input:
#             with st.chat_message("user"):
#                 st.markdown(user_input)

#             response = qa_chain({"query": user_input})
#             answer = response.get("result", "❌ No answer found.")

#             with st.chat_message("assistant"):
#                 st.markdown(answer)

#             chat_history.append({"role": "user", "content": user_input})
#             chat_history.append({"role": "assistant", "content": answer})


# # === Tab 4: Cross-File Chat ===
# with tab4:
#     st.subheader("📊 Ask Questions Across All Files")
#     if not metadata_cache:
#         st.warning("Please run the categorization first.")
#     else:
#         if "cross_qa_chain" not in st.session_state:
#             vs = load_or_create_metadata_vectorstore(metadata_cache)
#             st.session_state["cross_qa_chain"] = get_cross_file_chain(vs)
#             st.session_state["cross_chat_history"] = []

#         cross_qa = st.session_state["cross_qa_chain"]
#         cross_chat_history = st.session_state["cross_chat_history"]

#         if st.button("🔄 Reset Cross-File Chat", help="Reset cross-file chat history"):
#             cross_chat_history.clear()
#             st.rerun()
        
#         for msg in cross_chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         cross_query = st.chat_input("Ask a question across all case studies:")
#         if cross_query:
#             with st.chat_message("user"):
#                 st.markdown(cross_query)
            
#             with st.spinner("Thinking..."):
#                 result = cross_qa({"query": cross_query})
#                 answer = result.get("result", "❌ No answer found.")
            
#             with st.chat_message("assistant"):
#                 st.markdown(answer)
            
#             cross_chat_history.append({"role": "user", "content": cross_query})
#             cross_chat_history.append({"role": "assistant", "content": answer})

# run_validation(metadata_path=METADATA_FILE, output_path=VALIDATION_FILE)
