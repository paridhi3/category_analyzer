# # main.py

# import os
# from tqdm import tqdm
# from langchain_community.document_loaders import AzureBlobStorageContainerLoader
# from agents.reader_agent import run_reader_agent
# from agents.categorizer_agent import run_categorization_agent

# # Load documents from Azure Blob Storage
# loader = AzureBlobStorageContainerLoader(
#     conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
#     container=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
# )
# documents = loader.load()

# def process_documents(documents):
#     results = []
#     categories = []

#     for doc in tqdm(documents):
#         try:
#             source_path = doc.metadata.get("source")
#             file_name = os.path.basename(source_path)
#             if file_name.lower().endswith(('.pdf', '.pptx')):
#                 text = doc.page_content

#                 # Agent 1: Read and Summarize
#                 summary = run_reader_agent(text)
#                 print(f"\nSummary of {file_name}:\n{summary}")

#                 # Agent 2: Categorize
#                 category = run_categorization_agent(summary, categories)
#                 if category not in categories:
#                     categories.append(category)

#                 results.append({"filename": file_name, "category": category})
#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")

#     return results, categories

# if __name__ == "__main__":
#     results, all_categories = process_documents(documents)

#     print("\nCategorization Results:")
#     for item in results:
#         print(f"{item['filename']}: {item['category']}")

#     print("\nFinal Categories Identified:")
#     for c in all_categories:
#         print("-", c)

# main.py
# import os
# import streamlit as st
# from langchain_community.document_loaders import AzureBlobStorageContainerLoader
# from agents.reader_agent import run_reader_agent
# from agents.categorizer_agent import run_categorization_agent

# # === Streamlit Setup ===
# st.set_page_config(page_title="Case Study Categorizer", layout="wide")
# st.title("📁 Case Study Categorizer")

# # === Load Documents from Azure ===
# @st.cache_resource
# def load_documents():
#     loader = AzureBlobStorageContainerLoader(
#         conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
#         container=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
#     )
#     return loader.load()

# documents = load_documents()

# # === Prepare a list of supported file names ===
# supported_files = {}
# for doc in documents:
#     source_path = doc.metadata.get("source", "Unknown source")
#     file_name = os.path.basename(source_path)
#     if file_name.lower().endswith(('.pdf', '.pptx')):
#         supported_files[file_name] = doc.page_content

# # === Sidebar multiple file picker ===
# selected_files = st.sidebar.multiselect("📂 Select file(s)", options=list(supported_files.keys()))

# if selected_files:
#     for file in selected_files:
#         st.subheader(f"📝 Summary of `{file}`")
#         text = supported_files[file]

#         # 💡 Agent 1: Reader
#         summary = run_reader_agent(text)
#         st.markdown(summary)

#         # 📌 Agent 2: Categorizer
#         category = run_categorization_agent(summary, [])
#         st.markdown(f"**📁 Category:** {category}")

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

# === Prepare a dictionary of supported file names and contents ===
supported_files = {}
for doc in documents:
    source_path = doc.metadata.get("source", "Unknown source")
    file_name = os.path.basename(source_path)
    if file_name.lower().endswith(('.pdf', '.pptx')):
        supported_files[file_name] = doc.page_content

# === Sidebar: File Selection ===
selected_files = st.sidebar.multiselect("📂 Select file(s)", options=list(supported_files.keys()))

# === Embed Config ===
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_END_POINT")
api_version = os.getenv("API_VERSION")
embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# === Create Vector Store ===
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment=embed_deployment,
        openai_api_version=api_version,
        azure_endpoint=api_base,
        api_key=api_key
    )

    vector_store = Chroma.from_texts(
        chunks,
        embedding=embedding_model,
        collection_name="rag_store",
        persist_directory=".chromadb"
    )
    vector_store.persist()
    return vector_store

# === Setup QA Chain ===
def get_qa_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        k=3
    )

    qa_chain = RetrievalQA.from_chain_type(
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
    return qa_chain

# === Main Section: Reader + Categorizer ===
all_text = ""
if selected_files:
    for file in selected_files:
        st.subheader(f"📝 Summary of `{file}`")
        text = supported_files[file]
        all_text += "\n" + text  # For RAG vector store

        # 💡 Agent 1: Reader
        summary = run_reader_agent(text)
        st.markdown(summary)

        # 📌 Agent 2: Categorizer
        category = run_categorization_agent(summary, [])
        st.markdown(f"**📁 Category:** {category}")

# === Initialize Vector Store and QA Chain ===
if selected_files:
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = create_vector_store(all_text)
        st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)
        st.session_state.chat_history = []

    # 💬 Sidebar Chatbot
    st.sidebar.title("💬 Ask the Chatbot")
    user_question = st.sidebar.text_input("Ask a question:")

    if user_question:
        response = st.session_state.qa_chain({"query": user_question})
        answer = response.get('result', "❌ No answer found.")
        st.session_state.chat_history.append({"user": user_question, "bot": answer})
        st.markdown(f"**You asked:** {user_question}")
        st.markdown(f"**Answer:** {answer}")

    # 🧾 Chat History
    if st.session_state.chat_history:
        with st.expander("📜 Chat History"):
            for i, chat in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"**You {i}:** {chat['user']}")
                st.markdown(f"**Bot {i}:** {chat['bot']}")


