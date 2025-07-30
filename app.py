import streamlit as st
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# 1. Load and split document
loader = PyPDFLoader("rabbits.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
splits = splitter.split_documents(pages)

# 2. Embeddings
embedding = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version=os.getenv("API_VERSION"),
    azure_endpoint="https://gen-cim-eas-dep-genai-train-openai.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# 3. Vector store
persist_dir = "docs/chroma"
vectordb = Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_dir)

# 4. Custom prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" 
at the end of the answer. 
{chat_history}
Context: {context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_END_POINT"],
    azure_deployment=os.environ["DEPLOYMENT_NAME"],
    openai_api_version=os.environ["API_VERSION"],
)

# 6. Memory (with Streamlit session state)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="question", 
        output_key="answer", 
        return_messages=True
    )

memory = st.session_state.memory

# 7. Conversational Retrieval Chain (with prompt and sources)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": QA_CHAIN_PROMPT, 
        "output_key": "answer"
    },
    return_source_documents=True
)

# 8. Streamlit UI
st.set_page_config(page_title="Rabbit Q&A Bot", layout="centered")
st.title("🐰 Ask Me Anything About Rabbits!")

# Reset button
if st.button("🔄 Reset Chat", help="Reset Chat"):
    st.session_state.memory.clear()
    st.session_state.chat_history = []
    st.rerun()

# Chat input
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question:")

if user_input:
    result = qa_chain({"question": user_input})

    # Show sources if available
    if "source_documents" in result:
        with st.expander("Sources used"):
            for doc in result["source_documents"]:
                st.write(doc.page_content[:300] + "...")

    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
