__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import chromadb
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETTINGS ---
load_dotenv()
# Using a specific folder for production to avoid locking/permission issues
DB_DIR = "db_production_final"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. AUTOMATIC DATABASE SETUP ---
@st.cache_resource
def setup_database():
    try:
        # Check if DB is already extracted and valid
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Database Ready"

        if not os.path.exists(ZIP_NAME):
            return False, f"Error: '{ZIP_NAME}' not found in the root directory."

        # Extraction logic
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            # Extract to a temporary folder first
            temp_extract = "temp_db_extract"
            if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
            zip_ref.extractall(temp_extract)
            
            # Find the actual directory containing chroma.sqlite3
            sqlite_path = None
            for root, dirs, files in os.walk(temp_extract):
                if "chroma.sqlite3" in files:
                    sqlite_path = root
                    break
            
            if not sqlite_path:
                return False, "Could not find 'chroma.sqlite3' inside the zip."

            # Move files to final production path
            if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
            shutil.copytree(sqlite_path, CHROMA_PATH)
            shutil.rmtree(temp_extract)
            
        return True, "Database initialized successfully."
    except Exception as e:
        return False, f"Setup failed: {str(e)}"

# Run setup once
db_ready, db_status = setup_database()

# --- 3. UI LAYOUT ---
st.title("🌱 Eco-Education Curriculum Assistant")
st.caption("AI-powered insights into Ann Lewin-Benham's curriculum.")

if not db_ready:
    st.error(db_status)
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Please set the OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

# --- 4. AI ENGINE ---
@st.cache_resource
def get_retrieval_chain(_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
    
    # Connect to the extracted database
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="langchain"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
    
    system_prompt = (
        "You are an expert assistant for the Eco-Education curriculum. "
        "Use the provided context to answer questions accurately. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 5}), combine_docs_chain)

# Initialize Chat Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. INTERACTIVE CHAT ---
# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick Question Buttons
st.sidebar.title("Suggested Topics")
quick_queries = [
    "What is the focus of the Waste module?",
    "Explain the approach to Recycling.",
    "What activities are suggested for Biodiversity?"
]

for q in quick_queries:
    if st.sidebar.button(q):
        st.session_state.pending_query = q

# Handle Input
user_input = st.chat_input("Ask a question...")
if st.session_state.get("pending_query"):
    user_input = st.session_state.pop("pending_query")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching curriculum..."):
            chain = get_retrieval_chain(api_key)
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            
            result = chain.invoke({"input": user_input, "chat_history": history})
            response = result["answer"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
