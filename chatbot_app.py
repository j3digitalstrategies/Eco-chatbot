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
# We use a versioned folder name to ensure a fresh start on deployment
DB_DIR = "db_v26_fixed"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def prepare_database():
    try:
        # Check if already extracted
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Database Ready"

        if not os.path.exists(ZIP_NAME):
            return False, f"Error: {ZIP_NAME} not found in repository root."

        # Safety check for Git LFS pointer files
        if os.path.getsize(ZIP_NAME) < 10000:
            return False, "CORRUPT FILE: The zip file on GitHub is a 'link' (LFS). ACTION: Delete it on GitHub and re-upload by dragging the file directly into your browser."

        # Extraction Logic
        temp_extract = "temp_extract_dir"
        if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_extract)

        # Locate the actual database folder inside the zip
        sqlite_file = next(Path(temp_extract).rglob("chroma.sqlite3"), None)
        if not sqlite_file:
            return False, "Internal Error: chroma.sqlite3 not found inside the zip archive."

        if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
        shutil.copytree(sqlite_file.parent, CHROMA_PATH)
        shutil.rmtree(temp_extract)
        
        return True, "Database Successfully Extracted"
    except Exception as e:
        return False, f"Setup Error: {str(e)}"

# --- 3. UI HEADER ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

db_ok, db_status = prepare_database()

if not db_ok:
    st.error(db_status)
    st.info("After fixing the file, remember to 'Reboot App' in the Streamlit menu.")
    st.stop()

# --- 4. AI ENGINE (Fixed for KeyError: '_type') ---
@st.cache_resource
def get_retrieval_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        
        # FIX: We initialize the PersistentClient DIRECTLY. 
        # This bypasses the schema validation that causes the KeyError.
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain", # Default name from your ingest code
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Answer ONLY based on the context. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"AI Engine Failure: {e}")
        return None

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested Question Buttons (Horizontal Layout)
st.write("### Suggested Topics")
c1, c2, c3 = st.columns(3)
if c1.button("What is the focus of the Waste module?", use_container_width=True):
    st.session_state.pending_query = "What is the focus of the Waste module?"
if c2.button("Tell me about the Recycling approach.", use_container_width=True):
    st.session_state.pending_query = "Tell me about the Recycling approach."
if c3.button("Give me some eco-friendly tips.", use_container_width=True):
    st.session_state.pending_query = "Give me some eco-friendly tips."

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle Inputs
chat_input = st.chat_input("Ask a question about the curriculum...")
query = chat_input if chat_input else st.session_state.get("pending_query")

if query:
    if "pending_query" in st.session_state: del st.session_state.pending_query
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    api_key = st.secrets.get("OPENAI_API_KEY")
    chain = get_retrieval_chain(api_key)
    
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Reviewing curriculum documents..."):
                response = chain.invoke({"input": query, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()
