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

# --- 1. CONFIG ---
load_dotenv()
# Using a fresh versioned path to force Streamlit to ignore old cached folders
DB_VERSION = "v7_prod"
CHROMA_PATH = os.path.join(os.getcwd(), f"chroma_db_{DB_VERSION}")
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. RESILIENT DATABASE EXTRACTION ---
@st.cache_resource
def prepare_environment():
    try:
        # If folder exists and looks healthy, skip extraction
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Database Ready"

        zip_path = Path(ZIP_NAME)
        if not zip_path.exists():
            return False, f"CRITICAL ERROR: {ZIP_NAME} is missing from the GitHub repository."

        # Check file size to detect Git LFS pointers
        file_size = zip_path.stat().st_size
        if file_size < 5000: # Usually pointers are < 1KB
            return False, f"ERROR: The zip file is only {file_size} bytes. GitHub is serving a 'Link' (LFS) instead of the actual data. Please re-upload the zip by dragging it into your browser on GitHub.com."

        if not zipfile.is_zipfile(zip_path):
            return False, f"ERROR: {ZIP_NAME} is corrupted or not a valid zip file."

        # Extraction Process
        temp_dir = Path(f"temp_extract_{DB_VERSION}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        # Deep search for the sqlite file
        db_file = next(temp_dir.rglob("chroma.sqlite3"), None)
        if not db_file:
            return False, "ERROR: chroma.sqlite3 not found inside the zip file."

        # Move the directory containing the sqlite3 file to the final path
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        shutil.copytree(db_file.parent, CHROMA_PATH)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        return True, "Database Successfully Extracted"

    except Exception as e:
        return False, f"Extraction Exception: {str(e)}"

# Run Initialization
success, status_msg = prepare_environment()

# --- 3. UI LAYOUT ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

if not success:
    st.error(status_msg)
    st.stop()

# Validate OpenAI Key
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 Please add your OPENAI_API_KEY to the Streamlit Cloud Secrets tab.")
    st.stop()

# --- 4. ENGINE INITIALIZATION ---
@st.cache_resource
def get_rag_chain(_openai_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_openai_key)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(
            client=client, 
            collection_name="langchain", 
            embedding_function=embeddings
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_openai_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Use the provided context to answer the user's questions. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"AI Engine Error: {e}")
        return None

# Session State for History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. INTERACTIVE PROMPTS ---
st.write("### Suggested Questions")
c1, c2, c3 = st.columns(3)
if c1.button("What is the waste module?"):
    st.session_state.btn_input = "What is the waste module?"
if c2.button("Tell me about recycling"):
    st.session_state.btn_input = "Tell me about recycling"
if c3.button("Give me some eco-friendly tips"):
    st.session_state.btn_input = "Give me some eco-friendly tips"

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. CHAT LOGIC ---
user_query = st.chat_input("Ask a question about the curriculum...")
final_query = user_query if user_query else st.session_state.get("btn_input")

if final_query:
    if "btn_input" in st.session_state:
        del st.session_state["btn_input"]
        
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)

    chain = get_rag_chain(api_key)
    if chain:
        with st.chat_message("assistant"):
            chat_history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Analyzing curriculum..."):
                try:
                    response = chain.invoke({"input": final_query, "chat_history": chat_history})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    st.error(f"Execution Error: {e}")
    st.rerun()
