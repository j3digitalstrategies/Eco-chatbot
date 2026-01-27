__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import chromadb
import shutil
import sqlite3
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIG ---
load_dotenv()
DB_DIR = "eco_db_v7"
ZIP_NAME = "chroma_db.zip"
st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- SCHEMA REPAIR ---
def patch_database(db_path):
    sqlite_db = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_db): return False, "No sqlite3 file."
    try:
        conn = sqlite3.connect(sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collections';")
        if cursor.fetchone():
            cursor.execute("SELECT id, configuration_json FROM collections")
            for row_id, config_json in cursor.fetchall():
                if config_json and "_type" not in json.loads(config_json):
                    data = json.loads(config_json)
                    data["_type"] = "CollectionConfigurationInternal"
                    cursor.execute("UPDATE collections SET configuration_json = ? WHERE id = ?", (json.dumps(data), row_id))
            conn.commit()
        conn.close()
        return True, "Database patched."
    except Exception as e:
        return False, str(e)

# --- STARTUP ---
@st.cache_resource
def startup():
    if not os.path.exists(ZIP_NAME):
        return False, f"Missing {ZIP_NAME} in GitHub."
    
    # CHECK FOR GIT LFS POINTER
    with open(ZIP_NAME, 'rb') as f:
        chunk = f.read(100)
        if b"version https://git-lfs" in chunk:
            return False, "⚠️ GITHUB ERROR: Your 'chroma_db.zip' is an LFS pointer. Please delete it from GitHub and re-upload it manually via the 'Upload files' button in your browser to fix this."

    try:
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall("temp_v7")
        
        target = next(Path("temp_v7").rglob("chroma.sqlite3"), None)
        if not target: return False, "Zip is empty or invalid."
        
        shutil.copytree(target.parent, DB_DIR)
        shutil.rmtree("temp_v7")
        patch_database(DB_DIR)
        return True, "Ready!"
    except Exception as e:
        return False, f"Extraction failed: {e}"

# --- UI ---
st.title("🌱 Eco-Education Assistant")

success, status_msg = startup()
if not success:
    st.error(status_msg)
    st.info("💡 **How to fix this:** Go to your GitHub repo, delete 'chroma_db.zip', click 'Add file' -> 'Upload files', and upload the zip directly. Do not use 'git push' for the zip file.")
    st.stop()

# --- ENGINE ---
@st.cache_resource
def get_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings, collection_name="langchain")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the context to answer. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return create_retrieval_chain(vectorstore.as_retriever(), create_stuff_documents_chain(llm, prompt))
    except Exception as e:
        st.error(f"Engine failure: {e}")
        return None

# --- CHAT ---
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

query = st.chat_input("Ask a question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    chain = get_chain(st.secrets["OPENAI_API_KEY"])
    if chain:
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
        with st.spinner("Thinking..."):
            res = chain.invoke({"input": query, "chat_history": history})
            st.chat_message("assistant").markdown(res["answer"])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
