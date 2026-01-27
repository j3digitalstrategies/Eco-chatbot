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

# --- 1. CONFIGURATION ---
load_dotenv()
DB_DIR = "eco_legacy_fix_v1"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- 2. MULTI-VERSION SCHEMA REPAIR ---
def repair_chroma_metadata(db_path):
    sqlite_db = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_db):
        return False, "sqlite3 file not found"
    
    try:
        conn = sqlite3.connect(sqlite_db)
        cursor = conn.cursor()
        
        # Check if we are on a very old schema (pre-0.4.0)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collections';")
        has_collections_table = cursor.fetchone()
        
        if has_collections_table:
            # Modern-ish Schema Fix
            cursor.execute("PRAGMA table_info(collections)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if "configuration_json" in columns:
                cursor.execute("SELECT id, configuration_json FROM collections")
                for row_id, config_json in cursor.fetchall():
                    if config_json:
                        config_data = json.loads(config_json)
                        if "_type" not in config_data:
                            config_data["_type"] = "CollectionConfigurationInternal"
                            cursor.execute("UPDATE collections SET configuration_json = ? WHERE id = ?", (json.dumps(config_data), row_id))
                conn.commit()
                msg = "Modern schema patched."
            else:
                msg = "Collections table exists but no configuration column. Skipping."
        else:
            # Legacy Schema Logic: Older versions often just need to be opened by the client
            # to trigger internal migration, but we'll flag it here.
            msg = "Legacy schema detected. Attempting standard connection."
            
        conn.close()
        return True, msg
    except Exception as e:
        return False, f"Repair failed: {str(e)}"

# --- 3. STARTUP LOGIC ---
@st.cache_resource
def startup_sequence():
    logs = []
    if not os.path.exists(ZIP_NAME):
        return False, ["❌ Missing chroma_db.zip"]

    try:
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        temp_dir = "temp_extract"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_dir)
        
        sqlite_path = next(Path(temp_dir).rglob("chroma.sqlite3"), None)
        if not sqlite_path: return False, ["❌ No database in zip"]
        
        shutil.copytree(sqlite_path.parent, DB_DIR)
        shutil.rmtree(temp_dir)
        
        ok, repair_msg = repair_chroma_metadata(DB_DIR)
        logs.append(f"🛠 {repair_msg}")
        return True, logs
    except Exception as e:
        return False, [f"❌ Startup Error: {e}"]

# --- 4. UI ---
st.title("🌱 Eco-Education Assistant")

with st.expander("🔍 System Status", expanded=True):
    success, setup_logs = startup_sequence()
    for log in setup_logs: st.write(log)

if not success:
    st.error("Setup failed.")
    st.stop()

# --- 5. ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        # We use standard Settings to allow Chroma to attempt its own migration on older files
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Answer using the context. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"Engine Error: {e}")
        return None

# --- 6. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Buttons
c1, c2, c3 = st.columns(3)
if c1.button("Waste Module Focus"): st.session_state.query = "What is the focus of the Waste module?"
if c2.button("Recycling Approach"): st.session_state.query = "Tell me about the Recycling approach."
if c3.button("Eco-tips"): st.session_state.query = "Give me some eco-friendly tips."

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_input = st.chat_input("Ask a question...")
query = user_input if user_input else st.session_state.get("query")

if query:
    if "query" in st.session_state: del st.session_state.query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        api_key = st.secrets["OPENAI_API_KEY"]
        chain = get_bot_chain(api_key)
        if chain:
            history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
            with st.spinner("Searching..."):
                res = chain.invoke({"input": query, "chat_history": history})
                st.markdown(res["answer"])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    st.rerun()
