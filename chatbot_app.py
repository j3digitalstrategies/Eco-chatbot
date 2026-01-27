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
DB_DIR = "eco_db_final_v6"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- 2. THE SCHEMA FIXER (The "Secret Sauce") ---
def force_migrate_schema(db_path):
    sqlite_db = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_db):
        return False, "Database file not found."
    
    try:
        conn = sqlite3.connect(sqlite_db)
        cursor = conn.cursor()
        
        # Check if 'collections' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collections';")
        if cursor.fetchone():
            # Check for the configuration column
            cursor.execute("PRAGMA table_info(collections)")
            cols = [c[1] for c in cursor.fetchall()]
            
            if "configuration_json" in cols:
                cursor.execute("SELECT id, configuration_json FROM collections")
                rows = cursor.fetchall()
                for row_id, config_json in rows:
                    if config_json:
                        data = json.loads(config_json)
                        if "_type" not in data:
                            data["_type"] = "CollectionConfigurationInternal"
                            cursor.execute("UPDATE collections SET configuration_json = ? WHERE id = ?", (json.dumps(data), row_id))
                conn.commit()
                msg = "Schema successfully patched."
            else:
                msg = "Legacy schema detected (v0.3.x); attempting standard load."
        else:
            msg = "Pre-collections schema; attempting load."
            
        conn.close()
        return True, msg
    except Exception as e:
        return False, f"Patch failed: {str(e)}"

# --- 3. STARTUP & EXTRACTION ---
@st.cache_resource
def startup_sequence():
    logs = []
    if not os.path.exists(ZIP_NAME):
        return False, [f"❌ Error: {ZIP_NAME} not found in repository root."]

    try:
        # 1. Clean old attempts
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        temp_dir = "temp_v6"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        # 2. Extract
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_dir)
        
        # 3. Find and Move
        target = next(Path(temp_dir).rglob("chroma.sqlite3"), None)
        if not target: return False, ["❌ Error: No chroma.sqlite3 found inside zip."]
        
        shutil.copytree(target.parent, DB_DIR)
        shutil.rmtree(temp_dir)
        
        # 4. Patch
        ok, patch_msg = force_migrate_schema(DB_DIR)
        logs.append(f"✅ {patch_msg}")
        return True, logs
    except Exception as e:
        return False, [f"❌ Startup Error: {e}"]

# --- 4. UI ---
st.title("🌱 Eco-Education Assistant")

with st.expander("🔍 System Diagnostic", expanded=True):
    success, setup_logs = startup_sequence()
    for log in setup_logs: st.write(log)

if not success:
    st.error("System could not start. Check the logs above.")
    st.stop()

# --- 5. AI ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        
        # Use simple settings to allow internal migration
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Answer questions based on the context. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"Engine initialization failed: {e}")
        return None

# --- 6. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Access Buttons
cols = st.columns(3)
if cols[0].button("Waste Module Focus"): st.session_state.query = "What is the focus of the Waste module?"
if cols[1].button("Recycling Approach"): st.session_state.query = "Tell me about the Recycling approach."
if cols[2].button("Eco-tips"): st.session_state.query = "Give me some eco-friendly tips."

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_input = st.chat_input("Ask about the curriculum...")
query = user_input if user_input else st.session_state.get("query")

if query:
    if "query" in st.session_state: del st.session_state.query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        chain = get_bot_chain(st.secrets["OPENAI_API_KEY"])
        if chain:
            history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
            with st.spinner("Analyzing curriculum..."):
                response = chain.invoke({"input": query, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        else:
            st.error("The AI engine failed to load. Please check System Diagnostics.")
    st.rerun()
