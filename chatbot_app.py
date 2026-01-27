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
DB_DIR = "eco_final_stable_v2"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- 2. SURGICAL SCHEMA FIX (KeyError: '_type') ---
def repair_chroma_metadata(db_path):
    sqlite_db = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_db):
        return False, "sqlite3 file not found for patching"
    
    try:
        conn = sqlite3.connect(sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT id, configuration_json FROM collections")
        rows = cursor.fetchall()
        
        for row_id, config_json in rows:
            if config_json:
                config_data = json.loads(config_json)
                if "_type" not in config_data:
                    config_data["_type"] = "CollectionConfigurationInternal"
                    updated_json = json.dumps(config_data)
                    cursor.execute(
                        "UPDATE collections SET configuration_json = ? WHERE id = ?",
                        (updated_json, row_id)
                    )
        
        conn.commit()
        conn.close()
        return True, "Metadata patch applied successfully"
    except Exception as e:
        return False, f"Patching failed: {str(e)}"

# --- 3. INITIALIZATION & DEBUGGING ---
@st.cache_resource
def startup_sequence():
    status_log = []
    
    if not os.path.exists(ZIP_NAME):
        return False, [f"❌ CRITICAL: {ZIP_NAME} not found in repository root."]

    # Check file integrity before unzipping
    try:
        file_size = os.path.getsize(ZIP_NAME)
        status_log.append(f"📁 Detected file size: {file_size / (1024*1024):.2f} MB")
        
        with open(ZIP_NAME, 'rb') as f:
            header = f.read(4)
            # Zip magic number is 'PK\x03\x04'
            if header != b'\x50\x4b\x03\x04':
                f.seek(0)
                file_start = f.read(100)
                status_log.append(f"❌ INVALID ZIP HEADER: {header}")
                status_log.append(f"📄 File starts with: {file_start}")
                return False, status_log
            
        status_log.append("✅ Zip header verified (PK\\x03\\x04)")

        # Extraction
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        temp_extract = "temp_run"
        if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_extract)
        status_log.append("✅ Files unzipped successfully")

        # Find Database
        sqlite_path = next(Path(temp_extract).rglob("chroma.sqlite3"), None)
        if not sqlite_path:
            return False, status_log + ["❌ Could not find chroma.sqlite3 in the zip."]
        
        shutil.copytree(sqlite_path.parent, DB_DIR)
        shutil.rmtree(temp_extract)
        status_log.append("✅ Database moved to persistent path")

        # Patch
        ok, patch_msg = repair_chroma_metadata(DB_DIR)
        status_log.append(f"🛠 {patch_msg}")
        
        return True, status_log
    except Exception as e:
        return False, status_log + [f"❌ Startup Exception: {str(e)}"]

# --- 4. UI STARTUP ---
st.title("🌱 Eco-Education Assistant")

with st.expander("🔍 System Diagnostic Logs", expanded=True):
    success, logs = startup_sequence()
    for log in logs:
        st.write(log)

if not success:
    st.error("Application setup failed. Please check the logs above for the 'File starts with' output.")
    st.stop()

# --- 5. AI ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Use provided context to answer. Context: {context}"),
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

# --- 6. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("### Quick Questions")
c1, c2, c3 = st.columns(3)
if c1.button("Waste Module Focus", use_container_width=True): st.session_state.q = "What is the focus of the Waste module?"
if c2.button("Recycling Approach", use_container_width=True): st.session_state.q = "Tell me about the Recycling approach."
if c3.button("Eco-tips", use_container_width=True): st.session_state.q = "Give me some eco-friendly tips."

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_input = st.chat_input("Ask about the curriculum...")
final_query = user_input if user_input else st.session_state.get("q")

if final_query:
    if "q" in st.session_state: del st.session_state.q
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): st.markdown(final_query)

    with st.chat_message("assistant"):
        chain = get_bot_chain(st.secrets["OPENAI_API_KEY"])
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
        with st.spinner("Analyzing curriculum..."):
            response = chain.invoke({"input": final_query, "chat_history": history})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()
