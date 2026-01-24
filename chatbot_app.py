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

# --- 1. CONFIGURATION ---
load_dotenv()
DB_DIR = "db_production_final"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def prepare_database():
    try:
        # Check if DB is already extracted and valid
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Online"

        if not os.path.exists(ZIP_NAME):
            return False, f"Missing {ZIP_NAME} in root directory."

        # Extracting Zip
        temp_dir = "temp_unzip"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_dir)

        # Locate the database files (handling nested folders)
        sqlite_loc = next(Path(temp_dir).rglob("chroma.sqlite3"), None)
        if not sqlite_loc:
            return False, "Chroma files not found inside zip."

        if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
        shutil.copytree(sqlite_loc.parent, CHROMA_PATH)
        shutil.rmtree(temp_dir)
        
        return True, "Database Ready"
    except Exception as e:
        return False, f"Init Error: {e}"

# --- 3. UI HEADER ---
st.title("🌱 Eco-Education Assistant")
st.markdown("---")

db_ready, db_msg = prepare_database()
if not db_ready:
    st.error(db_msg)
    st.stop()

# --- 4. AI ENGINE ---
@st.cache_resource
def get_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        
        # PERSISTENT CLIENT FIX: Bypasses 'KeyError: _type'
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant for the Eco-Education curriculum. Use the following context to answer the user. Context: {context}"),
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

# --- 5. CHAT SYSTEM ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggestions
st.write("### Quick Questions")
c1, c2, c3 = st.columns(3)
opts = ["Focus of the Waste module?", "Recycling approach?", "Eco-friendly tips?"]
if c1.button(opts[0], use_container_width=True): st.session_state.auto_query = opts[0]
if c2.button(opts[1], use_container_width=True): st.session_state.auto_query = opts[1]
if c3.button(opts[2], use_container_width=True): st.session_state.auto_query = opts[2]

# Render History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input Logic
user_in = st.chat_input("Ask about the curriculum...")
query = user_in if user_in else st.session_state.get("auto_query")

if query:
    if "auto_query" in st.session_state: del st.session_state.auto_query
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    api_key = st.secrets.get("OPENAI_API_KEY")
    chain = get_chain(api_key)
    
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Analyzing curriculum..."):
                res = chain.invoke({"input": query, "chat_history": history})
                st.markdown(res["answer"])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    st.rerun()
