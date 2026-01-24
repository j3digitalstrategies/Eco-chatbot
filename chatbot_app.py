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
# Using a specific folder name to ensure a fresh extraction attempt
DB_DIR = "db_production_final"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def prepare_database():
    try:
        # If database already exists and is valid, skip extraction
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Database Online"

        if not os.path.exists(ZIP_NAME):
            return False, f"Error: {ZIP_NAME} missing from repository."

        # Safety check for Git LFS pointers
        if os.path.getsize(ZIP_NAME) < 5000:
            return False, "Database file is a 'pointer' (Git LFS). Please re-upload the zip by dragging it directly into GitHub."

        # Extraction Process
        temp_dir = "temp_unzip"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_dir)

        # Locate the internal db folder
        sqlite_loc = next(Path(temp_dir).rglob("chroma.sqlite3"), None)
        if not sqlite_loc:
            return False, "Internal Error: chroma.sqlite3 not found in zip."

        if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
        shutil.copytree(sqlite_loc.parent, CHROMA_PATH)
        shutil.rmtree(temp_dir)
        
        return True, "Success"
    except Exception as e:
        return False, f"Fatal Setup Error: {e}"

# --- 3. UI LAYOUT ---
st.title("🌱 Eco-Education Curriculum Assistant")
st.caption("AI-powered insights into Ann Lewin-Benham's curriculum.")

ready, status = prepare_database()

if not ready:
    st.error(status)
    st.info("Try clicking 'Manage app' -> 'Reboot app' in the Streamlit menu.")
    st.stop()

# --- 4. ENGINE (The fix for KeyError: '_type') ---
@st.cache_resource
def get_retrieval_chain(_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_key)
        
        # We use the PersistentClient directly. This bypasses the logic in 
        # langchain-chroma that triggers the KeyError: '_type'.
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Use the following context to answer. Context: {context}"),
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

# --- 5. INTERACTION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested Questions
st.write("### Suggested Topics")
c1, c2, c3 = st.columns(3)
btns = [
    "What is the focus of the Waste module?",
    "Tell me about the Recycling approach.",
    "Give me some eco-friendly tips."
]

if c1.button(btns[0], use_container_width=True): st.session_state.prompt = btns[0]
if c2.button(btns[1], use_container_width=True): st.session_state.prompt = btns[1]
if c3.button(btns[2], use_container_width=True): st.session_state.prompt = btns[2]

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Process Input
user_query = st.chat_input("Ask about the curriculum...")
final_query = user_query if user_query else st.session_state.get("prompt")

if final_query:
    if "prompt" in st.session_state: del st.session_state.prompt
    
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): st.markdown(final_query)

    api_key = st.secrets.get("OPENAI_API_KEY")
    chain = get_retrieval_chain(api_key)
    
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Searching curriculum..."):
                res = chain.invoke({"input": final_query, "chat_history": history})
                st.markdown(res["answer"])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    st.rerun()
