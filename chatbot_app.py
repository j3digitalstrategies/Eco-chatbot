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
DB_DIR = "eco_db_v8_final"
ZIP_NAME = "chroma_db.zip"
st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- 2. STARTUP & EXTRACTION ---
@st.cache_resource
def startup_sequence():
    if not os.path.exists(ZIP_NAME):
        return False, "Database zip missing from GitHub."
    
    try:
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall("temp_v8")
        
        target = next(Path("temp_v8").rglob("chroma.sqlite3"), None)
        if not target: return False, "Invalid database structure inside zip."
        
        shutil.copytree(target.parent, DB_DIR)
        shutil.rmtree("temp_v8")
        return True, "System Ready"
    except Exception as e:
        return False, f"Startup Error: {e}"

# --- 3. UI LAYOUT ---
st.title("🌱 Eco-Education Assistant")

success, status = startup_sequence()
if not success:
    st.error(status)
    st.stop()

# --- 4. THE ENGINE (Fixed for '_type' error) ---
@st.cache_resource
def get_bot_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        
        # Initialize the persistent client directly to handle legacy migrations
        persistent_client = chromadb.PersistentClient(path=os.path.abspath(DB_DIR))
        
        # Use the client directly in the LangChain wrapper
        vectorstore = Chroma(
            client=persistent_client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Answer using the provided context. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"Engine failure: {e}")
        return None

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_input = st.chat_input("Ask about the curriculum...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        chain = get_bot_chain(st.secrets["OPENAI_API_KEY"])
        if chain:
            # Reconstruct history for the chain
            history = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                else:
                    history.append(AIMessage(content=m["content"]))
            
            with st.spinner("Searching curriculum..."):
                response = chain.invoke({"input": user_input, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()
