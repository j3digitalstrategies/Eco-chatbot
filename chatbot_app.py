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
# Unique folder to force a fresh, clean start
DB_FOLDER = "db_production_v11"
CHROMA_PATH = os.path.join(os.getcwd(), DB_FOLDER)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE RECOVERY ENGINE ---
@st.cache_resource
def initialize_system():
    try:
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "System Ready"

        if not os.path.exists(ZIP_NAME):
            return False, f"File {ZIP_NAME} not found."

        # Extract
        temp_dir = "temp_extract"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_dir)

        # Find the db file
        sqlite_file = next(Path(temp_dir).rglob("chroma.sqlite3"), None)
        if not sqlite_file:
            return False, "chroma.sqlite3 not found in zip."

        # Move to production path
        if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
        shutil.copytree(sqlite_file.parent, CHROMA_PATH)
        shutil.rmtree(temp_dir)
        
        return True, "Success"
    except Exception as e:
        return False, str(e)

ready, status = initialize_system()

# --- 3. UI ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

if not ready:
    st.error(f"Startup Error: {status}")
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")

# --- 4. AI BRAIN (With Version Fix) ---
@st.cache_resource
def get_chain(_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_key)
        
        # Connect with Settings to bypass versioning KeyErrors
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        vectorstore = Chroma(
            client=client, 
            collection_name="langchain", 
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Use the context to answer. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. INTERFACE & BUTTONS ---
st.write("### Suggested Questions")
cols = st.columns(3)
btns = [
    "What is the focus of the Waste module?",
    "Tell me about the Recycling approach.",
    "Give me some eco-friendly tips."
]

for i, text in enumerate(btns):
    if cols[i].button(text):
        st.session_state.next_msg = text

# Display chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle Input
prompt = st.chat_input("Ask about the curriculum...")
final_prompt = prompt if prompt else st.session_state.get("next_msg")

if final_prompt:
    if "next_msg" in st.session_state: del st.session_state.next_msg
    
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"): st.markdown(final_prompt)

    chain = get_chain(api_key)
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Thinking..."):
                res = chain.invoke({"input": final_prompt, "chat_history": history})
                st.markdown(res["answer"])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    st.rerun()
