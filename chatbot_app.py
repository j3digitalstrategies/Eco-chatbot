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
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db_final_prod")
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. THE BULLETPROOF EXTRACTION ---
@st.cache_resource
def prepare_database():
    # If the database folder exists, we are good to go
    if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
        return "Success"

    # Find the zip file (checking root and current dir)
    zip_path = None
    possible_paths = [ZIP_NAME, os.path.join(os.getcwd(), ZIP_NAME)]
    for p in possible_paths:
        if os.path.exists(p):
            zip_path = p
            break
            
    if not zip_path:
        return f"Error: {ZIP_NAME} not found. Ensure it is in your GitHub root."

    try:
        # 1. Clear old temp data
        temp_extract_dir = "temp_unzip_dir"
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # 2. Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # 3. Find where chroma.sqlite3 actually landed
        found_db_dir = None
        for root, dirs, files in os.walk(temp_extract_dir):
            if "chroma.sqlite3" in files:
                found_db_dir = root
                break
        
        if found_db_dir:
            shutil.copytree(found_db_dir, CHROMA_PATH)
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            return "Success"
        else:
            return "Error: chroma.sqlite3 not found inside the zip."
            
    except Exception as e:
        return f"Extraction Failure: {str(e)}"

db_status = prepare_database()

# --- 3. AI ENGINE ---
@st.cache_resource
def load_rag_engine(_api_key):
    if not _api_key: return None
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        # Using the PersistentClient to ensure it points to our extracted folder
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(
            client=client, 
            collection_name="langchain", 
            embedding_function=embeddings
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for Eco-Education curriculum. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"Engine Initialization Error: {e}")
        return None

# --- 4. INTERFACE ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

if db_status != "Success":
    st.error(db_status)
    st.info("Check if chroma_db.zip is at the top level of your GitHub repo.")
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# SUGGESTED QUESTIONS
st.write("### Suggested Questions")
c1, c2, c3 = st.columns(3)
if c1.button("What is the waste module?"):
    st.session_state.btn_query = "What is the waste module?"
if c2.button("Tell me about recycling"):
    st.session_state.btn_query = "Tell me about recycling"
if c3.button("Eco-friendly tips"):
    st.session_state.btn_query = "Give me some eco-friendly tips"

# History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. EXECUTION ---
user_input = st.chat_input("Ask a question...")
query = user_input if user_input else st.session_state.get("btn_query")

if query:
    if "btn_query" in st.session_state:
        del st.session_state["btn_query"]
        
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    chain = load_rag_engine(api_key)
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Searching the Eco-Curriculum..."):
                try:
                    res = chain.invoke({"input": query, "chat_history": history})
                    st.markdown(res["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
                except Exception as e:
                    st.error(f"Chat Error: {e}")
    st.rerun()
