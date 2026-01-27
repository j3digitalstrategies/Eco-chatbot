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
DB_DIR = "final_eco_db_instance"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- 2. FORCED INITIALIZATION ---
@st.cache_resource
def force_init_db():
    try:
        # 1. Clean up old attempts
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        if not os.path.exists(ZIP_NAME):
            return False, f"CRITICAL: {ZIP_NAME} not found in repo."

        # 2. Extract files
        temp_dir = "extraction_temp"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_dir)

        # 3. Find the database (handles nested folders or Mac junk)
        sqlite_file = next(Path(temp_dir).rglob("chroma.sqlite3"), None)
        if not sqlite_file:
            return False, "Could not find chroma.sqlite3 inside zip."

        # 4. Move actual DB content to the CHROMA_PATH
        shutil.copytree(sqlite_file.parent, CHROMA_PATH)
        shutil.rmtree(temp_dir)
        return True, "Database Extracted"
    except Exception as e:
        return False, f"Extraction failed: {str(e)}"

# --- 3. UI DISPLAY ---
st.title("🌱 Eco-Education Assistant")
st.write("Curriculum: Ann Lewin-Benham")

# Check database
ready, msg = force_init_db()
if not ready:
    st.error(msg)
    st.stop()

# --- 4. ENGINE (The Fix for KeyError: '_type') ---
@st.cache_resource
def get_chain(_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_key)
        
        # We initialize the client explicitly to bypass standard LangChain validation
        # This is the specific fix for the 'KeyError: _type'
        settings = chromadb.Settings(allow_reset=True, anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=settings)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Use ONLY the context provided to answer. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None

# --- 5. CHAT SYSTEM ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggestions as a single row of buttons
st.write("### Quick Topics")
c1, c2, c3 = st.columns(3)
if c1.button("Waste module focus?"): st.session_state.btn_q = "What is the focus of the Waste module?"
if c2.button("Recycling approach?"): st.session_state.btn_q = "Tell me about the Recycling approach."
if c3.button("Eco-tips?"): st.session_state.btn_q = "Give me some eco-friendly tips."

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle Query
raw_input = st.chat_input("Type your question here...")
final_q = raw_input if raw_input else st.session_state.get("btn_q")

if final_q:
    if "btn_q" in st.session_state: del st.session_state.btn_q
    
    st.session_state.messages.append({"role": "user", "content": final_q})
    with st.chat_message("user"):
        st.markdown(final_q)

    # Use key from secrets
    api_key = st.secrets["OPENAI_API_KEY"]
    chain = get_chain(api_key)
    
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Consulting curriculum..."):
                response = chain.invoke({"input": final_q, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()
