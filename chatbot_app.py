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
# Versioning the path ensures a fresh start every time you update
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db_final_v8")
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. EXTRACTION LOGIC ---
@st.cache_resource
def initialize_database():
    try:
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Ready"

        zip_p = Path(ZIP_NAME)
        if not zip_p.exists():
            return False, f"File {ZIP_NAME} not found in repository."

        # Verify file size (Detects if it's a 1KB pointer vs a 40MB database)
        size_mb = zip_p.stat().st_size / (1024 * 1024)
        if size_mb < 0.1: # Less than 100KB is definitely a broken link
            return False, f"Upload Error: The file is only {size_mb:.4f} MB. This is a GitHub 'pointer' link. Please delete it and re-upload by dragging the file into your browser."

        # Unzip process
        temp_path = Path("temp_v8")
        if temp_path.exists(): shutil.rmtree(temp_path)
        temp_path.mkdir()

        with zipfile.ZipFile(zip_p, 'r') as z:
            z.extractall(temp_path)

        # Find the database file
        found_db = next(temp_path.rglob("chroma.sqlite3"), None)
        if not found_db:
            return False, "Could not find chroma.sqlite3 inside the zip."

        if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
        shutil.copytree(found_db.parent, CHROMA_PATH)
        shutil.rmtree(temp_path)
        
        return True, "Success"
    except Exception as e:
        return False, str(e)

success, msg = initialize_database()

# --- 3. UI ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

if not success:
    st.error(msg)
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

# --- 4. ENGINE ---
@st.cache_resource
def load_engine(_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_key)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(client=client, collection_name="langchain", embedding_function=embeddings)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Context: {context}"),
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

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. FULL SENTENCE PROMPTS ---
st.write("### Suggested Questions")
col1, col2, col3 = st.columns(3)
if col1.button("What is the waste module?"):
    st.session_state.hold_q = "What is the waste module?"
if col2.button("Tell me about recycling"):
    st.session_state.hold_q = "Tell me about recycling"
if col3.button("Give me some eco-friendly tips"):
    st.session_state.hold_q = "Give me some eco-friendly tips"

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# --- 6. CHAT EXECUTION ---
raw_input = st.chat_input("Type your question here...")
active_q = raw_input if raw_input else st.session_state.get("hold_q")

if active_q:
    if "hold_q" in st.session_state: del st.session_state.hold_q
    st.session_state.messages.append({"role": "user", "content": active_q})
    with st.chat_message("user"): st.markdown(active_q)

    rag = load_engine(api_key)
    if rag:
        with st.chat_message("assistant"):
            history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) 
                       for m in st.session_state.messages[:-1]]
            with st.spinner("Searching curriculum..."):
                try:
                    res = rag.invoke({"input": active_q, "chat_history": history})
                    st.markdown(res["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
                except Exception as e:
                    st.error(f"Error: {e}")
    st.rerun()
