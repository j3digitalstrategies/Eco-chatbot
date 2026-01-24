__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import chromadb
import shutil
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIG ---
load_dotenv()
# We use a unique name to ensure we aren't using old, broken data
CHROMA_PATH = "chroma_db_deploy_final"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. FORCED DATABASE EXTRACTION ---
@st.cache_resource
def prepare_db():
    # If the folder exists, we delete it to ensure a clean deploy every time
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    if not os.path.exists(ZIP_PATH):
        return f"❌ Error: {ZIP_PATH} not found in GitHub. Please check your repo root."
    
    try:
        temp_dir = "extraction_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # We look for the ACTUAL folder containing the sqlite file
        db_root = None
        for root, dirs, files in os.walk(temp_dir):
            if "chroma.sqlite3" in files:
                db_root = root
                break
        
        if db_root:
            shutil.copytree(db_root, CHROMA_PATH)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return "Success"
        else:
            return "❌ Error: chroma.sqlite3 not found inside the zip file."
            
    except zipfile.BadZipFile:
        return "❌ Error: The zip file is corrupted. Try re-uploading to GitHub."
    except Exception as e:
        return f"❌ Extraction Error: {str(e)}"

db_status = prepare_db()

# --- 3. AI ENGINE ---
@st.cache_resource
def get_rag_chain(_api_key):
    if not _api_key: return None
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
            ("system", "You are an assistant for Eco-Education. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"Engine Load Error: {e}")
        return None

# --- 4. UI ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

# Check database status
if db_status != "Success":
    st.error(db_status)
    st.stop()

# Check API Key
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 Please add your OPENAI_API_KEY to Streamlit Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# SUGGESTED QUESTIONS
st.write("### Suggested Questions")
c1, c2, c3 = st.columns(3)
if c1.button("What is the waste module?"):
    st.session_state.pending_query = "What is the waste module?"
if c2.button("Tell me about recycling"):
    st.session_state.pending_query = "Tell me about recycling"
if c3.button("Eco-friendly tips"):
    st.session_state.pending_query = "Give me some eco-friendly tips"

# Display Chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. LOGIC ---
user_input = st.chat_input("Ask about the curriculum...")

# Handle input source
query = user_input if user_input else st.session_state.get("pending_query")

if query:
    if "pending_query" in st.session_state:
        del st.session_state["pending_query"]
        
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    chain = get_rag_chain(api_key)
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Consulting curriculum..."):
                try:
                    res = chain.invoke({"input": query, "chat_history": history})
                    st.markdown(res["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
                except Exception as e:
                    st.error(f"Chat Error: {e}")
    st.rerun()
