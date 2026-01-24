__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import chromadb
import shutil
from dotenv import load_dotenv

# Silence the telemetry errors seen in your logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIG & STYLING ---
load_dotenv()
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #f0f7f4;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_headers=True)

# --- 2. DATABASE RECOVERY ---
@st.cache_resource
def prepare_db():
    # If the folder exists but is corrupted (causing the KeyError), 
    # we might need to clear it, but for now, we try to extract if missing.
    if not os.path.exists(CHROMA_PATH):
        if os.path.exists(ZIP_PATH):
            try:
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(".")
                return "✅ Database extracted."
            except Exception as e:
                return f"⚠️ Unzip failed: {e}"
        return "⚠️ Database zip missing."
    return "✅ Database ready."

db_status = prepare_db()

# --- 3. THE AI ENGINE ---
@st.cache_resource
def get_rag_chain():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API Key.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    try:
        # THE FIX: Explicitly using the PersistentClient to handle the local files
        # This bypasses the internal discovery logic that triggers the KeyError.
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        
        system_prompt = (
            "You are a helpful assistant specialized in Eco-Education curriculum. "
            "Use the provided context to answer questions accurately.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 5}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        # Catching the specific configuration error
        st.error(f"Database Configuration Error: {e}")
        st.info("Try deleting the 'chroma_db' folder in your repo to allow a clean rebuild.")
        return None

# --- 4. UI & SUGGESTED PROMPTS ---
st.title("🌱 Eco-Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Quick Questions")
cols = st.columns(3)
prompts = ["What is the waste module?", "Tell
