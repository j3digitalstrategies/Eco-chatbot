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
# Using a specific versioned folder to ensure a clean slate on deploy
DB_COLLECTION_NAME = "langchain"
PERSIST_DIR = os.path.join(os.getcwd(), "db_v14_prod")
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. ROBUST DATABASE INITIALIZATION ---
@st.cache_resource
def initialize_database():
    try:
        # If the specific versioned directory doesn't exist, extract it
        if not os.path.exists(PERSIST_DIR) or not os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
            if not os.path.exists(ZIP_NAME):
                return False, f"Missing {ZIP_NAME} in repository."
            
            # Check for Git LFS pointer files (small text files instead of real zip)
            if os.path.getsize(ZIP_NAME) < 5000:
                return False, "The zip file on GitHub is a pointer (LFS). Please re-upload by dragging the zip directly into the GitHub web interface."

            temp_extract = "temp_unzip"
            if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
            
            with zipfile.ZipFile(ZIP_NAME, 'r') as z:
                z.extractall(temp_extract)
            
            # Find the actual folder containing chroma.sqlite3
            sqlite_file = next(Path(temp_extract).rglob("chroma.sqlite3"), None)
            if not sqlite_file:
                return False, "Could not find chroma.sqlite3 inside the zip."
            
            if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
            shutil.copytree(sqlite_file.parent, PERSIST_DIR)
            shutil.rmtree(temp_extract)
            
        return True, "Database Ready"
    except Exception as e:
        return False, f"Initialization Error: {str(e)}"

# --- 3. UI HEADER ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

db_ok, db_status = initialize_database()

if not db_ok:
    st.error(db_status)
    st.stop()

# --- 4. AI CHAIN LOGIC ---
@st.cache_resource
def get_retrieval_chain(_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        
        # Explicitly use the PersistentClient to avoid the _type KeyError
        # This approach is more resilient to version mismatches
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        
        vectorstore = Chroma(
            client=client,
            collection_name=DB_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Use only the provided context. Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(llm, prompt)
        )
    except Exception as e:
        st.error(f"Failed to load AI Engine: {e}")
        return None

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested Questions (Centered Layout)
st.write("### Suggested Topics")
cols = st.columns(3)
suggestions = [
    "What is the focus of the Waste module?",
    "Tell me about the Recycling approach.",
    "Give me some eco-friendly tips."
]

for i, text in enumerate(suggestions):
    if cols[i].button(text, use_container_width=True):
        st.session_state.pending_query = text

# Show History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle Input
user_input = st.chat_input("Ask a question about the curriculum...")
final_query = user_input if user_input else st.session_state.get("pending_query")

if final_query:
    if "pending_query" in st.session_state: 
        del st.session_state.pending_query
    
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)

    api_key = st.secrets.get("OPENAI_API_KEY")
    chain = get_retrieval_chain(api_key)
    
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Consulting curriculum..."):
                response = chain.invoke({"input": final_query, "chat_history": history})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    
    st.rerun()
