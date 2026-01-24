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
# Using a unique folder name to force a clean extraction on this deploy
DB_DIR = "db_verified_eco_v1"
CHROMA_PATH = os.path.join(os.getcwd(), DB_DIR)
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def prepare_database():
    try:
        # Check if database already exists in the persistent path
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Database Online"

        if not os.path.exists(ZIP_NAME):
            return False, f"Missing {ZIP_NAME} in repository."

        # Verify the file isn't a Git LFS pointer
        if os.path.getsize(ZIP_NAME) < 10000:
            return False, "The zip file is a Git LFS pointer. Please re-upload the actual file to GitHub."

        # Clean extraction logic
        temp_dir = "temp_unzip"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall(temp_dir)

        # Look for the sqlite file regardless of nested folder structure
        sqlite_loc = next(Path(temp_dir).rglob("chroma.sqlite3"), None)
        if not sqlite_loc:
            return False, "Could not find chroma.sqlite3 inside the zip."

        if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
        shutil.copytree(sqlite_loc.parent, CHROMA_PATH)
        shutil.rmtree(temp_dir)
        
        return True, "Initialization Success"
    except Exception as e:
        return False, f"Setup Error: {e}"

# --- 3. UI LAYOUT ---
st.title("🌱 Eco-Education Curriculum Assistant")
st.caption("Knowledge base: Ann Lewin-Benham's Curriculum")

ready, status = prepare_database()

if not ready:
    st.error(status)
    st.stop()

# --- 4. RETRIEVAL ENGINE ---
@st.cache_resource
def get_retrieval_chain(_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_key)
        
        # FIX: Directly using PersistentClient avoids the KeyError: '_type'
        # This matches the 'langchain' collection name used in your ingest code
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant for the Eco-Education curriculum. Use the provided context to answer questions. Context: {context}"),
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

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested Topics
st.write("### Suggested Topics")
c1, c2, c3 = st.columns(3)
suggestions = [
    "What is the focus of the Waste module?",
    "Tell me about the Recycling approach.",
    "Give me some eco-friendly tips."
]

if c1.button(suggestions[0], use_container_width=True): st.session_state.trigger = suggestions[0]
if c2.button(suggestions[1], use_container_width=True): st.session_state.trigger = suggestions[1]
if c3.button(suggestions[2], use_container_width=True): st.session_state.trigger = suggestions[2]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Handling
user_query = st.chat_input("Ask a question about the curriculum...")
final_query = user_query if user_query else st.session_state.get("trigger")

if final_query:
    if "trigger" in st.session_state: del st.session_state.trigger
    
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
            with st.spinner("Searching the curriculum..."):
                response = chain.invoke({"input": final_query, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()
