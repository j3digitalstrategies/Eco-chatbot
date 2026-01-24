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
# Unique folder name to ensure no collisions with old deployments
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_prod_v5")
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. SAFE EXTRACTION ENGINE ---
@st.cache_resource
def initialize_system():
    """Extracts zip and prepares the RAG environment in one safe step."""
    try:
        # Check if already done
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return "Ready", None

        # 1. Find the zip file
        zip_path = Path(ZIP_NAME)
        if not zip_path.exists():
            return "Error", f"Could not find {ZIP_NAME} in the repository root."

        # 2. Setup temp extraction
        temp_dir = Path("temp_extract_v5")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        # 3. Unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 4. Search for the sqlite file (recursive search)
        sqlite_file = next(temp_dir.rglob("chroma.sqlite3"), None)
        if not sqlite_file:
            return "Error", "The file 'chroma.sqlite3' was not found inside the zip."

        # 5. Move the folder containing the sqlite file to our final path
        db_source_dir = sqlite_file.parent
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        shutil.copytree(db_source_dir, CHROMA_PATH)

        # 6. Cleanup
        shutil.rmtree(temp_dir)
        return "Ready", None

    except Exception as e:
        return "Error", str(e)

# Run the initialization
status, error_msg = initialize_system()

# --- 3. UI LAYOUT ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

# Handle Initialization Errors
if status == "Error":
    st.error(f"System Failed to Start: {error_msg}")
    st.info("Check if your chroma_db.zip is valid and contains the chroma.sqlite3 file.")
    st.stop()

# Check Secrets
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OPENAI_API_KEY is missing in Streamlit Secrets!")
    st.stop()

# --- 4. CHAT ENGINE ---
@st.cache_resource
def get_chain(_api_key):
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
            ("system", "You are an assistant for Eco-Education. Use the context to answer: {context}"),
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

# Chat History setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggestions
st.write("### Quick Questions")
c1, c2, c3 = st.columns(3)
if c1.button("Waste module info"):
    st.session_state.auto_query = "What is the waste module?"
if c2.button("About recycling"):
    st.session_state.auto_query = "Tell me about recycling"
if c3.button("Eco tips"):
    st.session_state.auto_query = "Give me some eco-friendly tips"

# Display Messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. EXECUTION ---
user_input = st.chat_input("Ask about the curriculum...")
query = user_input if user_input else st.session_state.get("auto_query")

if query:
    if "auto_query" in st.session_state:
        del st.session_state["auto_query"]
        
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    rag_chain = get_chain(api_key)
    if rag_chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Processing..."):
                try:
                    res = rag_chain.invoke({"input": query, "chat_history": history})
                    st.markdown(res["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
                except Exception as e:
                    st.error(f"Execution Error: {e}")
    st.rerun()
