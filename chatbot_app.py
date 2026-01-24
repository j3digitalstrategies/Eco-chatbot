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
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_final_v6")
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. THE FINAL EXTRACTION LOGIC ---
@st.cache_resource
def startup_logic():
    try:
        # 1. Skip if already extracted
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, None

        zip_path = Path(ZIP_NAME)
        
        # 2. Check if file exists
        if not zip_path.exists():
            return False, f"File '{ZIP_NAME}' not found in root directory."

        # 3. Check if it's a valid zip (Detects Git LFS issues)
        if not zipfile.is_zipfile(zip_path):
            file_size = zip_path.stat().st_size
            return False, f"'{ZIP_NAME}' is not a valid zip file (Size: {file_size} bytes). If this is very small, GitHub might be storing it as an LFS pointer instead of the actual file."

        # 4. Extract to temporary folder
        temp_dir = Path("temp_v6")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        # 5. Deep search for the database
        sqlite_loc = next(temp_dir.rglob("chroma.sqlite3"), None)
        if not sqlite_loc:
            return False, "Could not find 'chroma.sqlite3' anywhere inside the zip."

        # 6. Move to final destination
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        shutil.copytree(sqlite_loc.parent, CHROMA_PATH)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        return True, None

    except Exception as e:
        return False, str(e)

# Run Startup
success, err = startup_logic()

# --- 3. UI ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

if not success:
    st.error(f"Startup Error: {err}")
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in Secrets.")
    st.stop()

# --- 4. ENGINE ---
@st.cache_resource
def get_engine(_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_key)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(client=client, collection_name="langchain", embedding_function=embeddings)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a curriculum expert. Context: {context}"),
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

# Suggestions
cols = st.columns(3)
if cols[0].button("Waste Module"): st.session_state.q = "What is the waste module?"
if cols[1].button("Recycling"): st.session_state.q = "Tell me about recycling"
if cols[2].button("Eco Tips"): st.session_state.q = "Give me eco-friendly tips"

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# --- 5. CHAT ---
prompt = st.chat_input("Ask me anything...")
final_q = prompt if prompt else st.session_state.get("q")

if final_q:
    if "q" in st.session_state: del st.session_state.q
    st.session_state.messages.append({"role": "user", "content": final_q})
    with st.chat_message("user"): st.markdown(final_q)

    chain = get_engine(api_key)
    if chain:
        with st.chat_message("assistant"):
            hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) 
                    for m in st.session_state.messages[:-1]]
            with st.spinner("Thinking..."):
                res = chain.invoke({"input": final_q, "chat_history": hist})
                st.markdown(res["answer"])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    st.rerun()
