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
# We change the version name to force Streamlit to wipe any old, broken data
VER = "v9_final"
CHROMA_PATH = os.path.join(os.getcwd(), f"chroma_data_{VER}")
ZIP_NAME = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. THE REPAIR & EXTRACTION LOGIC ---
@st.cache_resource
def setup_db():
    try:
        # Check if already setup
        if os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            return True, "Database Active"

        zip_p = Path(ZIP_NAME)
        if not zip_p.exists():
            return False, f"File {ZIP_NAME} not found. Please check your GitHub file list."

        # Check file size to confirm it's the real data
        f_size = zip_p.stat().st_size / (1024 * 1024)
        if f_size < 0.5:
            return False, f"Upload Error: The zip file is too small ({f_size:.2f} MB). It is likely a GitHub LFS pointer. Delete it and re-upload by dragging and dropping."

        # Extraction
        tmp = Path(f"tmp_{VER}")
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir()

        with zipfile.ZipFile(zip_p, 'r') as z:
            z.extractall(tmp)

        # Deep Search: Find where chroma.sqlite3 is hiding inside the zip
        db_location = next(tmp.rglob("chroma.sqlite3"), None)
        if not db_location:
            return False, "Found the zip, but 'chroma.sqlite3' is missing inside it."

        # Move the folder containing the sqlite3 file to our final path
        if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
        shutil.copytree(db_location.parent, CHROMA_PATH)
        shutil.rmtree(tmp)
        
        return True, "Success"
    except Exception as e:
        return False, f"System Error: {str(e)}"

# Initialize
ready, status = setup_db()

# --- 3. UI ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

if not ready:
    st.error(status)
    st.info("If you just uploaded a new file, please go to 'Manage App' -> 'Reboot' to refresh the cache.")
    st.stop()

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit Secrets.")
    st.stop()

# --- 4. ENGINE ---
@st.cache_resource
def get_chat_engine(_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_key)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(client=client, collection_name="langchain", embedding_function=embeddings)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for the Eco-Education curriculum. Answer based on context: {context}"),
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

# --- 5. FULL SENTENCE BUTTONS ---
st.write("### Suggested Questions")
b1, b2, b3 = st.columns(3)
if b1.button("What is the waste module?"):
    st.session_state.pending_q = "What is the waste module?"
if b2.button("Tell me about recycling"):
    st.session_state.pending_q = "Tell me about recycling"
if b3.button("Give me some eco-friendly tips"):
    st.session_state.pending_q = "Give me some eco-friendly tips"

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# --- 6. CHAT LOGIC ---
chat_input = st.chat_input("Ask a question...")
final_q = chat_input if chat_input else st.session_state.get("pending_q")

if final_q:
    if "pending_q" in st.session_state: del st.session_state.pending_q
    st.session_state.messages.append({"role": "user", "content": final_q})
    with st.chat_message("user"): st.markdown(final_q)

    engine = get_chat_engine(api_key)
    if engine:
        with st.chat_message("assistant"):
            hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) 
                    for m in st.session_state.messages[:-1]]
            with st.spinner("Analyzing..."):
                try:
                    res = engine.invoke({"input": final_q, "chat_history": hist})
                    st.markdown(res["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
                except Exception as e:
                    st.error(f"Chat Error: {e}")
    st.rerun()
