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

# --- 1. CONFIG & API ---
load_dotenv()
CHROMA_PATH = "chroma_db_stable"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# Styling Fix: Use unsafe_allow_html=True (NOT unsafe_allow_headers)
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #4CAF50;
        background-color: white;
        color: #4CAF50;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("OPENAI_API_KEY")

# --- 2. DATABASE EXTRACTION ---
@st.cache_resource
def prepare_db():
    if not os.path.exists(CHROMA_PATH):
        if os.path.exists(ZIP_PATH):
            try:
                if os.path.exists("temp_extract"):
                    shutil.rmtree("temp_extract")
                
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall("temp_extract")
                
                found_path = None
                for root, dirs, files in os.walk("temp_extract"):
                    if "chroma.sqlite3" in files:
                        found_path = root
                        break
                
                if found_path:
                    if os.path.exists(CHROMA_PATH):
                        shutil.rmtree(CHROMA_PATH)
                    shutil.copytree(found_path, CHROMA_PATH)
                    shutil.rmtree("temp_extract", ignore_errors=True)
                    return "Database Ready"
                return "Error: chroma.sqlite3 not found in zip"
            except Exception as e:
                return f"Extraction Error: {e}"
        return "Error: zip file missing"
    return "Database Ready"

db_status = prepare_db()

# --- 3. AI ENGINE ---
@st.cache_resource
def get_rag_chain(_api_key):
    if not _api_key:
        return None
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for Eco-Education. Use the following context to answer: {context}"),
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
st.caption("Curriculum Assistant by Ann Lewin-Benham")

if not api_key:
    st.error("🔑 OpenAI API Key missing in Streamlit Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested Prompts (Restored)
st.write("### Suggested Questions")
cols = st.columns(3)
suggestions = ["What is the waste module?", "Tell me about recycling", "Eco-friendly tips"]

for i, prompt_text in enumerate(suggestions):
    if cols[i].button(prompt_text):
        st.session_state.prompt_trigger = prompt_text

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. CHAT ACTION ---
user_input = st.chat_input("Ask a question...")

# Use either text input or button trigger
final_query = user_input
if "prompt_trigger" in st.session_state:
    final_query = st.session_state.prompt_trigger
    del st.session_state.prompt_trigger

if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)
        
    chain = get_rag_chain(api_key)
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            try:
                with st.spinner("Thinking..."):
                    response = chain.invoke({"input": final_query, "chat_history": history})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Chat Error: {e}")
    st.rerun()
