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
CHROMA_PATH = "chroma_db_final"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE EXTRACTION (Targeting your specific zip structure) ---
@st.cache_resource
def prepare_db():
    if not os.path.exists(CHROMA_PATH):
        if not os.path.exists(ZIP_PATH):
            return "Error: chroma_db.zip not found in GitHub root."
        try:
            if os.path.exists("temp_extract"):
                shutil.rmtree("temp_extract")
            
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall("temp_extract")
            
            # Based on your file: the data is in 'temp_extract/chroma_db/'
            source_inner_folder = os.path.join("temp_extract", "chroma_db")
            
            if os.path.exists(source_inner_folder):
                shutil.copytree(source_inner_folder, CHROMA_PATH)
                shutil.rmtree("temp_extract", ignore_errors=True)
                return "Success"
            else:
                return "Error: Could not find 'chroma_db' folder inside zip."
        except Exception as e:
            return f"Extraction Error: {e}"
    return "Success"

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
            ("system", "You are an assistant for Eco-Education. Use the context to answer: {context}"),
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

# --- 4. UI ---
st.title("🌱 Eco-Chatbot")
st.write("Curriculum Assistant by Ann Lewin-Benham")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OpenAI API Key in Streamlit Secrets.")
    st.stop()

if db_status != "Success":
    st.error(db_status)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# SUGGESTED PROMPT BUTTONS (Standard Styling)
st.write("### Suggested Questions")
c1, c2, c3 = st.columns(3)
if c1.button("What is the waste module?"):
    st.session_state.btn_input = "What is the waste module?"
if c2.button("Tell me about recycling"):
    st.session_state.btn_input = "Tell me about recycling"
if c3.button("Eco-friendly tips"):
    st.session_state.btn_input = "Give me some eco-friendly tips"

# History Display
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. CHAT LOGIC ---
user_query = st.chat_input("Ask about the eco-curriculum...")
final_query = user_query if user_query else st.session_state.get("btn_input")

if final_query:
    if "btn_input" in st.session_state: del st.session_state["btn_input"]
    
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
            with st.spinner("Analyzing curriculum..."):
                try:
                    res = chain.invoke({"input": final_query, "chat_history": history})
                    st.markdown(res["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
                except Exception as e:
                    st.error(f"Chat Error: {e}")
    st.rerun()
