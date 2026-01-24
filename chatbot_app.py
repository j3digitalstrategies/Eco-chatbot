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
CHROMA_PATH = "chroma_db_v3_stable"  # New name to force a fresh start
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE EXTRACTION ---
@st.cache_resource
def prepare_db():
    # If the folder doesn't exist, we must build it from the zip
    if not os.path.exists(CHROMA_PATH):
        if not os.path.exists(ZIP_PATH):
            return f"❌ {ZIP_PATH} not found in GitHub repository root."
        
        try:
            # Temporary extraction spot
            temp_dir = "temp_unzip"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Look for the folder that contains 'chroma.sqlite3'
            found_db_dir = None
            for root, dirs, files in os.walk(temp_dir):
                if "chroma.sqlite3" in files:
                    found_db_dir = root
                    break
            
            if found_db_dir:
                shutil.copytree(found_db_dir, CHROMA_PATH)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return "Success"
            else:
                return "❌ Database files missing inside the zip."
        except Exception as e:
            return f"❌ Extraction Error: {str(e)}"
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
            ("system", "You are an assistant for Eco-Education. Use the provided context to answer questions about the curriculum. Context: {context}"),
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

if db_status != "Success":
    st.error(db_status)
    st.stop()

if not api_key:
    st.error("🔑 OpenAI API Key missing in Streamlit Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# SUGGESTED PROMPT BUTTONS
st.write("### Suggested Questions")
cols = st.columns(3)
if cols[0].button("What is the waste module?"):
    st.session_state.queued_prompt = "What is the waste module?"
if cols[1].button("Tell me about recycling"):
    st.session_state.queued_prompt = "Tell me about recycling"
if cols[2].button("Eco-friendly tips"):
    st.session_state.queued_prompt = "Give me some eco-friendly tips"

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. CHAT LOGIC ---
user_input = st.chat_input("Ask a question about the curriculum...")

# Handle button click or typing
final_query = None
if user_input:
    final_query = user_input
elif "queued_prompt" in st.session_state:
    final_query = st.session_state.queued_prompt
    del st.session_state.queued_prompt

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
                with st.spinner("Searching Eco-Curriculum..."):
                    response = chain.invoke({"input": final_query, "chat_history": history})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Chat Error: {e}")
    st.rerun()
