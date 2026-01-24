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
CHROMA_PATH = "chroma_db_final_v2" 
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE EXTRACTION (Aggressive Fix) ---
@st.cache_resource
def prepare_db():
    if not os.path.exists(CHROMA_PATH):
        if not os.path.exists(ZIP_PATH):
            return f"Error: {ZIP_PATH} not found in repository root."
        
        try:
            # Clean up old attempts
            if os.path.exists("temp_extract"):
                shutil.rmtree("temp_extract")
            
            # Open and extract
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall("temp_extract")
            
            # Based on your zip structure: the sqlite3 is in temp_extract/chroma_db/
            source_folder = os.path.join("temp_extract", "chroma_db")
            
            if os.path.exists(source_folder):
                shutil.copytree(source_folder, CHROMA_PATH)
                shutil.rmtree("temp_extract", ignore_errors=True)
                return "Success"
            else:
                # Fallback: search for chroma.sqlite3 anywhere in the zip
                for root, dirs, files in os.walk("temp_extract"):
                    if "chroma.sqlite3" in files:
                        shutil.copytree(root, CHROMA_PATH)
                        return "Success"
                return "Error: Could not find chroma.sqlite3 folder inside zip."
        except zipfile.BadZipFile:
            return "Error: The file 'chroma_db.zip' on GitHub is corrupted or not a valid zip."
        except Exception as e:
            return f"Extraction Error: {str(e)}"
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

api_key = st.secrets.get("OPENAI_API_KEY")

if db_status != "Success":
    st.error(f"Database Issue: {db_status}")
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
    st.session_state.prompt_queue = "What is the waste module?"
if cols[1].button("Tell me about recycling"):
    st.session_state.prompt_queue = "Tell me about recycling"
if cols[2].button("Eco-friendly tips"):
    st.session_state.prompt_queue = "Give me some eco-friendly tips"

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. CHAT ACTION ---
user_input = st.chat_input("Ask a question...")

final_query = None
if user_input:
    final_query = user_input
elif "prompt_queue" in st.session_state:
    final_query = st.session_state.prompt_queue
    del st.session_state.prompt_queue

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
                with st.spinner("Searching Curriculum..."):
                    response = chain.invoke({"input": final_query, "chat_history": history})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Chat Error: {e}")
    st.rerun()
