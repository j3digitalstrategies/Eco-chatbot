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
CHROMA_PATH = "chroma_db_stable"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# Styling
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE RECOVERY ---
@st.cache_resource
def prepare_db():
    if not os.path.exists(CHROMA_PATH):
        if os.path.exists(ZIP_PATH):
            try:
                if os.path.exists("temp_extract"):
                    shutil.rmtree("temp_extract")
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall("temp_extract")
                
                items = [f for f in os.listdir("temp_extract") if not f.startswith("__")]
                source = os.path.join("temp_extract", items[0]) if items else "temp_extract"
                
                if os.path.exists(CHROMA_PATH):
                    shutil.rmtree(CHROMA_PATH)
                shutil.move(source, CHROMA_PATH)
                shutil.rmtree("temp_extract", ignore_errors=True)
                return "✅ Database Ready"
            except Exception as e:
                return f"⚠️ DB Error: {e}"
    return "✅ Ready"

prepare_db()

# --- 3. THE AI ENGINE ---
@st.cache_resource
def get_rag_chain():
    # Check both Secrets and .env
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("🔑 API Key not found. Please check 'Secrets' in Streamlit Cloud.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    try:
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
        st.error(f"Engine Load Error: {e}")
        return None

# --- 4. UI ---
st.title("🌱 Eco-Chatbot")
st.markdown("### — by Ann Lewin-Benham")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Buttons
st.subheader("Quick Questions")
cols = st.columns(3)
prompts = ["What is the waste module?", "Tell me about recycling", "Eco-friendly tips"]

for i, p in enumerate(prompts):
    if cols[i].button(p):
        st.session_state.pending_prompt = p

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. CHAT LOGIC ---
query = st.chat_input("Ask about the curriculum...")

final_query = query
if st.session_state.get("pending_prompt"):
    final_query = st.session_state.pending_prompt
    del st.session_state["pending_prompt"]

if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)
        
    chain = get_rag_chain()
    if chain:
        with st.chat_message("assistant"):
            history = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                else:
                    history.append(AIMessage(content=m["content"]))
            
            with st.spinner("Searching curriculum..."):
                try:
                    response = chain.invoke({"input": final_query, "chat_history": history})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    st.error(f"Response Error: {e}")
    st.rerun()
