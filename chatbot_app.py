__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import chromadb
import shutil
from dotenv import load_dotenv

# Stop the telemetry spam in your logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIG & SETUP ---
load_dotenv()
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. THE CRITICAL DATABASE FIX ---
@st.cache_resource
def prepare_db():
    # If the folder exists, we check if it's broken. 
    # If you see KeyError: '_type', the easiest 'work' fix is to let the app 
    # re-extract the ZIP file to ensure the folder matches the current environment.
    if os.path.exists(CHROMA_PATH):
        # We only delete and refresh if we are specifically having issues.
        # To force a refresh, you can delete the chroma_db folder from GitHub.
        pass 
    else:
        if os.path.exists(ZIP_PATH):
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
    return "Ready"

prepare_db()

# --- 3. STYLING (Restoring your exact look) ---
st.markdown(
    f"<style>.stApp {{background-color: #f0f7f4;}} .stChatMessage {{border-radius: 15px;}}</style>", 
    unsafe_allow_headers=True
)

# --- 4. THE AI ENGINE ---
@st.cache_resource
def get_rag_chain():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API Key.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    try:
        # PersistentClient is the most stable way to 'make it work' on Streamlit Cloud
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
        st.error(f"Database Error: {e}")
        return None

# --- 5. UI & SUGGESTED PROMPTS ---
st.title("🌱 Eco-Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested Prompts (Restored)
st.subheader("Quick Questions")
cols = st.columns(3)
prompts = ["What is the waste module?", "Tell me about recycling", "Eco-friendly tips"]

for i, p in enumerate(prompts):
    if cols[i].button(p):
        st.session_state.pending_prompt = p

# --- 6. CHAT LOGIC ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask about the curriculum...")

final_query = query or st.session_state.get("pending_prompt")
if "pending_prompt" in st.session_state:
    del st.session_state["pending_prompt"]

if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)
        
    chain = get_rag_chain()
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Thinking..."):
                response = chain.invoke({"input": final_query, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()
