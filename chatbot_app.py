__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
from dotenv import load_dotenv

# CRITICAL: Disable telemetry BEFORE importing Chroma to fix your log errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETUP & PATHS ---
load_dotenv()
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Ann Lewin-Benham AI", layout="wide")

# --- 2. DATABASE EXTRACTION ---
@st.cache_resource
def prepare_db():
    if not os.path.exists(CHROMA_PATH):
        if os.path.exists(ZIP_PATH):
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
            return "✅ Database extracted."
        return "⚠️ Database zip not found."
    return "✅ Database ready."

status_msg = prepare_db()

# --- 3. AI ENGINE (Matching your specific script) ---
@st.cache_resource
def get_rag_chain():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API Key.")
        return None

    # Using the exact embedding model from your approved script
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    
    system_prompt = (
        "You are an expert on Ann Lewin-Benham's Eco-Education. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 5}),
        create_stuff_documents_chain(llm, prompt)
    )
    return chain

# --- 4. UI ---
st.title("🌱 Ann Lewin-Benham AI")
st.caption(status_msg)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Suggested Prompts
if not st.session_state.messages:
    prompts = [
        "What are the core principles of Eco-Education?",
        "How does the garden serve as a classroom?",
        "Describe Ann's approach to documentation."
    ]
    cols = st.columns(3)
    for i, p in enumerate(prompts):
        if cols[i].button(p):
            st.session_state.user_query = p

# Chat Logic
query = st.chat_input("Ask a question...")
if "user_query" in st.session_state:
    query = st.session_state.pop("user_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    chain = get_rag_chain()
    if chain:
        with st.chat_message("assistant"):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            response = chain.invoke({"input": query, "chat_history": history})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
