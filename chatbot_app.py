__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import chromadb
from dotenv import load_dotenv

# Disable the broken telemetry to keep logs clean
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIG ---
load_dotenv()
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

st.set_page_config(page_title="Eco-Chatbot", layout="wide")

# --- 2. DATABASE RECOVERY ---
@st.cache_resource
def prepare_db():
    if not os.path.exists(CHROMA_PATH):
        if os.path.exists(ZIP_PATH):
            try:
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(".")
                return "✅ Database extracted."
            except Exception as e:
                return f"⚠️ Unzip failed: {e}"
        return "⚠️ Database zip missing."
    return "✅ Database ready."

db_status = prepare_db()

# --- 3. THE AI ENGINE (Fixing the KeyError: '_type') ---
@st.cache_resource
def get_rag_chain():
    # Attempt to get API key from Streamlit Secrets or .env
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API Key. Please add it to Streamlit Secrets.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    try:
        # We use PersistentClient to bypass version-mismatch bugs
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # 'langchain' is the default collection name used by LangChain's Chroma wrapper
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        
        system_prompt = (
            "You are a helpful assistant specialized in Eco-Education curriculum. "
            "Use the provided context to answer questions accurately. "
            "If the answer isn't in the context, politely say you don't know.\n\n"
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
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

# --- 4. UI ---
st.title("🌱 Eco-Chatbot")
st.caption(f"Status: {db_status}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. INTERACTION ---
query = st.chat_input("Ask about the curriculum...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    chain = get_rag_chain()
    if chain:
        with st.chat_message("assistant"):
            try:
                # Format history for LangChain
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                
                with st.spinner("Analyzing curriculum..."):
                    response = chain.invoke({"input": query, "chat_history": history})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Response Error: {e}")

    st.rerun()
