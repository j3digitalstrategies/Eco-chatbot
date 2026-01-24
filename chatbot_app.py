__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import chromadb
from dotenv import load_dotenv

# Disable the broken telemetry to clean up your logs
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

# --- 3. THE AI ENGINE (Direct Client Fix) ---
@st.cache_resource
def get_chatbot_chain():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API Key.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    try:
        # Using the PersistentClient bypasses the version migration bug (KeyError: '_type')
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # We target the 'langchain' collection which is the default for your builder script
        vectorstore = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        
        system_prompt = (
            "You are a helpful assistant specialized in Eco-Education. "
            "Use the provided context to answer questions. "
            "If you can't find the answer, state that clearly.\n\n"
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
        st.error(f"Database Error: {e}")
        return None

# --- 4. UI ---
st.title("🌱 Eco-Chatbot")
st.caption(db_status)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. SUGGESTED PROMPTS ---
if not st.session_state.messages:
    prompts = [
        "What are the core principles of Eco-Education?",
        "How does the garden serve as a classroom?",
        "Describe the approach to documentation."
    ]
    cols = st.columns(3)
    for i, p in enumerate(prompts):
        if cols[i].button(p):
            st.session_state.pending_input = p

# --- 6. CHAT LOGIC ---
query = st.chat_input("Ask a question...")
if "pending_input" in st.session_state:
    query = st.session_state.pop("pending_input")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    chain = get_chatbot_chain()
    if chain:
        with st.chat_message("assistant"):
            try:
                # Convert history for LangChain
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                
                with st.spinner("Analyzing documents..."):
                    response = chain.invoke({"input": query, "chat_history": history})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Use st.rerun() to ensure the chat history displays correctly
    st.rerun()
