__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
from dotenv import load_dotenv

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Ann Lewin-Benham AI", layout="wide")

# LangChain Imports
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# --- 2. DATABASE MANAGEMENT ---
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

@st.cache_resource
def manage_database():
    if not os.path.exists(CHROMA_PATH):
        if os.path.exists(ZIP_PATH):
            try:
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(".")
                return "Database ready."
            except Exception as e:
                return f"Error: {str(e)}"
    return "Database ready."

db_msg = manage_database()

# --- 3. API KEY ---
def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("🔑 OPENAI_API_KEY is missing in Streamlit Secrets.")
    st.stop()

# --- 4. RAG ENGINE ---
@st.cache_resource
def setup_rag_chain():
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=api_key)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on Ann Lewin-Benham. Use the context to answer:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        st.error(f"AI Setup Error: {e}")
        return None

# --- 5. UI & SESSION STATE ---
st.title("🌱 Ann Lewin-Benham AI")
st.caption(f"System: {db_msg}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. SUGGESTED PROMPTS (EXACTLY 3) ---
suggested_prompts = [
    "What are the core principles of Eco-Education?",
    "How does the garden serve as a classroom?",
    "Describe Ann's approach to documentation."
]

if not st.session_state.messages:
    st.markdown("### Suggested Topics")
    cols = st.columns(len(suggested_prompts))
    for i, prompt in enumerate(suggested_prompts):
        if cols[i].button(prompt):
            st.session_state.active_input = prompt

# --- 7. INPUT LOGIC ---
user_input = st.chat_input("Ask a question about the curriculum...")

if "active_input" in st.session_state:
    user_input = st.session_state.pop("active_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        try:
            chain = setup_rag_chain()
            if chain:
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                
                with st.spinner("Searching documents..."):
                    response = chain.invoke({"input": user_input, "chat_history": history})
                    full_answer = response["answer"]
                    st.markdown(full_answer)
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})
        except Exception as e:
            st.error(f"Processing Error: {e}")

    st.rerun()
