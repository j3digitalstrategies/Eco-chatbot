__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
from dotenv import load_dotenv

# --- 1. MANDATORY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Ann Lewin-Benham AI", layout="wide")

# LangChain Imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 2. DATABASE MANAGEMENT ---
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

@st.cache_resource
def manage_database():
    if not os.path.exists(CHROMA_PATH) and os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
            return "ready"
        except Exception as e:
            return f"Database Error: {str(e)}"
    return "ready"

db_status = manage_database()

# --- 3. API KEY CHECK ---
# This is the most common reason for "no response"
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY is missing! Please add it to Streamlit Secrets.")
        st.stop()

DOCUMENT_AUTHOR = "Ann Lewin-Benham"
DOCUMENT_TITLE = "Eco-Education for Young Children"

# --- 4. RAG ENGINE SETUP ---
@st.cache_resource
def setup_rag_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question based on history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are the expert for {DOCUMENT_AUTHOR}. Answer using context:\n\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 5. UI LAYOUT ---
st.title(f"🌱 {DOCUMENT_TITLE}")
st.markdown(f"**Expert Guidance by {DOCUMENT_AUTHOR}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. SUGGESTED PROMPTS ---
suggested_prompts = [
    "What are the core principles of Eco-Education?",
    "How does the garden serve as a classroom?",
    "Explain Ann's approach to documentation."
]

# --- 7. DISPLAY CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. INPUT HANDLING ---
if not st.session_state.messages:
    st.info("Select a topic to start or type your own question below.")
    cols = st.columns(len(suggested_prompts))
    for i, prompt in enumerate(suggested_prompts):
        if cols[i].button(prompt):
            st.session_state.pending_input = prompt

user_input = st.chat_input("Ask a question...")

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# --- 9. EXECUTION ---
if user_input:
    st.chat_message("user").markdown(user_input)
    
    chat_history = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        try:
            rag_chain = setup_rag_chain()
            res_box = st.empty()
            full_res = ""
            
            for chunk in rag_chain.stream({"input": user_input, "chat_history": chat_history}):
                if "answer" in chunk:
                    full_res += chunk["answer"]
                    res_box.markdown(full_res + "▌")
            
            res_box.markdown(full_res)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            st.rerun()
            
        except Exception as e:
            st.error(f"Engine Error: {e}")
