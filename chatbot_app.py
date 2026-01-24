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

# --- 2. THE UNZIPPER ---
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
            return str(e)
    return "ready"

manage_database()

# --- 3. CONFIGURATION ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()

DOCUMENT_AUTHOR = "Ann Lewin-Benham" 
DOCUMENT_TITLE = "Eco-Education for Young Children" 

# --- 4. RAG ENGINE SETUP ---
@st.cache_resource
def setup_rag_chain():
    # Using the latest embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Initialize Vector Store
    vector_store = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True) 
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 1. Contextualize Question (Fixes the _type issue by being explicit)
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a user question, formulate a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    # 2. Answer Question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are the assistant for {DOCUMENT_AUTHOR}. Answer using the context below:\n\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 5. UI LAYOUT ---
st.title(f"🌱 {DOCUMENT_TITLE}")
st.markdown(f"**By {DOCUMENT_AUTHOR}**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 6. CHAT LOGIC ---
for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        try:
            rag_chain = setup_rag_chain()
            
            # Use explicit input mapping to satisfy LangChain's internal 'type' checks
            stream = rag_chain.stream({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            
            full_res = st.write_stream(chunk["answer"] for chunk in stream if "answer" in chunk)
            st.session_state.chat_history.append(AIMessage(content=full_res))
        except Exception as e:
            st.error(f"Error: {e}")
            if "type" in str(e).lower():
                st.info("Tip: This is usually a database version mismatch. Try refreshing the app or re-uploading your chroma_db.zip.")
