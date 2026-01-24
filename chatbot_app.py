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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True) 
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Contextualize Question
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a user question, formulate a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    # Answer Question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are the assistant for {DOCUMENT_AUTHOR}. Use the context to answer accurately:\n\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 5. UI LAYOUT & BRANDING ---
st.title(f"🌱 {DOCUMENT_TITLE}")
st.markdown(f"**By {DOCUMENT_AUTHOR}**")

# Use simple dictionaries for state to prevent the '_type' error
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. SUGGESTED PROMPTS ---
suggested_prompts = ["What is Eco-Education?", "Tell me about the garden", "Ann's teaching philosophy"]

# --- 7. DISPLAY HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. INPUT LOGIC (Suggestions + Chat Input) ---
if not st.session_state.messages:
    st.markdown("### Suggested Topics")
    cols = st.columns(len(suggested_prompts))
    for i, prompt in enumerate(suggested_prompts):
        if cols[i].button(prompt):
            st.session_state.pending_input = prompt

user_input = st.chat_input("Ask about nature or education...")

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# --- 9. PROCESSING & STREAMING ---
if user_input:
    st.chat_message("user").markdown(user_input)
    
    # Convert simple state back to LangChain objects for the AI
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
            
            # Save and Refresh
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
