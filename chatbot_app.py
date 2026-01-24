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
    """Extracts the vector database if it hasn't been extracted yet."""
    if not os.path.exists(CHROMA_PATH) and os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
            return "ready"
        except Exception as e:
            return f"Extraction Error: {str(e)}"
    return "ready"

db_status = manage_database()

# --- 3. CONFIGURATION & SECRETS ---
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
    
    # Contextualize the user's question based on chat history
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    # Answer the question using the retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are the expert assistant for {DOCUMENT_AUTHOR}. "
                   "Use the following pieces of retrieved context to answer the user's question. "
                   "If you don't know the answer based on the context, say that you don't know.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 5. UI LAYOUT & BRANDING ---
st.title(f"🌱 {DOCUMENT_TITLE}")
st.markdown(f"### Expert Guidance based on the work of {DOCUMENT_AUTHOR}")

if db_status != "ready":
    st.error(db_status)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. TARGETED SUGGESTED PROMPTS ---
# These are derived from specific themes in the curriculum documents
suggested_prompts = [
    "What are the foundational pillars of the Eco-Education curriculum?",
    "Explain the 'Six Realities of Nature' and their role in learning.",
    "How should educators prepare the environment to foster a child's bond with nature?",
    "Describe the 'documentation' process in this educational approach."
]

# --- 7. DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. INPUT HANDLING ---
# Display suggestion buttons if the conversation hasn't started
if not st.session_state.messages:
    st.info("Select a topic below or type your own question to begin.")
    cols = st.columns(2)
    for i, prompt in enumerate(suggested_prompts):
        if cols[i % 2].button(prompt, use_container_width=True):
            st.session_state.pending_input = prompt

user_input = st.chat_input("Ask a question about the curriculum...")

# Handle input from suggestion buttons
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# --- 9. CORE EXECUTION ---
if user_input:
    # 1. Display user message
    st.chat_message("user").markdown(user_input)
    
    # 2. Build LangChain history objects
    chat_history = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        try:
            rag_chain = setup_rag_chain()
            response_placeholder = st.empty()
            full_response = ""
            
            # 3. Stream the response
            for chunk in rag_chain.stream({"input": user_input, "chat_history": chat_history}):
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # 4. Save to session state and refresh
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
