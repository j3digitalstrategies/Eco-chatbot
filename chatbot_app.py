__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import logging
from dotenv import load_dotenv

# Configure logging to help debug in the Streamlit logs
logging.basicConfig(level=logging.INFO)

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Ann Lewin-Benham AI", layout="wide")

# LangChain Imports
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as e:
    st.error(f"Missing dependency: {e}. Please check your requirements.txt.")
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
                return "Database extracted successfully."
            except Exception as e:
                return f"Error extracting database: {str(e)}"
        else:
            return "Database folder and zip file are both missing."
    return "Database ready."

db_msg = manage_database()

# --- 3. API KEY VERIFICATION ---
def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("🔑 API Key Missing! Go to Streamlit Cloud > Settings > Secrets and add: OPENAI_API_KEY='your-key-here'")
    st.stop()

# --- 4. RAG ENGINE SETUP ---
@st.cache_resource
def setup_rag_chain():
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=api_key)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # System instructions
        system_prompt = (
            "You are an expert on the work of Ann Lewin-Benham. "
            "Use the provided context to answer questions about Eco-Education. "
            "If the answer isn't in the context, say you don't know based on the documents.\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        st.error(f"Failed to initialize AI Engine: {e}")
        return None

# --- 5. UI ELEMENTS ---
st.title("🌱 Ann Lewin-Benham AI")
st.caption(f"Status: {db_msg}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. CHAT INPUT ---
user_input = st.chat_input("Type your question here...")

if user_input:
    # Immediately show the user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process with AI
    with st.chat_message("assistant"):
        try:
            rag_chain = setup_rag_chain()
            if rag_chain:
                # Convert history for the chain
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                
                # Use a simpler response generation to avoid timeouts
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke({"input": user_input, "chat_history": history})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Chat Error: {e}")

    # Force refresh to update chat history
    st.rerun()
