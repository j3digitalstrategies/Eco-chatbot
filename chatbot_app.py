__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
from dotenv import load_dotenv

# Essential LangChain Imports - Updated for 0.3.x compatibility
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. THE DATABASE UNZIPPER ---
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

if not os.path.exists(CHROMA_PATH) and os.path.exists(ZIP_PATH):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
    except Exception as e:
        st.error(f"Error unzipping database: {e}")

# --- 2. CONFIGURATION ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
HISTORY_KEY = "chat_history"

# --- 3. RAG ENGINE ---
@st.cache_resource
def setup_rag_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Contextualize question logic
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    # Final QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Use the context to answer the question.\n\nContext:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Eco-Chatbot", page_icon="🌱")
st.title("🌱 Eco-Education Chatbot")

if HISTORY_KEY not in st.session_state:
    st.session_state[HISTORY_KEY] = []

# Display Chat
for msg in st.session_state[HISTORY_KEY]:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# Input
if user_input := st.chat_input("Ask about eco-education..."):
    st.session_state[HISTORY_KEY].append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        try:
            rag_chain = setup_rag_chain()
            response = rag_chain.invoke({
                "chat_history": st.session_state[HISTORY_KEY],
                "input": user_input
            })
            answer = response["answer"]
            st.markdown(answer)
            st.session_state[HISTORY_KEY].append(AIMessage(content=answer))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
