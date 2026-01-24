__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
from dotenv import load_dotenv

# Essential LangChain Imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. DATABASE UNZIPPER ---
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

if not os.path.exists(CHROMA_PATH) and os.path.exists(ZIP_PATH):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("Database extracted successfully!")
    except Exception as e:
        st.error(f"Error unzipping database: {e}")

# --- 2. CONFIGURATION ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
HISTORY_KEY = "chat_history_list_lc"

DOCUMENT_AUTHOR = "Ann Lewin-Benham"
DOCUMENT_TITLE = "Eco-Education for Young Children"

# --- 3. SYSTEM PROMPT ---
SYSTEM_TEMPLATE = """
You are the Eco-Education AI Assistant, an expert on the work of {DOCUMENT_AUTHOR}.
Use the provided context to answer the user's question accurately.
If the answer is not in the context, say you don't know, but offer to help with other education topics.

--- CONTEXT ---
{context}

--- QUESTION ---
{input}
"""

# --- 4. RAG ENGINE ---
@st.cache_resource
def setup_rag_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, streaming=True)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Contextualize question (History + New Question -> Standalone Question)
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    # Final QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]).partial(DOCUMENT_AUTHOR=DOCUMENT_AUTHOR)
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Eco-Chatbot", page_icon="🌱")

if HISTORY_KEY not in st.session_state:
    st.session_state[HISTORY_KEY] = []

st.title(f"🌱 {DOCUMENT_TITLE}")
st.caption(f"Knowledge Base: {DOCUMENT_AUTHOR}")

# Display history
for msg in st.session_state[HISTORY_KEY]:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# User Input
if user_input := st.chat_input("How can I help you today?"):
    st.session_state[HISTORY_KEY].append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        try:
            rag_chain = setup_rag_chain()
            
            # Stream the response
            response_container = st.empty()
            full_answer = ""
            
            for chunk in rag_chain.stream({
                "chat_history": st.session_state[HISTORY_KEY],
                "input": user_input
            }):
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                    response_container.markdown(full_answer + "▌")
            
            response_container.markdown(full_answer)
            st.session_state[HISTORY_KEY].append(AIMessage(content=full_answer))
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
