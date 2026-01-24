__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
import random
from dotenv import load_dotenv

# Essential LangChain Imports - Updated for 0.3.x compatibility
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. THE UNZIPPER ---
CHROMA_PATH = "chroma_db"
ZIP_PATH = "chroma_db.zip"

if not os.path.exists(CHROMA_PATH) and os.path.exists(ZIP_PATH):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.toast("Brain unzipped and ready! 🧠")
    except Exception as e:
        st.error(f"Unzip failed: {e}")

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

# --- 3. SAFETY & PROMPTS ---
OFF_LIMITS = ["sex", "penis", "vagina", "sexual", "porn", "intercourse", "genitals"]
REFUSAL_MESSAGE = "I am a specialized assistant for the Eco-Education curriculum. I don't provide information on that topic, but I can help you with questions about nature or Ann Lewin-Benham's methods."

def is_strictly_inappropriate(query: str) -> bool:
    query_clean = query.lower().replace("?", "").replace(".", "").split()
    return any(word in OFF_LIMITS for word in query_clean)

SYSTEM_TEMPLATE = """
You are the **Eco-Education AI Assistant**, an expert on the work of **{DOCUMENT_AUTHOR}**.
Answer questions about the author, her books, and teaching philosophy using the context.
If inappropriate topics are asked, use: "{REFUSAL_MESSAGE}"
--- CONTEXT ---
{context}
--- QUESTION ---
{input}
"""

# --- 4. RAG SETUP ---
@st.cache_resource
def setup_rag_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, streaming=True) 
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question based on history. Do NOT answer it."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]).partial(DOCUMENT_AUTHOR=DOCUMENT_AUTHOR, REFUSAL_MESSAGE=REFUSAL_MESSAGE)
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 5. UI LOGIC ---
st.set_page_config(page_title="Eco-Education Assistant", layout="wide")

if HISTORY_KEY not in st.session_state: 
    st.session_state[HISTORY_KEY] = []

st.title(f"🌱 {DOCUMENT_TITLE}")
st.markdown(f"**By {DOCUMENT_AUTHOR}**")

# Display Chat
for msg in st.session_state[HISTORY_KEY]:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").markdown(msg.content)

# Input
if user_input := st.chat_input("Ask about nature or education..."):
    st.session_state[HISTORY_KEY].append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
        
    with st.chat_message("assistant"):
        if is_strictly_inappropriate(user_input):
            st.markdown(REFUSAL_MESSAGE)
            st.session_state[HISTORY_KEY].append(AIMessage(content=REFUSAL_MESSAGE))
        else:
            try:
                rag_chain = setup_rag_chain()
                full_res = st.write_stream(
                    chunk["answer"] for chunk in rag_chain.stream({
                        "chat_history": st.session_state[HISTORY_KEY], 
                        "input": user_input
                    }) if "answer" in chunk
                )
                st.session_state[HISTORY_KEY].append(AIMessage(content=full_res))
            except Exception as e:
                st.error(f"Error: {e}")
