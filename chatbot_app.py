__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
from dotenv import load_dotenv

# Disable the broken telemetry
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
                return "✅ Database ready."
            except Exception as e:
                return f"⚠️ Unzip failed: {e}"
        return "⚠️ No database found. Please upload chroma_db.zip."
    return "✅ Database ready."

db_status = prepare_db()

# --- 3. THE AI ENGINE ---
@st.cache_resource
def get_chatbot_chain():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API Key in Streamlit Secrets.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    # Fix: We specify the collection_name to avoid the migration/KeyError bug
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="langchain" # Standard default for LangChain-created DBs
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    
    system_prompt = (
        "You are a helpful assistant for the Eco-Education curriculum. "
        "Use the provided context to answer the user's questions. "
        "If you don't know the answer, say you don't know.\n\n"
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

# --- 4. UI ---
st.title("🌱 Eco-Chatbot")
st.caption(db_status)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    chain = get_chatbot_chain()
    if chain:
        with st.chat_message("assistant"):
            try:
                # Build history
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                
                with st.spinner("Searching knowledge base..."):
                    response = chain.invoke({"input": query, "chat_history": history})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error processing question: {e}")

    st.rerun()
