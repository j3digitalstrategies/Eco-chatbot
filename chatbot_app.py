import streamlit as st
import os
import glob
from dotenv import load_dotenv

# Core LangChain & AI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIG & SYSTEM PROMPT ---
load_dotenv()
DOCS_DIR = "curriculum_docs"
VECTOR_DB_DIR = "vector_db"
st.set_page_config(page_title="Eco-Assistant", layout="wide", page_icon="🌱")

# Safeguards and Persona
SYSTEM_BEHAVIOR = """
You are the Eco-Education Curriculum Assistant. Your "brain" consists of the overarching 
Eco-Education curriculum written by Ann Lewin-Benham.

CORE RULES:
1. Treat all documents as a single unified curriculum by Ann Lewin-Benham.
2. If the answer is in the curriculum, prioritize that information.
3. If the answer is NOT in the curriculum, use your general knowledge but mention it's supplementary.
4. SAFEGUARD: Do not provide instructions on harmful activities, illegal acts, or topics 
   that contradict environmental wellness.
5. If a user is confused, guide them back to the big ideas of the curriculum.
"""

# --- 2. THE ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
    
    # Check if DB exists to avoid re-reading every time
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    else:
        if not os.path.exists(DOCS_DIR):
            st.error("Missing curriculum_docs folder.")
            return None

        all_files = []
        for ext in [".docx", ".pdf", ".txt"]:
            all_files.extend(glob.glob(os.path.join(DOCS_DIR, f"**/*{ext}"), recursive=True))

        documents = []
        for file_path in all_files:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".docx": loader = Docx2txtLoader(file_path)
                elif ext == ".pdf": loader = PyPDFLoader(file_path)
                elif ext == ".txt": loader = TextLoader(file_path)
                documents.extend(loader.load())
            except: continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECTOR_DB_DIR)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) 
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_BEHAVIOR + "\n\nContext from curriculum:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = (
        {
            "context": (lambda x: x["input"]) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt | llm | StrOutputParser()
    )
    return rag_chain

# --- 3. UI ---
st.title("🌱 Eco-Education Assistant")
st.subheader("by Ann Lewin-Benham")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key missing.")
    st.stop()

# Initialize Engine
chain = get_bot_chain(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SUGGESTED PROMPTS ---
st.sidebar.title("Suggested Topics")
suggestions = [
    "What are the big ideas of this curriculum?",
    "Tell me about the Greenhouse Effect in Chapter 12.",
    "How does the curriculum approach evolution?",
    "What is the 'Romance of Geology'?"
]

for suggestion in suggestions:
    if st.sidebar.button(suggestion):
        st.session_state.messages.append({"role": "user", "content": suggestion})
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
        with st.chat_message("assistant"):
            # Ensure the chain is ready before invoking
            if chain:
                response = chain.invoke({"input": suggestion, "chat_history": history})
                st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Display Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_input = st.chat_input("Ask a question about the curriculum...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    
    history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if chain:
                response = chain.invoke({"input": user_input, "chat_history": history})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
