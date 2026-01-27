import streamlit as st
import os
import glob
import json
import time
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

# --- 1. CONFIG & SYSTEM PROMPTS ---
load_dotenv()
DOCS_DIR = "curriculum_docs"
VECTOR_DB_DIR = "vector_db"
st.set_page_config(page_title="Saving Planet Earth", layout="wide", page_icon="🌱")

SYSTEM_BEHAVIOR = """
You are the Eco-Education Curriculum Assistant for "Saving Planet Earth: Revolutionary Ways to Teach Eco-Education."
You represent the pedagogical work of Ann Lewin-Benham.

STRICT CLASSROOM PROTOCOL:
1. SAFEGUARD: You must never discuss content that is sexually explicit, violent, or sensitive for young children. 
2. REFUSAL: If asked about adult/sensitive topics, refuse politely and pivot back to the curriculum.
3. SCOPE: Focus on the curriculum. If asked general child-safe questions, answer them through the lens of Ann Lewin-Benham's methods.
"""

SUGGESTION_PROMPT = """
Based on the discussion of Ann Lewin-Benham's Eco-Education curriculum, generate 3 child-safe "Suggested Prompts".
Return ONLY a JSON list of strings. Format: ["Prompt 1", "Prompt 2", "Prompt 3"]
"""

INTRO_TEXT = """Welcome! I am your AI assistant for **Saving Planet Earth: Revolutionary Ways to Teach Eco-Education**. 

I am here to help you navigate Ann Lewin-Benham’s curriculum. You can ask me about specific chapters, teaching strategies for young learners, or how to foster meaningful conversations about our planet. 

Try selecting a topic from the **Suggested Prompts** on the left, or type your own question below to begin."""

# --- 2. THE ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
    
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    else:
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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
    
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
    return rag_chain, llm

# --- 3. UI ---
st.title("Saving Planet Earth: Revolutionary Ways to Teach Eco-Education Chatbot")
st.subheader("by Ann Lewin-Benham")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("API Key missing.")
    st.stop()

# Initialize Engine
chain, llm_model = get_bot_chain(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Trigger the intro sequence on first load
    st.session_state.show_intro = True

if "suggestions" not in st.session_state:
    st.session_state.suggestions = ["What are the big ideas of this curriculum?", "How do we teach about ecosystems?", "Tell me about Ann Lewin-Benham's philosophy."]

# --- SIDEBAR: SUGGESTED PROMPTS ---
st.sidebar.title("Suggested Prompts")
for suggestion in st.session_state.suggestions:
    if st.sidebar.button(suggestion):
        st.session_state.user_query = suggestion

# --- DISPLAY CHAT ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# --- INTRO LOGIC ---
if st.session_state.get("show_intro"):
    with st.chat_message("assistant"):
        def stream_intro():
            for word in INTRO_TEXT.split(" "):
                yield word + " "
                time.sleep(0.03)
        intro_response = st.write_stream(stream_intro())
        st.session_state.messages.append({"role": "assistant", "content": intro_response})
    st.session_state.show_intro = False

# --- USER INPUT ---
user_input = st.chat_input("Ask a question about the curriculum...")
final_query = st.session_state.get("user_query") or user_input

if final_query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): st.markdown(final_query)
    
    history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
    
    with st.chat_message("assistant"):
        def stream_response():
            full_response = chain.invoke({"input": final_query, "chat_history": history})
            for word in full_response.split(" "):
                yield word + " "
                time.sleep(0.04)

        response_text = st.write_stream(stream_response())
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Adaptive suggestions update
        try:
            chat_context = f"User: {final_query}\nAssistant: {response_text}"
            res = llm_model.invoke([("system", SUGGESTION_PROMPT), ("human", chat_context)])
            st.session_state.suggestions = json.loads(res.content)
        except: pass
            
    st.rerun()
