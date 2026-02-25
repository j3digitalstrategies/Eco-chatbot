import streamlit as st
import os, glob, json, time, re
from datetime import datetime
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

# --- 1. CONFIG & STYLING ---
load_dotenv()
DOCS_DIR = "curriculum_docs"
VECTOR_DB_DIR = "vector_db"
st.set_page_config(page_title="Saving Planet Earth", layout="wide", page_icon="🌱")

# Boutique UI Styling
st.markdown("""
    <style>
    .stButton button {
        display: block;
        margin: 0 auto;
        padding: 8px 30px;
        border-radius: 15px;
        background-color: #2e7d32;
        color: white;
        border: none;
    }
    .welcome-text {
        text-align: center;
        font-size: 1.2em;
        color: #1b5e20;
        margin-bottom: 30px;
        line-height: 1.6;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        display: block;
    }
    .main-header {
        text-align: center;
        color: #2e7d32;
        margin-bottom: 5px;
    }
    .sub-header {
        text-align: center;
        color: #558b2f;
        font-style: italic;
        margin-bottom: 30px;
    }
    /* Ensure all markdown containers in the main area center their text if they are welcome-text */
    div[data-testid="stMarkdownContainer"] > p.welcome-text {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=_api_key)
    return retriever, llm

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "onboarded" not in st.session_state: st.session_state.onboarded = False
if "profile" not in st.session_state: 
    st.session_state.profile = {"zip": None, "role": "Other", "age": 35}
if "power_words" not in st.session_state: st.session_state.power_words = {}

# --- 4. API & ENGINE INIT ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 5. UNIFIED ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 class='main-header'>Saving Planet Earth: Eco-Education Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Based on the work of Ann Lewin-Benham</h3>", unsafe_allow_html=True)
    
    # The centered "Us" preamble
    st.markdown("<div class='welcome-text'>To help us understand the nature right outside your door and tailor the conversation to your needs, please share your location and role below.</div>", unsafe_allow_html=True)
    
    with st.container():
        # Single row layout restored for better looks
        u_role = st.selectbox("I am a...", ["Student", "Parent", "Teacher", "Other"], index=1)
        
        c1, c2 = st.columns(2)
        with c1:
            z_code = st.text_input("Zip Code", placeholder="e.g. 91231")
        with c2:
            if u_role == "Student":
                u_age = st.number_input("Age", min_value=3, max_value=100, value=10)
            else:
                u_age = 35 # Hidden default for adults
                st.write("") # Spacer
        
        st.write("") 
        if st.button("Start Exploring"):
            if z_code:
                st.session_state.profile.update({"zip": z_code, "role": u_role, "age": u_age})
                st.session_state.onboarded = True
                st.rerun()
            else:
                st.warning("Please enter a Zip Code so we can know your environment!")
    st.stop()

# --- 6. BEHAVIOR ---
p = st.session_state.profile
SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for Ann Lewin-Benham's Eco-Education.
CONTEXT: Role={p['role']}, Age={p['age']}, ZIP={p['zip']}.

STRICT RULES:
1. BREVITY: Max 2 short paragraphs.
2. LOCAL TRUTH: Use ZIP {p['zip']} to inform answers about wildlife/climate silently.
3. ADAPTIVE LANGUAGE: 
   - Age < 10: Simple metaphors. 
   - Age 10-15: Clear scientific terms. 
   - Adult: Professional/Pedagogical based on Ann Lewin-Benham's curriculum.
4. POWER WORDS: Only extract 'stretch' words relative to age {p['age']}. No common words.
"""

# --- 7. SIDEBAR ---
with st.sidebar:
    st.title("🌱 Mentor Sidebar")
    if st.session_state.power_words:
        st.subheader("📚 Power Words")
        for word, defn in st.session_state.power_words.items():
            st.markdown(f"**{word.capitalize()}**: {defn}")
    
    st.divider()
    if st.button("🔄 Reset Profile"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# --- 8. CHAT ENGINE ---
# (RAG and LLM logic remains unchanged)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BEHAVIOR + "\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

rag_chain = (
    {"context": (lambda x: x["input"]) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]}
    | prompt_template | llm_model | StrOutputParser()
)

# --- 9. DISPLAY ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if not st.session_state.messages:
    intro = f"Ready to explore as a **{p['role']}**. What's your focus today?"
    with st.chat_message("assistant"): st.markdown(intro)
    st.session_state.messages.append({"role": "assistant", "content": intro})

# --- 10. INPUT & DYNAMIC LOGIC ---
query = st.chat_input("Type here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        full_res = rag_chain.invoke({"input": query, "chat_history": history})
        st.markdown(full_res)
        st.session_state.messages.append({"role": "assistant", "content": full_res})
        
        try:
            update_p = f"""
            Analyze: '{full_res}'. Role: {p['role']}, Age: {p['age']}.
            Select 1-2 'Power Words'. 
            CRITICAL: 
            - If Adult, extract pedagogical/curriculum theory terms.
            - If Student, extract 'stretch' scientific terms for age {p['age']}.
            Return JSON: {{"vocab": {{"word": "definition"}}}}
            """
            u_res = llm_model.invoke([("system", update_p), ("human", query)])
            data = json.loads(u_res.content)
            st.session_state.power_words.update(data.get("vocab", {}))
        except: pass
    st.rerun()
