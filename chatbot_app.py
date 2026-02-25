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

# --- 1. CONFIG ---
load_dotenv()
DOCS_DIR = "curriculum_docs"
VECTOR_DB_DIR = "vector_db"
st.set_page_config(page_title="Eco-Education Assistant", layout="wide", page_icon="🌱")

# Custom CSS for Boutique UI
st.markdown("""
    <style>
    .stButton button {
        display: block;
        margin: 0 auto;
        padding: 10px 40px;
        border-radius: 25px;
        background-color: #2e7d32;
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #1b5e20;
        color: #e8f5e9;
    }
    .onboarding-card {
        background-color: #f1f8e9;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #c8e6c9;
        margin-bottom: 20px;
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
    st.session_state.profile = {"zip": None, "role": "Other", "age": 10}
if "power_words" not in st.session_state: st.session_state.power_words = {}

# --- 4. API CHECK ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 5. CLEAN ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 style='text-align: center;'>Eco-Education Assistant</h1>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class='onboarding-card'>
            <p style='font-size: 1.1em; color: #33691e;'><b>Tell us a little bit more about yourself.</b><br>
            This helps me understand your local environment and how you want to use this mentor.</p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            z_code = st.text_input("Zip Code", placeholder="e.g. 91231")
        with c2:
            u_role = st.selectbox("I am a...", ["Parent", "Teacher", "Student", "Other"])
        with c3:
            default_age = 10 if u_role == "Student" else 35
            u_age = st.number_input("Your Age", min_value=3, max_value=100, value=default_age)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Exploring"):
            if z_code:
                st.session_state.profile.update({"zip": z_code, "role": u_role, "age": u_age})
                st.session_state.onboarded = True
                st.rerun()
            else:
                st.error("Please enter a Zip Code to personalize your experience.")
    st.stop()

# --- 6. BEHAVIOR ---
p = st.session_state.profile
SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for Ann Lewin-Benham's Eco-Education.
CONTEXT: Role={p['role']}, UserAge={p['age']}, ZIP={p['zip']}.

STRICT RULES:
1. BREVITY: Max 2 short paragraphs.
2. LOCAL TRUTH: Use ZIP {p['zip']} to know what wildlife/climate is present. Never repeat the ZIP number.
3. ADAPTIVE LANGUAGE: 
   - If age {p['age']} is < 10: Use very simple words, high metaphors.
   - If age {p['age']} is 10-15: Use clear scientific terms but avoid jargon.
   - If age {p['age']} is > 15/Adult: Use professional pedagogical/scientific language.
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

# --- 9. DISPLAY & INPUT ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if not st.session_state.messages:
    intro = f"Ready to explore as a **{p['role']}**. What's your focus today?"
    with st.chat_message("assistant"): st.markdown(intro)
    st.session_state.messages.append({"role": "assistant", "content": intro})

query = st.chat_input("Type here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        full_res = rag_chain.invoke({"input": query, "chat_history": history})
        st.markdown(full_res)
        st.session_state.messages.append({"role": "assistant", "content": full_res})
        
        # DYNAMIC AGE-BASED POWER WORDS
        try:
            update_p = f"""
            Analyze this response: '{full_res}'. 
            The user is {p['age']} years old.
            Select 1-2 'Power Words' that are "stretch" words for a {p['age']} year old.
            
            CRITICAL SELECTION CRITERIA:
            - Age < 8: Define basic science words (e.g. 'Habitat').
            - Age 8-12: Define intermediate words (e.g. 'Phenomenon', 'Ecosystem').
            - Age 13-17: Define advanced terms (e.g. 'Biodiversity', 'Biophilia').
            - Age 18+: Define pedagogical/theoretical terms (e.g. 'Socratic', 'Emergent').
            - NEVER define common words (e.g. 'Role', 'Active', 'Discuss', 'Plants').
            
            Return JSON: {{"vocab": {{"word": "simple definition"}}}}
            """
            u_res = llm_model.invoke([("system", update_p), ("human", query)])
            data = json.loads(u_res.content)
            st.session_state.power_words.update(data.get("vocab", {}))
        except: pass
    st.rerun()
