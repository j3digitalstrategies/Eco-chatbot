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
    u {
        text-decoration: underline;
        color: #2e7d32;
        font-weight: bold;
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

# --- 3. ROLE DEFAULTS ---
DEFAULT_PROMPTS = {
    "Parent": ["How do I start an observation?", "What is Meaning-FULL conversation?", "How to foster biophilia?"],
    "Teacher": ["Classroom implementation?", "Documentation strategies?", "What are the core pillars?"],
    "Student": ["What can I explore today?", "Tell me a cool nature fact.", "How do I start a nature journal?"],
    "Other": ["Tell me about the curriculum.", "Who is Ann Lewin-Benham?", "What is Eco-Education?"]
}

# --- 4. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "onboarded" not in st.session_state: st.session_state.onboarded = False
if "profile" not in st.session_state: 
    st.session_state.profile = {"zip": None, "role": "Other", "age": 35}
if "suggestions" not in st.session_state: st.session_state.suggestions = DEFAULT_PROMPTS["Other"]
if "power_words" not in st.session_state: st.session_state.power_words = {}

# --- 5. API & ENGINE INIT ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 6. UNIFIED ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 class='main-header'>Saving Planet Earth: Eco-Education Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Based on the work of Ann Lewin-Benham</h3>", unsafe_allow_html=True)
    st.markdown("<div class='welcome-text'>To help us understand the nature right outside your door and tailor the conversation to your needs, please share your location and role below.</div>", unsafe_allow_html=True)
    
    with st.container():
        u_role = st.selectbox("I am a...", ["Student", "Parent", "Teacher", "Other"], index=1)
        c1, c2 = st.columns(2)
        with c1:
            z_code = st.text_input("Zip Code", placeholder="e.g. 91231")
        with c2:
            if u_role == "Student":
                u_age = st.number_input("Age", min_value=3, max_value=100, value=10)
            else:
                u_age = 35 
                st.write("") 
        
        st.write("") 
        if st.button("Start Exploring"):
            if z_code:
                st.session_state.profile.update({"zip": z_code, "role": u_role, "age": u_age})
                st.session_state.suggestions = DEFAULT_PROMPTS[u_role]
                st.session_state.onboarded = True
                st.rerun()
            else:
                st.warning("Please enter a Zip Code so we can know your environment!")
    st.stop()

# --- 7. BEHAVIOR ---
p = st.session_state.profile
SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for Ann Lewin-Benham's Eco-Education.
CONTEXT: Role={p['role']}, Age={p['age']}, ZIP={p['zip']}.

STRICT RULES:
1. BREVITY: Max 2 short paragraphs.
2. LOCAL TRUTH: Use ZIP {p['zip']} to inform answers about wildlife/climate silently.
3. ADAPTIVE LANGUAGE: 
   - Adults (Parent/Teacher): Professional and pedagogical. Do NOT define common words.
   - Students: Age-appropriate metaphors or scientific terms.
4. UNDERLINING & POWER WORDS: 
   - You MUST wrap 1-2 key pedagogical or scientific terms per response in <u>word</u> tags.
   - ONLY underline words you are currently explaining or using in the text.
   - For adults, ONLY underline curriculum-specific terms (e.g., <u>Meaning-FULL</u>).
"""

# --- 8. SIDEBAR ---
with st.sidebar:
    st.title("Suggested Prompts")
    for s in st.session_state.suggestions:
        if st.button(s, use_container_width=True): 
            st.session_state.user_query = s
            st.rerun()

    if st.session_state.power_words:
        st.divider()
        st.subheader("📚 Power Words")
        # Sort words so they appear alphabetically or by most recent
        for word, defn in st.session_state.power_words.items():
            st.markdown(f"**{word}**: {defn}")
    
    st.divider()
    if st.button("🔄 Reset Profile"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# --- 9. CHAT ENGINE ---
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

# --- 10. DISPLAY ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"], unsafe_allow_html=True)

if not st.session_state.messages:
    intro = f"Ready to explore as a **{p['role']}**. What's your focus today?"
    with st.chat_message("assistant"): st.markdown(intro)
    st.session_state.messages.append({"role": "assistant", "content": intro})

# --- 11. INPUT & DYNAMIC LOGIC ---
user_input = st.chat_input("Type here...")
query = st.session_state.get("user_query") or user_input

if query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        full_res = rag_chain.invoke({"input": query, "chat_history": history})
        st.markdown(full_res, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_res})
        
        # 1. FIND UNDERLINED WORDS IN THE ACTUAL TEXT
        found_underlines = re.findall(r'<u>(.*?)</u>', full_res)
        
        try:
            update_p = f"""
            Task:
            1. Suggest 3 short user follow-up questions for a {p['role']}.
            2. Provide curriculum-specific definitions ONLY for these words found in the text: {found_underlines}.
            
            Return ONLY JSON: {{"prompts": [], "vocab": {{"word": "definition"}}}}
            """
            u_res = llm_model.invoke([("system", update_p), ("human", full_res)])
            data = json.loads(u_res.content)
            
            # 2. STRICT FILTER: Only add word to session state if it was in 'found_underlines'
            # This prevents "hidden" words or words from suggested prompts from entering the list.
            valid_vocab = {k: v for k, v in data.get("vocab", {}).items() if k in found_underlines}
            
            st.session_state.suggestions = data.get("prompts", [])
            st.session_state.power_words.update(valid_vocab)
        except: pass
    st.rerun()
