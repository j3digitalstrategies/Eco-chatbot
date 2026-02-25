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

st.markdown("""
    <style>
    .stButton button { display: block; margin: 0 auto; padding: 8px 30px; border-radius: 15px; background-color: #2e7d32; color: white; border: none; }
    .welcome-text { text-align: center; font-size: 1.2em; color: #1b5e20; margin-bottom: 30px; line-height: 1.6; max-width: 900px; margin-left: auto; margin-right: auto; display: block; }
    .main-header { text-align: center; color: #2e7d32; margin-bottom: 5px; }
    .sub-header { text-align: center; color: #558b2f; font-style: italic; margin-bottom: 30px; }
    u { text-decoration: underline; color: #2e7d32; font-weight: bold; }
    .badge-card { background-color: #e8f5e9; border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; font-size: 25px; margin: 10px; border: 2px solid #2e7d32; }
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
        for f in all_files:
            try:
                ext = os.path.splitext(f)[1].lower()
                if ext == ".docx": loader = Docx2txtLoader(f)
                elif ext == ".pdf": loader = PyPDFLoader(f)
                elif ext == ".txt": loader = TextLoader(f)
                documents.extend(loader.load())
            except: continue
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECTOR_DB_DIR)
    return vectorstore.as_retriever(search_kwargs={"k": 5}), ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=_api_key)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "onboarded" not in st.session_state: st.session_state.onboarded = False
if "profile" not in st.session_state: 
    st.session_state.profile = {"zip": None, "role": "Other", "age": 35, "child_age": None}
if "suggestions" not in st.session_state: st.session_state.suggestions = []
if "power_words" not in st.session_state: st.session_state.power_words = {}

# --- 4. ENGINE INIT ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 5. ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 class='main-header'>Saving Planet Earth: Eco-Education Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Based on the work of Ann Lewin-Benham</h3>", unsafe_allow_html=True)
    
    with st.container():
        u_role = st.selectbox("I am a...", ["Student", "Parent", "Teacher", "Other"], index=1)
        c1, c2 = st.columns(2)
        with c1: z_code = st.text_input("Zip Code", placeholder="e.g. 91231")
        with c2:
            if u_role == "Student": u_age = st.number_input("Your Age", 3, 18, 10); c_age = None
            elif u_role == "Parent": u_age = 35; c_age = st.number_input("Child's Age", 1, 18, 5)
            else: u_age = 35; c_age = None
        
        if st.button("Start Exploring"):
            if z_code:
                st.session_state.profile.update({"zip": z_code, "role": u_role, "age": u_age, "child_age": c_age})
                defaults = {
                    "Parent": ["How to start an observation?", "What is Meaning-FULL?", "Fostering biophilia?"],
                    "Teacher": ["How to implement this in class?", "Documentation tips?", "Explain core pillars."],
                    "Student": ["What can I explore today?", "Cool nature fact?", "How to start a journal?"],
                    "Other": ["About the curriculum?", "Ann Lewin-Benham?", "What is Eco-Education?"]
                }
                st.session_state.suggestions = defaults.get(u_role, defaults["Other"])
                st.session_state.onboarded = True
                st.rerun()
            else: st.warning("Please enter a Zip Code!")
    st.stop()

# --- 6. BEHAVIOR ---
p = st.session_state.profile
is_student = p['role'] == "Student"
target_age = p['age'] if is_student else p['child_age']

SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for Ann Lewin-Benham's Eco-Education.
CONTEXT: Role={p['role']}, User Age={p['age']}, Child Age={p['child_age']}, ZIP={p['zip']}.

STRICT RULES:
1. ADAPTIVE: Tailor language and advice to a {target_age} year old's comprehension level.
2. UNDERLINING: Wrap exactly 1-2 'Power Words' in <u>word</u> tags.
3. DISCRETION: ONLY underline words that are "stretch" vocabulary for the user. 
   - For an Adult: ONLY underline technical curriculum pedagogy (e.g., <u>Biophilia</u>). 
   - For a Student ({p['age']}yr): Underline scientific concepts they might not fully know yet.
   - NEVER underline common, everyday words (e.g., moss, garden, green, child, observation). If the word is part of basic conversation for their age, do NOT underline it.
4. NO BOT QUESTIONS: Do not end with a question.
"""

# --- 7. SIDEBAR ---
with st.sidebar:
    st.title("Suggested Prompts")
    for s in st.session_state.suggestions:
        if st.button(s, use_container_width=True): 
            st.session_state.user_query = s
            st.rerun()

    st.divider()
    if is_student:
        st.subheader("🏅 My Observation Badges")
        badges = {"🌱": "Sprout Spotter", "🐦": "Bird Watcher", "☁️": "Cloud Reader", "🐞": "Bug Explorer", "🪨": "Rock Scientist"}
        cols = st.columns(3)
        for i, (emoji, name) in enumerate(badges.items()):
            cols[i % 3].markdown(f"<div class='badge-card' title='{name}'>{emoji}</div>", unsafe_allow_html=True)
    
    if st.session_state.power_words:
        st.subheader("📚 Power Words")
        for word, defn in st.session_state.power_words.items(): 
            st.markdown(f"**{word}**: {defn}")
    
    st.divider()
    if st.button("🔄 Reset Profile"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# --- 8. ENGINE & DISPLAY ---
prompt_template = ChatPromptTemplate.from_messages([("system", SYSTEM_BEHAVIOR + "\n\nContext:\n{context}"), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
rag_chain = ({"context": (lambda x: x["input"]) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]} | prompt_template | llm_model | StrOutputParser())

for m in st.session_state.messages: st.chat_message(m["role"]).markdown(m["content"], unsafe_allow_html=True)
if not st.session_state.messages:
    intro = f"Ready to explore as a **{p['role']}**. What's your focus today?"
    st.chat_message("assistant").markdown(intro); st.session_state.messages.append({"role": "assistant", "content": intro})

query = st.session_state.get("user_query") or st.chat_input("Type here...")
if query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    
    hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        res = rag_chain.invoke({"input": query, "chat_history": hist})
        st.markdown(res, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": res})
        
        found_underlines = re.findall(r'<u>(.*?)</u>', res)
        
        try:
            target_desc = f"student who is {p['age']} years old" if is_student else f"parent of a {p['child_age']}yr old"
            update_p = f"""
            Identify 3 short questions the {target_desc} would logically ask the bot next.
            Then, look at this list of underlined words: {found_underlines}.
            
            CRITICAL FILTER: 
            Is the word common knowledge for a {target_desc}? 
            If yes (e.g., 'moss', 'garden', 'low-maintenance'), DISCARD IT.
            Only provide definitions for 'Stretch' or 'Power' words.
            
            Return JSON: {{"prompts": [], "vocab": {{}}}}
            """
            u_res = llm_model.invoke([("system", update_p), ("human", res)])
            data = json.loads(u_res.content)
            
            st.session_state.suggestions = data.get("prompts", [])
            st.session_state.power_words.update(data.get("vocab", {}))
        except: pass
    st.rerun()
