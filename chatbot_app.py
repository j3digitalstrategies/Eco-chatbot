import streamlit as st
import os, glob, json, re
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
    u { text-decoration: underline; color: #2e7d32; font-weight: bold; }
    .calendar-box { background-color: #f1f8e9; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    else:
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_DIR)
    return vectorstore.as_retriever(search_kwargs={"k": 5}), ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=_api_key)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "onboarded" not in st.session_state: st.session_state.onboarded = False
if "profile" not in st.session_state: st.session_state.profile = {"zip": None, "role": "Other", "age": 35, "child_age": None}
if "suggestions" not in st.session_state: st.session_state.suggestions = []
if "power_words" not in st.session_state: st.session_state.power_words = {}
if "local_calendar" not in st.session_state: st.session_state.local_calendar = ""

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 4. ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 style='text-align:center;'>Saving Planet Earth</h1>", unsafe_allow_html=True)
    with st.container():
        u_role = st.selectbox("I am a...", ["Student", "Parent", "Teacher", "Other"], index=1)
        c1, c2 = st.columns(2)
        with c1: z_code = st.text_input("Zip Code", placeholder="e.g. 96813")
        with c2:
            if u_role == "Student": u_age = st.number_input("Your Age", 3, 18, 10); c_age = None
            elif u_role == "Parent": u_age = 35; c_age = st.number_input("Child's Age", 1, 18, 5)
            else: u_age = 35; c_age = None
        
        if st.button("Start Exploring"):
            if z_code:
                st.session_state.profile.update({"zip": z_code, "role": u_role, "age": u_age, "child_age": c_age})
                # Fetch Local Calendar once
                cal_p = f"Based on Zip Code {z_code} and today's date {datetime.now().strftime('%B %d')}, list 3 nature events (blooming, migration, or weather) for a {u_age if u_role=='Student' else c_age} year old. Keep it short."
                cal_res = llm_model.invoke([("system", "You are a local naturalist."), ("human", cal_p)])
                st.session_state.local_calendar = cal_res.content
                st.session_state.onboarded = True
                st.rerun()
            else: st.warning("Zip Code required!")
    st.stop()

# --- 5. BEHAVIOR ---
p = st.session_state.profile
is_student = p['role'] == "Student"
target_age = p['age'] if is_student else p['child_age']

SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for Eco-Education.
LOCATION: Zip Code {p['zip']}. You MUST acknowledge the specific plants, animals, and weather of this ZIP.
ADAPTATION: Speak for a {target_age} year old. 
- 10yo: Use "Nature's recycling" instead of "Decomposition".
- 16yo: Use "Nutrient cycling" and "Symbiosis".

STRICT RULES:
1. NO GEAR: Never tell users to buy items (flashlights, kits). Focus on what they can see/do for free.
2. ZIP TRUTH: Only confirm sightings possible in {p['zip']}.
3. UNDERLINING: Wrap exactly 1-2 'Power Words' in <u>word</u> tags.
4. DISCRETION: Do NOT underline easy words (moss, garden, sun).
5. NO BOT QUESTIONS: End with a statement.
"""

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown(f"### 📍 Nature in {p['zip']}")
    st.markdown(f"<div class='calendar-box'>{st.session_state.local_calendar}</div>", unsafe_allow_html=True)
    
    st.title("Next Steps")
    for s in st.session_state.suggestions:
        if st.button(s, use_container_width=True): 
            st.session_state.user_query = s
            st.rerun()
            
    if st.session_state.power_words:
        st.divider()
        st.subheader("📚 Power Words")
        for word, defn in list(st.session_state.power_words.items())[-5:]: 
            st.markdown(f"**{word}**: {defn}")
            
    if st.button("🔄 Reset"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# --- 7. CHAT ENGINE ---
prompt_template = ChatPromptTemplate.from_messages([("system", SYSTEM_BEHAVIOR + "\n\nContext:\n{context}"), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
rag_chain = ({"context": (lambda x: x["input"]) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]} | prompt_template | llm_model | StrOutputParser())

for m in st.session_state.messages: st.chat_message(m["role"]).markdown(m["content"], unsafe_allow_html=True)
if not st.session_state.messages:
    intro = f"Ready to explore {p['zip']} as a **{p['role']}**. What should we look for?"
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
        
        # Socratic Logic + Vocab Filter
        try:
            target_desc = f"student who is {p['age']} years old" if is_student else f"parent of a {p['child_age']}yr old"
            found_underlines = re.findall(r'<u>(.*?)</u>', res)
            u_res = llm_model.invoke([("system", f"Suggest 3 short Socratic prompts a {target_desc} would ask next about {p['zip']}. Define these words ONLY if they are difficult for age {target_age}: {found_underlines}. Discard easy words. Return JSON: {{'prompts': [], 'vocab': {{}}}}"), ("human", res)])
            data = json.loads(u_res.content)
            st.session_state.suggestions = data.get("prompts", [])
            st.session_state.power_words.update(data.get("vocab", {}))
        except: pass
    st.rerun()
