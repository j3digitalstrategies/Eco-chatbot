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
    .sidebar-label { font-weight: bold; color: #2e7d32; margin-top: 10px; margin-bottom: 5px; display: block; }
    .safety-note { 
        background-color: #fff3e0; 
        color: #432818; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #e65100; 
        margin: 15px 0; 
        line-height: 1.5;
        font-weight: 500;
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
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_DIR)
    return vectorstore.as_retriever(search_kwargs={"k": 5}), ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=_api_key)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "onboarded" not in st.session_state: st.session_state.onboarded = False
if "profile" not in st.session_state: st.session_state.profile = {"zip": None, "city": "your area", "role": "Other", "age": 35, "child_age": None}
if "suggestions" not in st.session_state: st.session_state.suggestions = []
if "power_words" not in st.session_state: st.session_state.power_words = {}

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 4. ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 style='text-align:center;'>Saving Planet Earth: Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size: 1.2em;'>Based on the book by Ann Lewin-Benham</p>", unsafe_allow_html=True)
    
    # YOUR UPDATED PREAMBLE
    st.markdown("<p style='text-align: center; margin: 30px 0; font-weight: 500;'>Tell us a little bit more about yourself so we can understand how to help you explore</p>", unsafe_allow_html=True)
    
    with st.container():
        u_role = st.selectbox("I am a...", ["Student", "Parent", "Teacher", "Other"], index=1)
        c1, c2 = st.columns(2)
        with c1: z_code = st.text_input("Zip Code", placeholder="e.g. 96814")
        with c2:
            if u_role == "Student": u_age = st.number_input("Your Age", 3, 18, 10); c_age = None
            elif u_role == "Parent": u_age = 35; c_age = st.number_input("Child's Age", 1, 18, 5)
            elif u_role == "Teacher": u_age = 35; c_age = st.number_input("Student Age Level", 3, 18, 7)
            else: u_age = 35; c_age = None
        
        if st.button("Start Exploring"):
            if z_code:
                city_lookup = llm_model.invoke(f"What city is Zip Code {z_code}? Return ONLY the city and state name.").content
                st.session_state.profile.update({"zip": z_code, "city": city_lookup, "role": u_role, "age": u_age, "child_age": c_age})
                
                defaults = {
                    "Parent": ["How to start an observation?", "What is biophilia?", "Nature-based safety tips."],
                    "Teacher": ["Implementing in class?", "Documentation tips?", "Explain curriculum."],
                    "Student": ["What's in my backyard?", "Tell me a nature secret.", "How can I explore safely?"],
                    "Other": ["About the curriculum?", "Eco-Education?"]
                }
                st.session_state.suggestions = defaults.get(u_role, defaults["Other"])
                st.session_state.onboarded = True
                st.rerun()
            else: st.warning("Zip Code required!")
    st.stop()

# --- 5. BEHAVIOR ---
p = st.session_state.profile
is_student = p['role'] == "Student"
target_age = p['age'] if is_student else p['child_age']

SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for the Saving Planet Earth curriculum by Ann Lewin-Benham.
LOCATION: {p['city']}. TARGET AUDIENCE: {target_age} year old level.

STRICT CONVERSATION RULES:
1. NATURAL SPEECH: NEVER mention the "Zip Code" {p['zip']}. Refer to "{p['city']}" or "your neighborhood." 
2. GEOGRAPHIC REALISM: Only discuss species present in {p['city']}.
3. ADAPTIVE TONE: Speak at a level perfect for {target_age} years old.
4. SAFETY PROTOCOL: If you discuss hazards, use:
   <div class='safety-note'><b>⚠️ Safety First:</b> [Instructions]</div>.
5. UNDERLINING: Wrap exactly 1-2 'Power Words' in <u>word</u> tags. Use ONLY advanced scientific terms.
6. NO BOT QUESTIONS: End your response with a statement.
"""

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<span class='sidebar-label'>💡 Suggested Prompts</span>", unsafe_allow_html=True)
    for s in st.session_state.suggestions:
        if st.button(s, key=f"btn_{s}", use_container_width=True): 
            st.session_state.user_query = s
            st.rerun()
            
    if st.session_state.power_words:
        st.divider()
        st.markdown("<span class='sidebar-label'>📚 Power Words</span>", unsafe_allow_html=True)
        for word, defn in list(st.session_state.power_words.items())[-5:]: 
            st.markdown(f"**{word}**: {defn}")
            
    st.divider()
    if st.button("🔄 Reset Profile"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# --- 7. CHAT ENGINE ---
prompt_template = ChatPromptTemplate.from_messages([("system", SYSTEM_BEHAVIOR + "\n\nContext:\n{context}"), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
rag_chain = ({"context": (lambda x: x["input"]) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]} | prompt_template | llm_model | StrOutputParser())

for m in st.session_state.messages: st.chat_message(m["role"]).markdown(m["content"], unsafe_allow_html=True)
if not st.session_state.messages:
    intro = f"Ready to explore {p['city']}! Let's discover what's waiting in your neighborhood."
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
        
        try:
            found_underlines = re.findall(r'<u>(.*?)</u>', res)
            target_desc = f"student who is {p['age']} years old" if is_student else f"parent/teacher of a {p['child_age']}yr old"
            u_res = llm_model.invoke([
                ("system", f"Suggest 3 natural Socratic prompts a {target_desc} would ask next about {p['city']}. Define these underlined words ONLY if they are advanced scientific terms: {found_underlines}. Return JSON: {{'prompts': [], 'vocab': {{}}}}"),
                ("human", res)
            ])
            data = json.loads(u_res.content)
            st.session_state.suggestions = data.get("prompts", [])
            st.session_state.power_words = data.get("vocab", {})
        except: pass
    st.rerun()
