import streamlit as st
import os, json, re
from dotenv import load_dotenv

# Core LangChain & AI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIG & STYLING ---
load_dotenv()
VECTOR_DB_DIR = "vector_db"
st.set_page_config(page_title="Saving Planet Earth", layout="wide", page_icon="🌱")

st.markdown("""
    <style>
    .stButton button { display: block; margin: 0 auto; padding: 8px 30px; border-radius: 15px; background-color: #2e7d32; color: white; border: none; }
    u { text-decoration: underline; color: #2e7d32; font-weight: bold; }
    .sidebar-label { font-weight: bold; color: #2e7d32; margin-top: 10px; margin-bottom: 5px; display: block; }
    .safety-note { 
        background-color: #fff3e0; color: #432818; padding: 12px; border-radius: 8px; border-left: 5px solid #e65100; margin: 10px 0; font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5}), ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "onboarded" not in st.session_state: st.session_state.onboarded = False
if "profile" not in st.session_state: st.session_state.profile = {"zip": None, "city": "your area", "role": "Student", "age": 10}
if "suggestions" not in st.session_state: st.session_state.suggestions = []
if "power_words" not in st.session_state: st.session_state.power_words = {}

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 4. ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 style='text-align:center;'>Saving Planet Earth</h1>", unsafe_allow_html=True)
    with st.container():
        u_role = st.selectbox("I am a...", ["Student", "Parent", "Teacher", "Other"])
        z_code = st.text_input("Zip Code")
        u_age = st.number_input("Age level", 3, 100, 10)
        if st.button("Start"):
            city = llm_model.invoke(f"Zip {z_code} city/state?").content
            st.session_state.profile.update({"zip": z_code, "city": city, "role": u_role, "age": u_age})
            st.session_state.suggestions = [f"What's in {city}?", "Curriculum help"] if u_role != "Student" else ["Backyard animals"]
            st.session_state.onboarded = True
            st.rerun()
    st.stop()

# --- 5. BEHAVIOR ---
p = st.session_state.profile
SYSTEM_BEHAVIOR = f"""
You are a peer-mentor for the Saving Planet Earth curriculum. Location: {p['city']}. 
STRICT RULES:
1. NEVER end with a question. No "What do you think?" or "How can I help?". End with a strong statement.
2. If role is Parent/Teacher: Focus heavily on pedagogy (<u>biophilia</u>, <u>scaffolding</u>, documentation).
3. If role is Student: Connect their hobbies to nature physics/biology.
4. Underline 1-2 'Power Words' using <u>word</u> tags.
"""

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<span class='sidebar-label'>💡 Suggested Prompts</span>", unsafe_allow_html=True)
    for idx, s in enumerate(st.session_state.suggestions):
        if st.button(s, key=f"s_{idx}"): 
            st.session_state.user_query = s
            st.rerun()
    if st.session_state.power_words:
        st.divider()
        st.markdown("<span class='sidebar-label'>📚 Power Words</span>", unsafe_allow_html=True)
        for w, d in st.session_state.power_words.items(): st.markdown(f"**{w}**: {d}")
    if st.button("🔄 Reset"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# --- 7. CHAT ENGINE ---
prompt_template = ChatPromptTemplate.from_messages([("system", SYSTEM_BEHAVIOR + "\nContext: {context}"), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
rag_chain = ({"context": (lambda x: x["input"]) | retriever | (lambda docs: "\n".join(d.page_content for d in docs)), "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]} | prompt_template | llm_model | StrOutputParser())

for m in st.session_state.messages: st.chat_message(m["role"]).markdown(m["content"], unsafe_allow_html=True)

query = st.session_state.get("user_query") or st.chat_input("Type here...")
if query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    
    hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        res = rag_chain.invoke({"input": query, "chat_history": hist})
        # Clean trailing questions just in case
        res = re.sub(r'\?$', '.', res.strip())
        st.markdown(res, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": res})
        
        # EXTRACT UNDERLINED WORDS
        found_words = re.findall(r'<u>(.*?)</u>', res)
        
        try:
            u_res = llm_model.invoke([
                ("system", f"Suggest 3 SHORT prompts for a {p['role']} to ask NEXT. Write from USER'S perspective. Provide definitions ONLY for these specific words: {found_words}. Return JSON: {{'prompts': [], 'vocab': {{}}}}")
            ])
            data = json.loads(u_res.content)
            st.session_state.suggestions = data.get("prompts", [])
            # Filter vocab to ensure only words actually in the text appear
            st.session_state.power_words = {k: v for k, v in data.get("vocab", {}).items() if k.lower() in [w.lower() for w in found_words]}
        except: pass
    st.rerun()
