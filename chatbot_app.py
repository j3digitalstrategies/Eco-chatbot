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
    .safety-banner { 
        background-color: #fff3e0; color: #d35400; padding: 10px; border-radius: 10px; 
        border: 2px solid #e67e22; font-weight: bold; margin-bottom: 15px; text-align: center;
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
    st.markdown("<h1 style='text-align:center;'>Saving Planet Earth: Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size: 1.2em;'>Based on the book by Ann Lewin-Benham</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin: 30px 0; font-weight: 500;'>Tell us a little bit more about yourself so we can understand how to help you explore</p>", unsafe_allow_html=True)
    
    with st.container():
        u_role = st.selectbox("I am a...", ["Student", "Parent", "Teacher", "Other"], index=0)
        c1, c2 = st.columns(2)
        with c1: z_code = st.text_input("Zip Code", placeholder="e.g. 90210")
        with c2: u_age = st.number_input("Age (or target student age)", 3, 100, 10)
        
        if st.button("Start Exploring"):
            if z_code:
                # Lookup city based on Zip Code provided
                city_lookup = llm_model.invoke(f"What city and state is Zip Code {z_code}? Return ONLY 'City, State'.").content
                st.session_state.profile.update({"zip": z_code, "city": city_lookup, "role": u_role, "age": u_age})
                
                if u_role == "Student":
                    st.session_state.suggestions = [f"What animals live in {city_lookup}?", "How do I spot local nature?", "What should I bring outside?"]
                else:
                    st.session_state.suggestions = ["How do I foster biophilia?", "Observation tips", "Curriculum details"]
                
                st.session_state.onboarded = True
                st.rerun()
            else: st.warning("Zip Code required!")
    st.stop()

# --- 5. BEHAVIOR ---
p = st.session_state.profile
age = p['age']

SYSTEM_BEHAVIOR = f"""
You are a mentor for the Saving Planet Earth curriculum. 
STRICT LOCATION: {p['city']} (Zip: {p['zip']}). 
USER ROLE: {p['role']}. USER AGE: {age}.

STRICT RULES:
1. DYNAMIC GEOGRAPHY: You must ONLY talk about nature and wildlife that exists in {p['city']}. If a user mentions an animal or plant not native to {p['city']}, you must gently correct them and provide a real local alternative.
2. SAFETY FIRST: Every outdoor suggestion MUST include the warning: "Ask a parent or adult first before heading out."
3. ASK BEFORE ADVISING: Ask the user what they like to do or what their local yard/park is like in the middle of your response.
4. BE CONCISE: Max 2 sentences. No academic jargon for students.
5. NO END QUESTIONS: End with a supportive statement.
6. UNDERLINE: Wrap 1-2 'Power Words' in <u>word</u> tags.
"""

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<span class='sidebar-label'>💡 Suggested Prompts</span>", unsafe_allow_html=True)
    for idx, s in enumerate(st.session_state.suggestions):
        if st.button(s, key=f"sug_{idx}_{hash(s)}", use_container_width=True): 
            st.session_state.user_query = s
            st.rerun()
            
    if st.session_state.power_words:
        st.divider()
        st.markdown("<span class='sidebar-label'>📚 Power Words</span>", unsafe_allow_html=True)
        for word, defn in st.session_state.power_words.items(): 
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
    intro = f"Ready to explore {p['city']}! What do you like to do most when you're outside?"
    st.chat_message("assistant").markdown(intro); st.session_state.messages.append({"role": "assistant", "content": intro})

query = st.session_state.get("user_query") or st.chat_input("Type here...")
if query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    
    hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        res = rag_chain.invoke({"input": query, "chat_history": hist})
        res = re.sub(r'\?\s*$', '.', res.strip())
        
        # Mandatory Safety Warning
        if any(w in res.lower() for w in ["outside", "explore", "walk", "yard", "park"]):
            st.markdown("<div class='safety-banner'>⚠️ Always ask an adult before going outside!</div>", unsafe_allow_html=True)

        st.markdown(res, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": res})
        
        underlined = re.findall(r'<u>(.*?)</u>', res)
        try:
            suggest_prompt = f"""
            Generate 3 SHORT questions the {p['role']} (Age {age}) would ask about nature in {p['city']}.
            STRICT RULE: Factual accuracy for {p['city']}. 
            Student prompts must be curious. Parent prompts must be about <u>scaffolding</u> or <u>biophilia</u>.
            Include definitions ONLY for: {underlined}.
            Return JSON: {{"prompts": [], "vocab": {{}}}}
            """
            u_res = llm_model.invoke([("system", suggest_prompt), ("human", res)])
            data = json.loads(u_res.content)
            st.session_state.suggestions = data.get("prompts", [])
            st.session_state.power_words = {k: v for k, v in data.get("vocab", {}).items() if k.lower() in [w.lower() for w in underlined]}
        except: pass
    st.rerun()
