import streamlit as st
import os, glob, json, re
from datetime import datetime
from dotenv import load_dotenv

# Core LangChain & AI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
if "persistent_vocab" not in st.session_state: st.session_state.persistent_vocab = {} 

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()
retriever, llm_model = get_bot_chain(api_key)

# --- 4. ONBOARDING ---
if not st.session_state.onboarded:
    st.markdown("<h1 style='text-align:center;'>Saving Planet Earth: Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size: 1.2em;'>Based on the book by Ann Lewin-Benham</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin: 30px 0; font-weight: 500;'>Tell us a little bit more about yourself so we can understand how to help you explore</p>", unsafe_allow_html=True)
    
    with st.container():
        # MERGED DROPDOWN
        u_role = st.selectbox("I am a...", ["Student", "Parent/Teacher"], index=0)
        c1, c2 = st.columns(2)
        with c1: z_code = st.text_input("Zip Code", placeholder="e.g. 90210")
        with c2: u_age = st.number_input("Child's Age (or target student age)", 3, 100, 10)
        
        if st.button("Start Exploring"):
            if z_code:
                city_lookup = llm_model.invoke(f"What city and state is Zip Code {z_code}? Return ONLY 'City, State'.").content
                st.session_state.profile.update({"zip": z_code, "city": city_lookup, "role": u_role, "age": u_age})
                
                if u_role == "Student":
                    st.session_state.suggestions = ["What are the secrets of the trees?", "How do I start exploring?", "Tell me a nature secret"]
                else:
                    st.session_state.suggestions = ["What is the core philosophy?", "How do I facilitate an inquiry?", "Explain the importance of documentation"]
                
                st.session_state.onboarded = True
                st.rerun()
            else: st.warning("Zip Code required!")
    st.stop()

# --- 5. BEHAVIOR ---
p = st.session_state.profile

if p['role'] == 'Parent/Teacher':
    ROLE_SPECIFIC_RULES = """
    1. PROFESSIONAL MENTOR: You are an expert in the 'Saving Planet Earth' curriculum pedagogy.
    2. PEDAGOGY OVER RECREATION: Focus on <u>documentation</u>, <u>scaffolding</u>, and <u>representation</u>. 
    3. NO SAFETY LECTURES: Never tell an adult to 'bring a parent' or 'stay safe.' 
    4. SUBSTANCE OVER QUESTIONS: Provide concrete pedagogical strategies from the textbook.
    5. CURRICULUM FOCUS: Discuss the 'subject of inquiry' and how to guide the child's analytic and collaborative behaviors.
    """
else:
    ROLE_SPECIFIC_RULES = """
    1. SAFETY: Tell them to bring a parent/adult for outdoor activities.
    2. NO TOUCHING: Remind them to observe from a distance.
    3. AGE APPROPRIATE: No drugs or inappropriate content. Pivot to nature.
    """

SYSTEM_BEHAVIOR = f"""
You are an expert mentor for the Saving Planet Earth curriculum. Location: {p['city']}. 
USER ROLE: {p['role']}. TARGET AGE: {p['age']}.

{ROLE_SPECIFIC_RULES}

GENERAL RULES:
6. NO THERAPY: Answer the user directly. Ask at most ONE targeted question to build context for a <u>subject of inquiry</u>.
7. PUNCTUATION: Every response MUST end with a (.) OR a (?).
8. CONCISE: 3-4 sentences maximum.
"""

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<span class='sidebar-label'>💡 Suggested Prompts</span>", unsafe_allow_html=True)
    for idx, s in enumerate(st.session_state.suggestions):
        if st.button(s, key=f"sug_{idx}_{hash(s)}", use_container_width=True): 
            st.session_state.user_query = s
            st.rerun()
            
    if st.session_state.persistent_vocab:
        st.divider()
        st.markdown("<span class='sidebar-label'>📚 Power Words</span>", unsafe_allow_html=True)
        for word, defn in st.session_state.persistent_vocab.items(): 
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
    if p['role'] == 'Student':
        intro = f"Hi! I'm your nature mentor in {p['city']}. I'm here to help you uncover the hidden secrets of the world outside your door."
    else:
        intro = f"Welcome. I am here to support you in mentoring a {p['age']}-year-old in {p['city']} using the Saving Planet Earth curriculum. We can explore how to foster a deeper connection to nature through observation and meaningful play."
    st.chat_message("assistant").markdown(intro); st.session_state.messages.append({"role": "assistant", "content": intro})

query = st.session_state.get("user_query") or st.chat_input("Type here...")
if query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    
    hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        res = rag_chain.invoke({"input": query, "chat_history": hist}).strip()
        
        # FINAL PUNCTUATION FIX
        if not (res.endswith('.') or res.endswith('?')):
            res += "."
        
        st.markdown(res, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": res})
        
        underlined = re.findall(r'<u>(.*?)</u>', res)
        try:
            suggest_prompt = f"""
            Generate exactly 3 short follow-up questions that the {p['role']} would ask the AI. 
            The questions MUST be from the user's perspective to the AI.
            STRICT: If role is Parent/Teacher, focus on pedagogical methods.
            Return ONLY JSON: {{"prompts": [], "vocab": {{}}}}
            """
            u_res = llm_model.invoke([("system", suggest_prompt), ("human", res)])
            data = json.loads(u_res.content)
            st.session_state.suggestions = data.get("prompts", [])
            
            new_defs = data.get("vocab", {})
            blocklist = ["relationship", "nature", "observation", "community", "environment", "parent", "teacher"]
            for word, defn in new_defs.items():
                w_lower = word.lower()
                if any(u.lower() == w_lower for u in underlined):
                    if p['role'] != 'Student' and w_lower in blocklist: continue
                    st.session_state.persistent_vocab[w_lower] = defn
        except: pass
    st.rerun()
