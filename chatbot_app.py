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
    st.session_state.profile = {"zip": None, "role": "Other", "age": None, "kid_age": None, "season": datetime.now().strftime("%B")}
if "suggestions" not in st.session_state:
    st.session_state.suggestions = DEFAULT_PROMPTS["Other"]
if "power_words" not in st.session_state: st.session_state.power_words = {}

# --- 5. UI SETUP ---
st.title("Saving Planet Earth: Eco-Education Assistant")
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: st.error("API Key missing."); st.stop()

retriever, llm_model = get_bot_chain(api_key)

# --- 6. CONSOLIDATED ONBOARDING ---
if not st.session_state.onboarded:
    with st.container():
        st.markdown("### 🌱 Welcome! Tell us a little about yourself.")
        st.write("This helps me understand your local environment and tailor my guidance to how you want to use this assistant.")
        
        col1, col2 = st.columns(2)
        with col1:
            zip_code = st.text_input("What is your Zip Code?", placeholder="e.g. 90210")
        with col2:
            role = st.selectbox("I am a...", ["Parent", "Teacher", "Student", "Other"])
        
        user_age = None
        if role == "Student":
            user_age = st.number_input("How old are you?", min_value=3, max_value=100, value=10)

        if st.button("Start Exploring", use_container_width=True):
            if zip_code:
                st.session_state.profile["zip"] = zip_code
                st.session_state.profile["role"] = role
                st.session_state.profile["age"] = user_age
                st.session_state.suggestions = DEFAULT_PROMPTS[role]
                st.session_state.onboarded = True
                st.rerun()
            else:
                st.warning("Please enter your Zip Code to continue.")
    st.stop()

# --- 7. BEHAVIOR ---
p = st.session_state.profile
SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for Ann Lewin-Benham's Eco-Education.
CONTEXT: Role={p['role']}, Age={p['age']}, KidAge={p['kid_age'] if p['kid_age'] else 'Unknown'}, ZIP={p['zip']}.

STRICT RULES:
1. BREVITY: Max 2 paragraphs. No filler.
2. HIDDEN METADATA: Do NOT repeat the ZIP code or technical profile details back. Use them silently.
3. LOCAL KNOWLEDGE: Speak as if you naturally know the environment of ZIP {p['zip']}.
4. AGE-APPROPRIATE: For a Student or child, use simple words and metaphors. 
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
        for word, defn in st.session_state.power_words.items():
            if isinstance(defn, str):
                st.markdown(f"**{word}**: {defn}")

    for _ in range(5): st.write("") 
    st.divider()
    if st.button("🔄 Reset User Profile"):
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
    with st.chat_message(m["role"]): st.markdown(m["content"])

if not st.session_state.messages:
    intro = f"Ready to explore as a **{p['role']}**. What's your focus today?"
    with st.chat_message("assistant"): st.markdown(intro)
    st.session_state.messages.append({"role": "assistant", "content": intro})

# --- 11. INPUT & DYNAMIC LOGIC ---
user_input = st.chat_input("Type here...")
query = st.session_state.get("user_query") or user_input

if query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    
    # Internal Role Switch check
    for r_key, r_val in {"student": "Student", "teacher": "Teacher", "parent": "Parent"}.items():
        if f"switch to {r_key}" in query.lower() or f"i am a {r_key}" in query.lower():
            st.session_state.profile["role"] = r_val
            st.session_state.suggestions = DEFAULT_PROMPTS[r_val]
            st.session_state.power_words = {}
    
    # Internal Age Extraction
    nums = re.findall(r'\d+', query)
    if nums and any(w in query.lower() for w in ["year", "age", "is"]): 
        if p['role'] == "Student": st.session_state.profile["age"] = int(nums[0])
        else: st.session_state.profile["kid_age"] = int(nums[0])

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        full_res = rag_chain.invoke({"input": query, "chat_history": history})
        st.markdown(full_res)
        st.session_state.messages.append({"role": "assistant", "content": full_res})
        
        target_age = p['age'] if p['role'] == "Student" else p['kid_age']
        try:
            update_p = f"""
            Analyze: '{full_res}'. User Age: {target_age or 'Unknown'}.
            1. Suggest 3 short questions for role: {p['role']}. 
            2. VOCAB: Only pedagogical/complex for Adults; 'Stretch' words for Students.
            Return JSON: {{"prompts": [], "vocab": {{}}}}
            """
            u_res = llm_model.invoke([("system", update_p), ("human", query)])
            data = json.loads(u_res.content)
            st.session_state.suggestions = data.get("prompts", [])
            clean_vocab = {k: v for k, v in data.get("vocab", {}).items() if isinstance(v, str)}
            st.session_state.power_words.update(clean_vocab)
        except: pass
    st.rerun()
