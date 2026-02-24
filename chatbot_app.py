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

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "step" not in st.session_state: st.session_state.step = "zip"
if "profile" not in st.session_state: 
    st.session_state.profile = {"zip": None, "role": None, "age": None, "kid_age": None, "season": datetime.now().strftime("%B")}
if "suggestions" not in st.session_state:
    st.session_state.suggestions = ["How to start an observation?", "Core pillars of the curriculum?", "Tell me about Ann Lewin-Benham."]
if "power_words" not in st.session_state: st.session_state.power_words = {}

# --- 4. UI SETUP ---
st.title("Saving Planet Earth: Eco-Education Assistant")
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("API Key missing."); st.stop()

retriever, llm_model = get_bot_chain(api_key)

# --- 5. ONBOARDING ---
if st.session_state.step != "complete":
    with st.container():
        st.info("🌱 **Quick Personalization**")
        if st.session_state.step == "zip":
            zip_in = st.text_input("Zip Code:")
            if st.button("Next") and zip_in:
                st.session_state.profile["zip"] = zip_in
                st.session_state.step = "role"; st.rerun()
        elif st.session_state.step == "role":
            st.write("I am a:")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Teacher"): st.session_state.profile["role"] = "Teacher"; st.session_state.step = "complete"; st.rerun()
            if c2.button("Parent"): st.session_state.profile["role"] = "Parent"; st.session_state.step = "complete"; st.rerun()
            if c3.button("Student"): st.session_state.profile["role"] = "Student"; st.session_state.step = "age"; st.rerun()
            if c4.button("Other"): st.session_state.profile["role"] = "Other"; st.session_state.step = "complete"; st.rerun()
        elif st.session_state.step == "age":
            age_in = st.number_input("How old are you?", min_value=3, max_value=100, value=14)
            if st.button("Finish"):
                st.session_state.profile["age"] = age_in; st.session_state.step = "complete"; st.rerun()
    st.stop()

# --- 6. BEHAVIOR ---
p = st.session_state.profile
SYSTEM_BEHAVIOR = f"""
You are a peer-like Socratic mentor for Ann Lewin-Benham's Eco-Education.
PROFILE: Role={p['role']}, UserAge={p['age']}, KidAge={p['kid_age'] if p['kid_age'] else 'Unknown'}, Zip={p['zip']}, Season={p['season']}.

STRICT RULES:
1. BREVITY: Max 2 paragraphs. No fluff.
2. ROLE SHIFT: If user says "Switch to student/parent/teacher", confirm the change and shift tone.
3. PARENT ROLE: Suggest 1 activity and 1-2 prompts for the child. 
4. STUDENT ROLE: Mentor age {p['age'] or p['kid_age'] or '12'} directly. 
5. POWER WORDS: Bold them in your text. Only use definitions for Adults if the word is academic/complex (e.g. Taxonomy). Use more definitions for Students.
"""

# --- 7. SIDEBAR ---
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
            st.markdown(f"**{word}**: {defn}")

    for _ in range(5): st.write("") 
    st.divider()
    if st.button("📄 Session Summary"):
        summary_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download Summary", summary_text, file_name="eco_session.txt")
    
    if st.button("🔄 Reset Profile"):
        st.session_state.step = "zip"; st.session_state.messages = []; st.session_state.profile["kid_age"] = None; st.session_state.power_words = {}; st.rerun()

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

# --- 9. DISPLAY ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if not st.session_state.messages:
    intro = f"Ready to explore as a **{p['role']}**. What's your focus today?"
    with st.chat_message("assistant"): st.markdown(intro)
    st.session_state.messages.append({"role": "assistant", "content": intro})

# --- 10. INPUT & DYNAMIC UPDATES ---
user_input = st.chat_input("Type here...")
query = st.session_state.get("user_query") or user_input

if query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    
    # Improved Role switch detection
    for k, v in {"student": "Student", "teacher": "Teacher", "parent": "Parent"}.items():
        if f"switch to {k}" in query.lower() or f"i am a {k}" in query.lower() or f"for my {k}" in query.lower():
            st.session_state.profile["role"] = v
            # Clear power words when role changes to refresh vocabulary relevance
            st.session_state.power_words = {}

    # Age detection
    if st.session_state.profile["kid_age"] is None:
        nums = re.findall(r'\d+', query)
        if nums and any(w in query.lower() for w in ["year", "age", "is"]): 
            st.session_state.profile["kid_age"] = nums[0]

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        res_container = st.empty()
        full_res = rag_chain.invoke({"input": query, "chat_history": history})
        
        typed_res = ""
        for word in full_res.split(" "):
            typed_res += word + " "
            res_container.markdown(typed_res)
            time.sleep(0.01)
        
        st.session_state.messages.append({"role": "assistant", "content": full_res})
        
        # DYNAMIC UPDATES with Age-Aware Vocabulary
        try:
            update_p = f"""
            Analyze response: '{full_res}'.
            1. Generate 3 short USER questions for a {p['role']} (JSON list 'prompts').
            2. Extract vocabulary.
               - If Role is Parent/Teacher: Only extract academic/high-level terms (e.g., Biophilia, Taxonomy).
               - If Role is Student: Extract scientific terms (e.g., Interdependence, Spores).
               Return JSON: {{"prompts": [], "vocab": {{}}}}
            """
            update_res = llm_model.invoke([("system", update_p), ("human", query)])
            data = json.loads(update_res.content)
            st.session_state.suggestions = data.get("prompts", [])
            st.session_state.power_words.update(data.get("vocab", {}))
        except: pass
    st.rerun()
