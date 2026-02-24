import streamlit as st
import os
import glob
import json
import time
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

# --- 1. CONFIG & SYSTEM PROMPTS ---
load_dotenv()
DOCS_DIR = "curriculum_docs"
VECTOR_DB_DIR = "vector_db"
st.set_page_config(page_title="Saving Planet Earth", layout="wide", page_icon="🌱")

# Theme-aware Mobile Tip
st.markdown("""
    <style>
    .mobile-tip {
        display: none;
        background-color: rgba(128, 128, 128, 0.1);
        color: inherit;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e7d32;
        margin-bottom: 20px;
        text-align: center;
    }
    @media (max-width: 768px) { .mobile-tip { display: block; } }
    </style>
    <div class="mobile-tip">
        <b>📱 Mobile User Tip:</b> Tap the <b>></b> arrow in the top-left corner to find your <b>Suggested Prompts</b>!
    </div>
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) 
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=_api_key)
    return retriever, llm

# --- 3. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "step" not in st.session_state: st.session_state.step = "zip"
if "profile" not in st.session_state: 
    st.session_state.profile = {"zip": None, "role": None, "age": None, "season": datetime.now().strftime("%B")}
if "suggestions" not in st.session_state:
    st.session_state.suggestions = ["What are the big ideas of this curriculum?", "Tell me about ecosystems.", "Ann Lewin-Benham's philosophy."]

# --- 4. UI HEADERS ---
st.title("Saving Planet Earth: Revolutionary Ways to Teach Eco-Education Chatbot")
st.subheader("by Ann Lewin-Benham")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("API Key missing.")
    st.stop()

retriever, llm_model = get_bot_chain(api_key)

# --- 5. ONBOARDING WIZARD ---
if st.session_state.step != "complete":
    with st.container():
        st.info("🌱 **Let's personalize your experience.**")
        
        if st.session_state.step == "zip":
            zip_in = st.text_input("Please enter your Zip Code (to tailor activities to your local season):")
            if st.button("Next") and zip_in:
                st.session_state.profile["zip"] = zip_in
                st.session_state.step = "role"
                st.rerun()

        elif st.session_state.step == "role":
            st.write("Are you a:")
            col1, col2, col3, col4 = st.columns(4)
            if col1.button("Teacher"): 
                st.session_state.profile["role"] = "Teacher"
                st.session_state.step = "complete"
                st.rerun()
            if col2.button("Parent"): 
                st.session_state.profile["role"] = "Parent"
                st.session_state.step = "complete"
                st.rerun()
            if col3.button("Student"): 
                st.session_state.profile["role"] = "Student"
                st.session_state.step = "age"
                st.rerun()
            if col4.button("Other"): 
                st.session_state.profile["role"] = "Other"
                st.session_state.step = "complete"
                st.rerun()

        elif st.session_state.step == "age":
            age_in = st.number_input("How old are you?", min_value=3, max_value=100, value=12)
            if st.button("Finish Setup"):
                st.session_state.profile["age"] = age_in
                st.session_state.step = "complete"
                st.rerun()
    st.stop() 

# --- 6. PERSONALIZED SYSTEM PROMPT ---
p = st.session_state.profile
role_context = {
    "Teacher": "Focus on classroom implementation, pedagogical theory, and curriculum mapping.",
    "Parent": "Focus on home-based activities, simple explanations for kids, and fostering curiosity.",
    "Student": f"Adjust ALL verbiage to be age-appropriate for a {p['age']}-year-old. Use relatable analogies.",
    "Other": "Provide clear, professional, and accessible summaries of the curriculum."
}

SYSTEM_BEHAVIOR = f"""
You are the Eco-Education Assistant for Ann Lewin-Benham's curriculum. 
USER PROFILE: Role: {p['role']}, Age: {p['age']}, Zip Code: {p['zip']}, Current Month: {p['season']}.

INSTRUCTIONS:
1. PIVOT: Always consider the user's location and the current season ({p['season']}) when suggesting outdoor activities.
2. TONE & SCOPE: {role_context.get(p['role'])}
3. RULES: Only child-safe content. Treat documents as a unified curriculum.
4. REFUSAL: Refuse adult/sensitive topics politely and return to the curriculum.
"""

# --- 7. CHAT ENGINE SETUP ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BEHAVIOR + "\n\nContext from curriculum documents:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

rag_chain = (
    {"context": (lambda x: x["input"]) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]}
    | prompt_template | llm_model | StrOutputParser()
)

# --- 8. SIDEBAR & DISPLAY ---
st.sidebar.title("Suggested Prompts")
for s in st.session_state.suggestions:
    if st.sidebar.button(s): st.session_state.user_query = s

if st.sidebar.button("🔄 Reset Profile"):
    st.session_state.step = "zip"
    st.session_state.messages = []
    st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Intro sequence
if not st.session_state.messages:
    with st.chat_message("assistant"):
        intro_msg = f"Welcome! I have personalized my responses for a **{p['role']}** in Zip Code **{p['zip']}**. How can I help you explore the curriculum today?"
        st.markdown(intro_msg)
        st.session_state.messages.append({"role": "assistant", "content": intro_msg})

# --- 9. USER INPUT HANDLING ---
user_input = st.chat_input("Ask a question about the curriculum...")
final_query = st.session_state.get("user_query") or user_input

if final_query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): st.markdown(final_query)
    
    # Reconstructed history logic with closed brackets/parentheses
    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" 
        else AIMessage(content=m["content"]) 
        for m in st.session_state.messages[:-1]
    ]
    
    with st.chat_message("assistant"):
        def stream_response():
            full_response = rag_chain.invoke({"input": final_query, "chat_history": history})
            for word in full_response.split(" "):
                yield word + " "
                time.sleep(0.04)
        res_text = st.write_stream(stream_response())
        st.session_state.messages.append({"role": "assistant", "content": res_text})
        
        # Adaptive suggestions
        try:
            suggest_p = f"Based on this topic for a {p['role']}, generate 3 follow-up prompts. Return ONLY a JSON list of strings."
            res = llm_model.invoke([("system", suggest_p), ("human", res_text)])
            st.session_state.suggestions = json.loads(res.content)
        except Exception: 
            pass 
    st.rerun()
