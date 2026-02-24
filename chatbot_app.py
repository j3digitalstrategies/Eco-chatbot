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

# --- 1. CONFIG ---
load_dotenv()
DOCS_DIR = "curriculum_docs"
VECTOR_DB_DIR = "vector_db"
st.set_page_config(page_title="Saving Planet Earth", layout="wide", page_icon="🌱")

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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=_api_key)
    return retriever, llm

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "step" not in st.session_state: st.session_state.step = "zip"
if "profile" not in st.session_state: 
    st.session_state.profile = {"zip": None, "role": None, "age": None, "season": datetime.now().strftime("%B")}
if "suggestions" not in st.session_state:
    st.session_state.suggestions = ["What makes moss bouncy?", "How does moss help trees?", "Where can we find more moss?"]

# --- 4. UI SETUP ---
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
            zip_in = st.text_input("Please enter your Zip Code:")
            if st.button("Next") and zip_in:
                st.session_state.profile["zip"] = zip_in
                st.session_state.step = "role"; st.rerun()
        elif st.session_state.step == "role":
            st.write("Are you a:")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Teacher"): st.session_state.profile["role"] = "Teacher"; st.session_state.step = "complete"; st.rerun()
            if c2.button("Parent"): st.session_state.profile["role"] = "Parent"; st.session_state.step = "complete"; st.rerun()
            if c3.button("Student"): st.session_state.profile["role"] = "Student"; st.session_state.step = "age"; st.rerun()
            if c4.button("Other"): st.session_state.profile["role"] = "Other"; st.session_state.step = "complete"; st.rerun()
        elif st.session_state.step == "age":
            age_in = st.number_input("How old are you?", min_value=3, max_value=100, value=12)
            if st.button("Finish Setup"):
                st.session_state.profile["age"] = age_in; st.session_state.step = "complete"; st.rerun()
    st.stop()

# --- 6. SOCRATIC SYSTEM BEHAVIOR ---
p = st.session_state.profile
role_context = {
    "Teacher": "Provide pedagogical frameworks and classroom-ready prompts.",
    "Parent": "Act as a co-explorer. Focus on sparking conversation between parent and child.",
    "Student": f"Speak as a mentor to a {p['age']}-year-old. Use clear, vivid, and simple language.",
    "Other": "Provide accessible summaries of the Eco-Education philosophy."
}

SYSTEM_BEHAVIOR = f"""
You are a Socratic Eco-Education Mentor based on Ann Lewin-Benham's work.
USER: {p['role']}, Age: {p['age']}, Zip: {p['zip']}, Month: {p['season']}.

CONVERSATIONAL PROTOCOL:
1. NO DUMPING: Never provide a full list of activities or long explanations in the first response.
2. OBSERVE & PROMPT: Use the user's descriptions (e.g., 'stringy', 'bouncy') to explain a specific concept from the curriculum. 
3. MEANING-FULL CONVERSATION: End your response with 1-2 questions for the user to ask their child, or questions to help you narrow down the next step.
4. TONE: {role_context.get(p['role'])}. Be warm, curious, and grounded in the text.
"""

# --- 7. SIDEBAR (RESET AT BOTTOM) ---
with st.sidebar:
    st.title("Suggested Prompts")
    st.caption("Click a prompt to ask the chatbot:")
    for s in st.session_state.suggestions:
        if st.button(s): st.session_state.user_query = s
    
    # Push the Reset button to the bottom using empty space
    for _ in range(15): st.write("") 
    st.divider()
    if st.button("🔄 Reset Profile"):
        st.session_state.step = "zip"; st.session_state.messages = []; st.rerun()

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

# --- 9. DISPLAY & INPUT ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if not st.session_state.messages:
    with st.chat_message("assistant"):
        intro = f"Welcome! I'm ready to explore with a **{p['role']}** in Zip Code **{p['zip']}**. What have you observed in nature today?"
        st.markdown(intro); st.session_state.messages.append({"role": "assistant", "content": intro})

user_input = st.chat_input("Type your observation or question here...")
final_query = st.session_state.get("user_query") or user_input

if final_query:
    if "user_query" in st.session_state: del st.session_state["user_query"]
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): st.markdown(final_query)
    
    # FIXED: Reconstructed history logic with clean list comprehension
    history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        def stream_response():
            full_response = rag_chain.invoke({"input": final_query, "chat_history": history})
            for word in full_response.split(" "):
                yield word + " "; time.sleep(0.04)
        res_text = st.write_stream(stream_response())
        st.session_state.messages.append({"role": "assistant", "content": res_text})
        
        # IMPROVED SUGGESTION LOGIC: Force prompt to be from User's perspective
        try:
            suggest_p = f"""
            Based on the current topic ({final_query}), generate 3 short prompts. 
            These MUST be questions the USER (a {p['role']}) would ask the CHATBOT.
            Do NOT ask about the user's child's interest. 
            Instead, suggest questions like: "What is moss made of?" or "Give me a simple moss activity."
            Return ONLY a JSON list of strings.
            """
            res = llm_model.invoke([("system", suggest_p), ("human", res_text)])
            st.session_state.suggestions = json.loads(res.content)
        except Exception: 
            st.session_state.suggestions = ["Tell me more about this.", "What is a good next step?", "Explain this simply."]
            
    st.rerun()
