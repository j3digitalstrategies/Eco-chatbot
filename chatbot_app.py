import streamlit as st
import os
import glob
from dotenv import load_dotenv

# Core LangChain Imports
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
st.set_page_config(page_title="Eco-Assistant", layout="wide", page_icon="🌱")

# --- 2. THE ENGINE ---
@st.cache_resource
def get_bot_chain(_api_key):
    if not os.path.exists(DOCS_DIR):
        st.error(f"❌ Folder '{DOCS_DIR}' not found.")
        return None

    try:
        # Strictly ignore .doc and only look for modern formats
        valid_extensions = [".docx", ".pdf", ".txt"]
        all_files = []
        for ext in valid_extensions:
            all_files.extend(glob.glob(os.path.join(DOCS_DIR, f"**/*{ext}"), recursive=True))

        if not all_files:
            st.error("No compatible (.docx, .pdf, or .txt) files found.")
            return None

        with st.spinner(f"🌱 Building Brain from {len(all_files)} curriculum files..."):
            documents = []
            for file_path in all_files:
                ext = os.path.splitext(file_path)[1].lower()
                try:
                    if ext == ".docx":
                        loader = Docx2txtLoader(file_path)
                    elif ext == ".pdf":
                        loader = PyPDFLoader(file_path)
                    elif ext == ".txt":
                        loader = TextLoader(file_path)
                    documents.extend(loader.load())
                except Exception:
                    continue 

            if not documents:
                return None
            
            # Text Processing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(documents)
            
            # Vector DB
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # AI Model
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert on the Eco-Education curriculum. Use the context to answer accurately. Context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # --- FIXED CHAIN LOGIC ---
            # We use a dictionary to prepare context, then pass the input string directly
            rag_chain = (
                {
                    "context": (lambda x: x["input"]) | retriever | format_docs,
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"],
                }
                | prompt 
                | llm 
                | StrOutputParser()
            )
            return rag_chain

    except Exception as e:
        st.error(f"Critical System Error: {e}")
        return None

# --- 3. UI ---
st.title("🌱 Eco-Education Assistant")

# GET API KEY
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API Key not found in Streamlit Secrets.")
else:
    # PRE-LOAD: Build the chain as soon as the app starts
    chain = get_bot_chain(api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    user_input = st.chat_input("Ask me about Chapter 12 or any other curriculum topic...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        if chain:
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) 
                for m in st.session_state.messages[:-1]
            ]
            
            with st.chat_message("assistant"):
                with st.spinner("Searching curriculum..."):
                    # Pass the required dictionary structure to the chain
                    response = chain.invoke({"input": user_input, "chat_history": history})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
