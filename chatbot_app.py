import streamlit as st
import os
import glob
from dotenv import load_dotenv

# LangChain Core and OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    Docx2txtLoader, 
    TextLoader, 
    PyPDFLoader
)
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

    # Define loaders for different file types to avoid 'unstructured' errors
    loaders = {
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
    }

    def create_loader(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in loaders:
            return loaders[ext](file_path)
        return None

    try:
        with st.spinner("🌱 Reading curriculum chapters..."):
            documents = []
            # Manually find files to ensure we see what's happening
            all_files = glob.glob(os.path.join(DOCS_DIR, "**/*.*"), recursive=True)
            
            for file_path in all_files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in loaders:
                    try:
                        loader = loaders[ext](file_path)
                        documents.extend(loader.load())
                    except Exception as e:
                        st.warning(f"Could not load {file_path}: {e}")

            if not documents:
                st.error("No documents were successfully loaded.")
                return None
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create Vector Store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Setup LLM
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
            
            # The Prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for the Eco-Education curriculum. Use the provided context to answer questions accurately. Context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            # Modern Chain Logic
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            return rag_chain

    except Exception as e:
        st.error(f"Engine failure: {e}")
        return None

# --- 3. UI ---
st.title("🌱 Eco-Education Assistant")
st.markdown("Ask questions about the curriculum documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("What would you like to know about the curriculum?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OpenAI API Key in Streamlit Secrets!")
        else:
            chain = get_bot_chain(api_key)
            if chain:
                # Build history for context
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) 
                    for m in st.session_state.messages[:-1]
                ]
                
                with st.spinner("Searching curriculum..."):
                    response_text = chain.invoke({"input": user_input, "chat_history": history})
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()
