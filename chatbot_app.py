import streamlit as st
import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CONFIG ---
load_dotenv()
DOCS_DIR = "curriculum_docs"
st.set_page_config(page_title="Eco-Assistant", layout="wide", page_icon="🌱")

# --- 2. THE ENGINE (With Compatibility Filter) ---
@st.cache_resource
def get_bot_chain(_api_key):
    # Check if folder exists
    if not os.path.exists(DOCS_DIR):
        st.error(f"❌ Folder '{DOCS_DIR}' not found in GitHub. Please create it and upload your files.")
        return None

    # Supported Extensions
    valid_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
    
    # Diagnostic: Count valid files
    all_files = glob.glob(os.path.join(DOCS_DIR, "**/*.*"), recursive=True)
    actual_docs = [f for f in all_files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not actual_docs:
        st.warning(f"⚠️ No compatible documents found in /{DOCS_DIR}. Only PDF, DOCX, and TXT are supported.")
        return None

    try:
        with st.spinner(f"🌱 Indexing {len(actual_docs)} compatible files..."):
            # The 'silent_errors=True' and 'show_progress=True' help handle rogue files
            loader = DirectoryLoader(
                DOCS_DIR, 
                glob="**/*.*", # Look everywhere
                loader_cls=UnstructuredFileLoader,
                silent_errors=True, # Skip files it can't read
                recursive=True
            )
            
            # This loads only the documents that Unstructured can actually handle
            docs = loader.load()
            
            if not docs:
                st.error("❌ Failed to extract text from your documents. Check if they are password protected.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for the Eco-Education curriculum. Answer based ONLY on the provided context. Context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            return create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 5}),
                create_stuff_documents_chain(llm, prompt)
            )
    except Exception as e:
        st.error(f"Engine failure: {e}")
        return None

# --- 3. UI ---
st.title("🌱 Eco-Education Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_input = st.chat_input("Ask about the curriculum...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        api_key = st.secrets.get("OPENAI_API_KEY")
        chain = get_bot_chain(api_key)
        
        if chain:
            history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
            with st.spinner("Analyzing curriculum..."):
                response = chain.invoke({"input": user_input, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()
