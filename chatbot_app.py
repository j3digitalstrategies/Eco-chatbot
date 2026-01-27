import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CONFIGURATION ---
load_dotenv()
# This folder must exist in your GitHub repo
DOCS_DIR = "curriculum_docs" 

st.set_page_config(page_title="Eco-Chatbot", layout="wide", page_icon="🌱")

# --- 2. THE ENGINE (The "Crawl-Everything" Loader) ---
@st.cache_resource
def get_bot_chain(_api_key):
    # Verify the folder exists
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        st.error(f"❌ Looking for folder: '{DOCS_DIR}'. Ensure it's in your GitHub repo and contains files.")
        return None

    try:
        with st.spinner("🌱 Indexing all curriculum subfolders..."):
            # glob="**/*.*" tells it to go as deep as needed into subfolders
            loader = DirectoryLoader(
                DOCS_DIR, 
                glob="**/*.*", 
                loader_cls=UnstructuredFileLoader
            )
            docs = loader.load()
            
            # Break the text into chunks so the AI can pinpoint specific details
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # Create the searchable vector database in the app's memory
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            # Initialize the AI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for the Eco-Education curriculum. Use the following context to answer precisely. Context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            return create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 5}),
                create_stuff_documents_chain(llm, prompt)
            )
    except Exception as e:
        st.error(f"Engine initialization failure: {e}")
        return None

# --- 3. CHAT INTERFACE ---
st.title("🌱 Eco-Education Assistant")
st.info(f"Connected to curriculum files in /{DOCS_DIR}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
user_input = st.chat_input("Ask a question about the curriculum...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        api_key = st.secrets.get("OPENAI_API_KEY")
        chain = get_bot_chain(api_key)
        
        if chain:
            # Reconstruct chat history for the AI
            history = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                else:
                    history.append(AIMessage(content=m["content"]))
            
            with st.spinner("Searching through all modules..."):
                response = chain.invoke({"input": user_input, "chat_history": history})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    # Rerun to update the chat display
    st.rerun()
