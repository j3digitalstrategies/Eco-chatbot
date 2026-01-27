import streamlit as st
import os
import glob
from dotenv import load_dotenv

# LangChain & OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, TextLoader
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
        with st.spinner("🌱 Building the Curriculum Brain... This happens once."):
            documents = []
            all_files = glob.glob(os.path.join(DOCS_DIR, "**/*.*"), recursive=True)
            
            # Progress bar for visual feedback
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file_path in enumerate(all_files):
                ext = os.path.splitext(file_path)[1].lower()
                status_text.text(f"Processing: {os.path.basename(file_path)}")
                
                try:
                    if ext in [".docx", ".doc"]:
                        loader = UnstructuredWordDocumentLoader(file_path)
                    elif ext == ".pdf":
                        loader = PyPDFLoader(file_path)
                    elif ext == ".txt":
                        loader = TextLoader(file_path)
                    else:
                        continue
                    
                    documents.extend(loader.load())
                except Exception as e:
                    # Silently skip files that are corrupted or unreadable
                    continue
                
                progress_bar.progress((i + 1) / len(all_files))

            if not documents:
                st.error("Could not read any documents. Please check file formats.")
                return None
            
            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(documents)
            
            # Vector Store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=_api_key)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # LLM & Chain
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_api_key)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert on the Eco-Education curriculum. Use the context to answer. Context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            
            status_text.text("✅ Brain Ready! Ask me anything.")
            return rag_chain

    except Exception as e:
        st.error(f"Critical Error: {e}")
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
            with st.spinner("Searching..."):
                response = chain.invoke({"input": user_input, "chat_history": history})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
