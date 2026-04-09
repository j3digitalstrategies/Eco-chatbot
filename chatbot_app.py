import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_app():
    st.set_page_config(page_title="Eco-Chatbot", layout="wide")

    # Sidebar for Navigation
    st.sidebar.title("Navigation")
    user_role = st.sidebar.selectbox("Select Section", ["Home", "Student", "Parent/Teacher"])

    if user_role == "Student":
        st.title("Student Learning Portal")
        st.write("Welcome! Ask anything about ecology and the environment.")

        # Updated ChatPromptTemplate logic to fix the ValueError
        # Ensure that no single curly braces exist unless they are valid input variables
        system_template = (
            "You are a helpful assistant specialized in ecology. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "\n\n"
            "Context: {context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}"),
        ])

        # Initialize LLM and Chain components
        # Note: Ensure your Chroma DB and API Keys are correctly configured
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            
            # Placeholder for your retriever logic
            # retriever = vectorstore.as_retriever()

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Example Chain (Requires retriever to be defined)
            # rag_chain = (
            #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
            #     | prompt_template
            #     | llm
            #     | StrOutputParser()
            # )

            user_query = st.text_input("What would you like to learn today?")
            if user_query:
                # response = rag_chain.invoke(user_query)
                # st.markdown(response)
                st.info("Retriever logic and API keys must be active to generate live responses.")

        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")

    elif user_role == "Parent/Teacher":
        st.title("Pedagogy & Teaching Resources")
        kid_age = st.number_input("Enter the student's age:", min_value=3, max_value=18, value=10)
        st.write(f"Showing pedagogical resources tailored for age {kid_age}.")
        # Add pedagogy-specific content here

    else:
        st.title("Eco-Chatbot Home")
        st.write("Please select a section from the sidebar to begin.")

if __name__ == "__main__":
    run_app()
