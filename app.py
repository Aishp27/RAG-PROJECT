# import os
# from dotenv import load_dotenv

# import streamlit as st

# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# from utils.prompts import SYSTEM_PROMPT

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# VECTOR_DB_PATH = "vectorstore"


# @st.cache_resource

# def load_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )


# @st.cache_resource

# def load_vectorstore():
#     embeddings = load_embeddings()

#     vectorstore = FAISS.load_local(
#         VECTOR_DB_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     return vectorstore


# @st.cache_resource

# def load_llm():
#     llm = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama3-8b-8192",
#         temperature=0.3,
#         max_tokens=1024
#     )

#     return llm



# def create_qa_chain():
#     vectorstore = load_vectorstore()

#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 4}
#     )

#     llm = load_llm()

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )

#     return qa_chain

#     main()


import os
from dotenv import load_dotenv
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

VECTOR_DB_PATH = "vectorstore"


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


@st.cache_resource
def load_llm():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024
    )

    return llm

def create_qa_chain():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def main():
    st.set_page_config(page_title="Movie RAG Chatbot")

    st.title("🎬 Movie RAG Chatbot")

    query = st.text_input("Ask a movie question:")

    if query:
        qa_chain = create_qa_chain()

        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": query})

        st.subheader("Answer")
        st.write(response["result"])


if __name__ == "__main__":
    main()
