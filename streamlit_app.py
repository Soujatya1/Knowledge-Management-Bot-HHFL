import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# App Title
st.title("Knowledge Management Chatbot with History")

# Persistent conversation history using session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# File upload functionality
uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

if uploaded_file is not None:
    with open("HRPolicy_Test.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("HRPolicy_Test.pdf")
    docs = loader.load()

    # Text Splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=15)
    documents = text_splitter.split_documents(docs)

    # Initialize HuggingFace embeddings
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Initialize the ChatGroq LLM
    llm = ChatGroq(groq_api_key="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", 
                   model_name='llama3-70b-8192', 
                   temperature=0, 
                   top_p=0.2)

    # Vector database storage
    vector_db = FAISS.from_documents(docs, hf_embedding)

    # Create memory to store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Chat prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a Knowledge Management specialist.
    Answer the following questions based only on the provided context and the uploaded documents.
    Think step by step before providing a detailed answer.
    Wherever required, answer in a point-wise format.
    Do not answer any unrelated questions which are not in the provided documents.
    <context>
    {context}
    </context>
    Question: {input}
    {chat_history}""")

    # Create a chain to handle documents
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retriever from vector database
    retriever = vector_db.as_retriever()

    # Create a retrieval chain with memory
    retrieval_chain = create_retrieval_chain(retriever, document_chain, memory=memory)

    # Continuous Chat Interface
    user_question = st.text_input("Ask a question about the document")

    if user_question:
        # Retrieve and process the answer
        response = retrieval_chain.invoke({"input": user_question})
        answer = response['answer']
        
        # Store conversation history in session state
        st.session_state.chat_history.append({"user": user_question, "bot": answer})

    # Display conversation history
    with st.container():
        for chat in st.session_state.chat_history:
            st.write(f"You: {chat['user']}")
            st.write(f"Bot: {chat['bot']}")
