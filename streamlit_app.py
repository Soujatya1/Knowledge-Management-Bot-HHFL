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
import sentence_transformers
from langchain.embeddings import HuggingFaceEmbeddings

#App Title
st.title("Knowledge Management Chatbot")

uploaded_file = st.file_uploader("Upload a file", type = ["pdf", "doc"])

if uploaded_file is not None:
  with open("HRPolicy_Test.pdf", "wb") as f:
    f.write(uploaded_file.getbuffer())
  st.success("File uploaded successfully!")

  #Load PDF
  loader = PyPDFLoader("HRPolicy_Test.pdf")
  docs = loader.load()

  #Text Splitting into chunks
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 15)
  documents = text_splitter.split_documents(docs)

  hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


  llm = ChatGroq(groq_api_key = "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", model_name = 'llama3-70b-8192', temperature = 0, top_p = 0.2)

  #Vector database storage
  vector_db = FAISS.from_documents(docs, hf_embedding)

  #Craft ChatPrompt Template
  prompt = ChatPromptTemplate.from_template("""
  You are an AI Knowledge Management specialist. Answer the following questions based only on the provided context and the uploaded documents.
  Think step by step before providing a detailed answer. Also, provide the page number of the document from where the answers are provided. Wherever required, answer in a point-wise format.
  I will tip you with a $1000 if the answer provided is helpful.
  <context>
  {context}
  </context>
  Question: {input}""")

  #Stuff Document Chain Creation
  document_chain = create_stuff_documents_chain(llm, prompt)

  #Retriever from Vector store
  retriever = vector_db.as_retriever()

  #Create a retrieval chain
  retrieval_chain = create_retrieval_chain(retriever,document_chain)

  #User input
  user_question = st.text_input("Ask a question about the relevant document")

  if user_question:
    response = retrieval_chain.invoke({"input": user_question})
    st.write(response['answer'])
