import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from io import BytesIO
import boto3
from PyPDF2 import PdfReader
from langchain.schema import Document
 
# App Title
st.title("Knowledge Management Chatbot")
 
# Initialize session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
 
aws_access_key = ''
aws_secret_key = ''
region_name = 'us-east-1'
 
s3 = boto3.client('s3', region_name=region_name, aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
 
bucket_name = 'docsummhhfl'
object_key = 'hrpolicytest.pdf'
 
st.write(f"Attempting to access file in Bucket: {bucket_name}, Key: {object_key}")
 
def stream_file_from_s3(bucket_name, object_key):
    try:
        file_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        return BytesIO(file_obj['Body'].read())
    except Exception as e:
        st.error("Error streaming file from S3")
        return None
 
pdf_file = stream_file_from_s3(bucket_name, object_key)
 
if pdf_file:
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    docs = []
 
    # Load PDF and create Document instances
    for page in reader.pages:
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text)) 
 
    st.success("Loaded")
 
    # Text Splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=15)
 
    # Combine page contents into a single string for splitting
    combined_text = "".join([doc.page_content for doc in docs])
    split_texts = text_splitter.split_text(combined_text)
    documents = [Document(page_content=text) for text in split_texts]
 
    # Debugging print statement to show the contents of documents
    print("Documents after splitting:", documents)  # This line is for debugging
 
    # Initialize embeddings and LLM
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatGroq(groq_api_key="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", model_name='llama3-70b-8192', temperature=0, top_p=0.2)
 
    # Vector database storage
    vector_db = FAISS.from_documents(documents, hf_embedding)
 
    # Craft ChatPrompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are a Bandhan Life Insurance specialist. Answer the queries from an insurance specialist perspective who wants to resolve customer queries as asked.
    Answer the following questions based only on the provided context, previous responses, and the uploaded documents.
    Think step by step before providing a detailed answer.
    Wherever required, answer in a point-wise format.
    Do not answer any unrelated questions which are not in the provided documents, please be careful on this.
    I will tip you with a $1000 if the answer provided is helpful.
 
    <context>
    {context}
    </context>
    Conversation History:
    {chat_history}
 
    Question: {input}
    """)
 
    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)
 
    # Retriever from vector store
    retriever = vector_db.as_retriever()
 
    # Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
 
    # Chat interface
    user_question = st.text_input("Ask a question about the relevant document", key="input")
 
    if user_question:
        # Build conversation history
        conversation_history = ""
        for chat in st.session_state['chat_history']:
            conversation_history += f"You: {chat['user']}\nBot: {chat['bot']}\n"
 
        # Get response from the retrieval chain with context
        response = retrieval_chain.invoke({
            "input": user_question,
            "chat_history": conversation_history
        })
 
        # Add the user's question and the model's response to chat history
        st.session_state.chat_history.append({"user": user_question, "bot": response['answer']})
 
    # Display chat history with a conversational format
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #DCF8C6;'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #ECECEC; margin-top: 5px;'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
