import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
import time

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")

# Replace InMemoryVectorStore with FAISS
class FAISSVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []

    def add_documents(self, document_chunks):
        # Convert document chunks to embeddings
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in document_chunks])
        
        # Initialize FAISS index if it doesn't exist
        if self.index is None:
            dimension = len(embeddings[0])  # Get the dimension of the embeddings
            self.index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity search
        
        # Add embeddings to the FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(document_chunks)  # Store the documents for retrieval

    def similarity_search(self, query, k=5):
        # Convert query to embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search for the k most similar documents
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        # Retrieve the corresponding documents
        return [self.documents[i] for i in indices[0]]

# Initialize FAISS vector store
DOCUMENT_VECTOR_DB = FAISSVectorStore(EMBEDDING_MODEL)

LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def save_uploaded_file(uploaded_file):
    # Define the directory where you want to save the file
    directory = 'document_store/pdfs'
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the full file path
    file_path = os.path.join(directory, uploaded_file.name)
    
    # Save the uploaded file
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        add_start_index=False
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    print("Starting to index documents...")
    start_time = time.time()
    try:
        DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    except Exception as e:
        print(f"Error indexing documents: {e}")
    print(f"Finished indexing documents in {time.time() - start_time} seconds.")

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_documents = context_documents[:5]
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    
    # Measure time taken to load documents
    start_time = time.time()
    raw_docs = load_pdf_documents(saved_path)
    print(f"Loading documents took {time.time() - start_time} seconds")
    
    # Measure time taken to chunk documents
    start_time = time.time()
    processed_chunks = chunk_documents(raw_docs)
    print(f"Chunking documents took {time.time() - start_time} seconds")
    
    # Measure time taken to index documents
    start_time = time.time()
    try:
        index_documents(processed_chunks)
    except Exception as e:
        print(f"Error indexing documents: {e}")
    print(f"Indexing documents took {time.time() - start_time} seconds")
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Add user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Analyzing document..."):
            # Measure time taken to find related documents
            print("Starting to find related documents...")
            start_time = time.time()
            try:
                relevant_docs = find_related_documents(user_input)
            except Exception as e:
                print(f"Error finding related documents: {e}")
            print(f"Finding related documents took {time.time() - start_time} seconds")
            print("Finished finding related documents.")
            
            # Measure time taken to generate answer
            start_time = time.time()
            ai_response = generate_answer(user_input, relevant_docs)
            print(f"Generating answer took {time.time() - start_time} seconds")
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"]) 