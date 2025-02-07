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
    /* Main background and text color */
    .stApp {
        background-color: #f4f4f9;
        color: #333333;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding: 20px;
    }
    
    /* Text input styling */
    .stTextInput textarea {
        color: #333333 !important;
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    /* Select box styling */
    .stSelectbox div[data-baseweb="select"] {
        color: #333333 !important;
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox svg {
        fill: #333333 !important;
    }
    
    .stSelectbox option {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    
    /* Dropdown menu items */
    div[role="listbox"] div {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #007BFF !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: #0056b3 !important;
    }
    
    /* Header and caption styling */
    .stMarkdown h1 {
        color: #007BFF;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .stMarkdown h2 {
        color: #333333;
        font-size: 1.5em;
        font-weight: bold;
    }
    
    .stMarkdown p {
        color: #666666;
        font-size: 1em;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are a sophisticated AI research analyst with expertise in document analysis and information extraction. Your role is to:

1. Analyze the provided context thoroughly
2. Extract relevant information precisely
3. Synthesize clear, accurate responses

Guidelines:
- Focus on information directly supported by the context
- Maintain academic/professional tone
- Cite specific sections when relevant
- Acknowledge any ambiguities or limitations in the source material
- Structure complex responses for clarity

Query: {user_query}
Context: {document_context}
Response (be concise and factual, max 3 sentences):
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
st.markdown("### Your Premier Intelligent Document Assistant")
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