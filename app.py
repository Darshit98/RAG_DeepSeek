import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from fpdf import FPDF
import json
from datetime import datetime
import os

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    # Export options
    if len(st.session_state.message_log) > 1:  # Only show if there are messages
        st.markdown("### Export Chat")
        export_format = st.selectbox(
            "Choose Format",
            ["Select Format", "Text", "PDF"],
            key="export_format"
        )
        
        if export_format != "Select Format":
            if st.button(f"Export as {export_format}"):
                try:
                    if export_format == "Text":
                        filename = export_as_text(st.session_state.message_log)
                    else:  # PDF
                        filename = export_as_pdf(st.session_state.message_log)
                    
                    # Provide download link
                    with open(filename, "rb") as f:
                        st.download_button(
                            label=f"Download {export_format} file",
                            data=f,
                            file_name=filename,
                            mime="text/plain" if export_format == "Text" else "application/pdf"
                        )
                    # Clean up the file after download
                    os.remove(filename)
                except Exception as e:
                    st.error(f"Error exporting chat: {str(e)}")
        st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    if st.button("Clear Chat History"):
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
        st.rerun()


# initiate the chat engine

llm_engine=ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",

    temperature=0.3

)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an expert AI coding assistant with the following specialized capabilities:

    1. 🐍 Python Expert:
    - Write clean, efficient, and Pythonic code
    - Provide best practices and modern Python features
    - Explain Python-specific concepts and implementations

    2. 🐞 Debugging Assistant:
    - Analyze code for potential bugs and edge cases
    - Add strategic print statements and logging
    - Suggest debugging approaches and tools
    - Provide error handling recommendations

    3. 📝 Code Documentation:
    - Write clear docstrings and comments
    - Follow PEP documentation standards
    - Include usage examples and type hints
    - Explain complex logic and design decisions

    4. 💡 Solution Design:
    - Propose scalable and maintainable architectures
    - Consider performance implications
    - Suggest alternative approaches when relevant
    - Break down complex problems into manageable steps

    Always provide concise, practical solutions with explanations. Include error handling where appropriate. 
    Format code blocks with proper syntax highlighting. If the solution is complex, break it down into steps.
    Always respond in English and prioritize clarity and maintainability in your solutions."""
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        try:
            # Escape curly braces in the content by doubling them
            escaped_content = msg["content"].replace("{", "{{").replace("}", "}}")
            if msg["role"] == "user":
                prompt_sequence.append(HumanMessagePromptTemplate.from_template(escaped_content))
            elif msg["role"] == "ai":
                prompt_sequence.append(AIMessagePromptTemplate.from_template(escaped_content))
        except Exception as e:
            print(f"Error processing message: {e}")
            continue
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()

def export_as_text(messages):
    """Export chat history as a text file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        for msg in messages:
            role = "DeepSeek" if msg["role"] == "ai" else "User"
            f.write(f"{role}: {msg['content']}\n\n")
    
    return filename

def export_as_pdf(messages):
    """Export chat history as a PDF file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.pdf"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="DeepSeek Code Companion - Chat History", ln=True, align='C')
    pdf.ln(10)
    
    # Add content
    pdf.set_font("Arial", size=12)
    for msg in messages:
        role = "DeepSeek" if msg["role"] == "ai" else "User"
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt=f"{role}:", ln=True)
        pdf.set_font("Arial", size=12)
        
        # Split content into lines to handle long messages
        content = msg["content"]
        lines = [content[i:i+90] for i in range(0, len(content), 90)]
        for line in lines:
            pdf.cell(200, 10, txt=line, ln=True)
        pdf.ln(5)
    
    pdf.output(filename)
    return filename