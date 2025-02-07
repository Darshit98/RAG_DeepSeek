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

# Initialize session state
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek, your AI coding assistant. How can I assist you with your coding challenges today? 💻"}]

def export_as_text(messages):
    """Export chat history as a text file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.txt"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for msg in messages:
                role = "DeepSeek" if msg["role"] == "ai" else "User"
                # Clean the content of emojis and special characters
                content = msg["content"].encode('ascii', 'ignore').decode('ascii')
                f.write(f"{role}:\n{content}\n\n")
        return filename
    except Exception as e:
        st.error(f"Error in text export: {str(e)}")
        return None

def export_as_pdf(messages):
    """Export chat history as a PDF file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.pdf"
    
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Add title using default font
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "DeepSeek Code Companion - Chat History", ln=True, align='C')
        pdf.ln(10)
        
        # Add content
        for msg in messages:
            role = "DeepSeek" if msg["role"] == "ai" else "User"
            # Clean the content of emojis and special characters
            content = msg["content"].encode('ascii', 'ignore').decode('ascii')
            
            # Add role
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"{role}:", ln=True)
            
            # Add message content
            pdf.set_font("Arial", "", 10)
            # Split content into lines to handle long messages
            lines = [content[i:i+90] for i in range(0, len(content), 90)]
            for line in lines:
                pdf.multi_cell(0, 10, line)
            pdf.ln(5)
        
        pdf.output(filename)
        return filename
    except Exception as e:
        st.error(f"Error in PDF export: {str(e)}")
        return None

# Custom CSS styling
st.markdown("""
<style>
    /* Main background and text color */
    .main {
        background-color: #f0f2f6;
        color: #333333;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Text input styling */
    .stTextInput textarea {
        color: #333333 !important;
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 5px !important;
    }
    
    /* Select box styling */
    .stSelectbox div[data-baseweb="select"] {
        color: #333333 !important;
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 5px !important;
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
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: #45a049 !important;
    }
    
    /* Header and caption styling */
    .stMarkdown h1 {
        color: #4CAF50;
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
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
st.title("🧠 DeepSeek AI Assistant")
st.caption("🚀 Your Intelligent Coding Partner with Debugging Skills")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Model Selected",
        ["deepseek-r1:1.5b"],
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
                    filename = None
                    if export_format == "Text":
                        filename = export_as_text(st.session_state.message_log)
                    else:  # PDF
                        filename = export_as_pdf(st.session_state.message_log)
                    
                    if filename:
                        # Provide download link
                        with open(filename, "rb") as f:
                            file_data = f.read()
                            st.download_button(
                                label=f"Download {export_format} file",
                                data=file_data,
                                file_name=filename,
                                mime="text/plain" if export_format == "Text" else "application/pdf"
                            )
                        # Clean up the file after download
                        os.remove(filename)
                    else:
                        st.error("Failed to generate export file")
                except Exception as e:
                    st.error(f"Error exporting chat: {str(e)}")
        st.divider()
    if st.button("Clear Chat History"):
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek, your AI coding assistant. How can I assist you with your coding challenges today? 💻"}]
        st.rerun()
    
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    
    # Footer at the bottom of the sidebar
    st.markdown("<footer style='text-align: center; margin-top: 100px;'>© Darshit Shah</footer>", unsafe_allow_html=True)


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

# Add this function to execute code
def execute_code(code):
    try:
        # Check for input() usage
        if "input(" in code:
            return "Warning: The use of input() is not supported in this environment. Please modify your code to use direct assignments or Streamlit widgets for input."

        # Redirect stdout to capture print statements
        import io
        import contextlib

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {})
        
        return output.getvalue()
    except Exception as e:
        return str(e)

# Add this section in your Streamlit app
st.markdown("### 🛠️ Code Execution Environment")
code_input = st.text_area("Enter your Python code here:", height=200)

if st.button("Run Code"):
    if code_input.strip():
        with st.spinner("Running your code..."):
            result = execute_code(code_input)
            st.text_area("Output", result, height=200)
    else:
        st.warning("Please enter some code to run.")
