# 🤖 AI Assistant Projects

This repository contains two AI-powered applications: DeepSeek Code Companion and DocuMind AI. Both applications leverage the power of LangChain and Ollama to provide intelligent assistance.

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

### DeepSeek Code Companion (app.py)
An AI-powered coding assistant that helps with:
- Python code generation and debugging
- Code documentation
- Best practices suggestions
- Solution design

### DocuMind AI (ragDeep_faiss.py)
A document analysis tool that:
- Processes PDF documents
- Answers questions about document content
- Provides intelligent document summarization
- Uses FAISS for efficient document search

## 💻 Requirements

- Python 3.8+
- Ollama (with deepseek-r1:1.5b model)
- 8GB+ RAM recommended
- PDF documents for DocuMind AI

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and Configure Ollama**
   - Download Ollama from [ollama.ai](https://ollama.ai)
   - Install the deepseek model:
     ```bash
     ollama pull deepseek-r1:1.5b
     ```

## 🚀 Usage

### Running DeepSeek Code Companion
```bash
streamlit run app.py
```

### Running DocuMind AI
```bash
streamlit run ragDeep_faiss.py
```

Features:
- Interactive code assistance
- Code execution environment
- Export chat history
- Multiple model selection

## ✨ Features

### DeepSeek Code Companion
- 🐍 Python Expert
- 🐞 Debugging Assistant
- 📝 Code Documentation
- 💡 Solution Design
- 💻 Code Execution Environment
- 📤 Chat Export (PDF/Text)

### DocuMind AI
- 📄 PDF Processing
- 🔍 Semantic Search
- 💬 Interactive Q&A
- 📊 Document Analysis
- 🚀 Fast Response Times

## ⚠️ Troubleshooting

### Common Issues
1. **Ollama Connection Error**
   - Ensure Ollama is running locally
   - Check if the model is properly installed
   ```bash
   ollama list
   ```

2. **Memory Issues**
   - Reduce chunk size in document processing
   - Close other memory-intensive applications

3. **PDF Processing Issues**
   - Ensure PDF is text-based, not scanned
   - Check PDF file permissions

### Support
For additional support:
- Check the [Issues](link-to-issues) section
- Contact: [Your Contact Information]


## 🙏 Acknowledgments
- [Ollama](https://ollama.ai)
- [LangChain](https://python.langchain.com/)
- [Streamlit](https://streamlit.io/)