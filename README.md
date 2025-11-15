# AmbedkarGPT - Intern Task

A command-line Q&A system that answers questions based on Dr. B.R. Ambedkar's speech excerpt using Retrieval-Augmented Generation (RAG).

## Overview

This project implements a RAG pipeline using LangChain to:
- Load and process text from Dr. Ambedkar's speech
- Create embeddings and store them in a local vector database
- Retrieve relevant context based on user questions
- Generate accurate answers using a local LLM

## Tech Stack

- **Framework**: LangChain
- **Vector Database**: ChromaDB (local, no API keys required)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Ollama with Mistral (100% free, local)
- **Language**: Python 3.8+

## Prerequisites

### 1. Python Installation
Ensure you have Python 3.8 or higher installed:
```bash
python --version
```

### 2. Ollama Installation
Install Ollama to run the LLM locally:

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### 3. Pull the Llama Model
After installing Ollama, pull the Mistral model:
```bash
ollama pull mistral
```

Verify Ollama is running:
```bash
ollama list
```

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 2: Create a Virtual Environment
**Using venv:**
```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n ambedkargpt python=3.10
conda activate ambedkargpt
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── app.py              # Main RAG pipeline implementation
├── speech.txt           # Dr. Ambedkar's speech excerpt
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore file (optional)
```

## 💻 Usage

### Running the Q&A System

1. Ensure Ollama is running in the background
2. Make sure `speech.txt` is in the same directory as `app.py`
3. Run the program:

```bash
python app.py
```

### Example Interaction

```
Loading embeddings model...
Creating vector store...
Initializing Ollama LLM...

AmbedkarGPT is ready.
Ask questions based on the speech. Type 'exit' to quit.

Question: What is the real remedy according to the text?
Answer: According to the text, the real remedy is to destroy the belief in the sanctity of the shastras.

Question: What is the real enemy?
Answer: The real enemy is the belief in the shastras.

Question: exit
Exiting...
```

## 🔧 How It Works

### RAG Pipeline Steps

1. **Document Loading**: Loads `speech.txt` using LangChain's TextLoader
2. **Text Chunking**: Splits the document into 500-character chunks with 50-character overlap
3. **Embedding Generation**: Creates vector embeddings using HuggingFace's all-MiniLM-L6-v2 model
4. **Vector Storage**: Stores embeddings in ChromaDB for efficient similarity search
5. **Question Processing**: Takes user input and retrieves relevant chunks
6. **Answer Generation**: Sends context and question to Mistral via Ollama for answer generation

### Key Components

- **TextLoader**: Loads the speech text file
- **CharacterTextSplitter**: Breaks text into manageable chunks
- **HuggingFaceEmbeddings**: Converts text to vector embeddings
- **Chroma**: Local vector database for semantic search
- **Ollama**: Runs mistral LLM locally
- **LCEL Chain**: Orchestrates the RAG pipeline using LangChain Expression Language

## Contact

For questions or issues related to this assignment, contact:
- **Email**: devarajan8.official@gmail.com

## 📄 License

This project is created as part of an internship assignment for Kalpit Pvt Ltd.

---

**Note**: This is a prototype demonstration of RAG concepts and is not intended for production use.
