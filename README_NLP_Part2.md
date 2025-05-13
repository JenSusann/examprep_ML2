# Pip install for the REG

## Overview and Description
1. **AI & LLMs**
```bash
pip install ollama #ollama – Interface for running and managing local LLMs via Ollama (used for local model inference).
pip install google-generativeai #google.generativeai – Google's Gemini model SDK for text and multimodal generation.
pip install openai #openai – Official OpenAI API client (used for ChatGPT, GPT-4, embeddings, etc.).
pip install groq #groq – SDK to access Groq’s ultra-fast inference API for running LLMs like LLaMA-3.
```

2. **Async + System Utilities**
```bash
pip install python-dotenv #pip install python-dotenv
pip install nest_asyncio #nest_asyncio – Allows nested asyncio loops (useful when using asyncio inside notebooks or environments that already run an event loop).
```

3. **PDF Reading and File Handling**
```bash
pip install PyPDF2 #PyPDF2 – Read and extract text from PDF files.
pip install glob2 #glob – Standard Python library for file matching; glob2 extends it. You don't need to install glob separately unless using extended functionality.
```

4. **LangChain Ecosystem**
```bash
pip install langchain #langchain – Core framework for chaining LLM interactions with tools like memory, prompt templates, retrievers, etc.
pip install langchain-google-genai #langchain-google-genai – Integrates Google’s Gemini with LangChain.
pip install langchain-community #langchain_community – Community-contributed integrations like FAISS, HuggingFaceEmbeddings, etc.
```

5. **Embeddings + Transformers**
```bash
pip install sentence-transformers #sentence_transformers – For creating high-quality sentence embeddings (e.g., for search and clustering).
pip install huggingface-hub #HuggingFaceEmbeddings – Used in LangChain to fetch embedding models from Hugging Face.
```

6. **Vector Store**
```bash
pip install faiss-cpu #faiss – Vector database for similarity search. Use faiss-cpu for CPU-only systems.
pip install pickle5 #pickle – Standard library, but pickle5 is sometimes used for extended features or compatibility in older environments.
```

7. **Visualization**
```bash
pip install matplotlib #matplotlib – Visualization library, used here possibly for plotting embeddings (e.g., after UMAP dimensionality reduction).
pip install umap-learn #umap.umap_ – Dimensionality reduction technique for visualizing high-dimensional embeddings (e.g., from sentence transformers).
```

8. **Others**
```bash
pip install weave #weave – Tool for observability and visual debugging of ML apps (optional, depending on your use).
pip install tqdm #tqdm – Progress bar utility for loops (useful when processing many documents).
```

## 📋 Import-to-Package Mapping

This table helps you understand which `pip install` command is required for each Python import used in this project.

| **Import** | **pip install** | **Description** |
|------------|------------------|-----------------|
| `import ollama` | `pip install ollama` | Local LLM runner and manager |
| `import google.generativeai as genai` | `pip install google-generativeai` | Access Google's Gemini AI models |
| `from openai import OpenAI` | `pip install openai` | OpenAI API client (GPT-4, embeddings, etc.) |
| `from groq import Groq` | `pip install groq` | Access Groq’s fast LLMs (e.g., LLaMA-3) |
| `from dotenv import load_dotenv` | `pip install python-dotenv` | Load environment variables from `.env` |
| `import nest_asyncio` | `pip install nest_asyncio` | Allow nested asyncio event loops |
| `import weave` | `pip install weave` | Visual observability for ML workflows |
| `from PyPDF2 import PdfReader` | `pip install PyPDF2` | Read and extract text from PDF files |
| `import glob` | *Standard Library* | File pattern matching (no install needed) |
| `from langchain.text_splitter import RecursiveCharacterTextSplitter` | `pip install langchain` | LangChain text splitter for chunking |
| `from langchain.text_splitter import SentenceTransformersTokenTextSplitter` | `pip install langchain` + `pip install sentence-transformers` | Token-aware text splitting |
| `from langchain_community.embeddings import HuggingFaceEmbeddings` | `pip install langchain-community` + `pip install huggingface-hub` | HuggingFace embeddings in LangChain |
| `from langchain_google_genai import ChatGoogleGenerativeA` | `pip install langchain-google-genai` | Gemini chat model in LangChain |
| `from langchain_community.vectorstores import FAISS` | `pip install langchain-community` + `pip install faiss-cpu` | FAISS vector search in LangChain |
| `from langchain_core.prompts import ChatPromptTemplate` | `pip install langchain` | LangChain core prompt structuring |
| `from langchain_core.messages import HumanMessage, AIMessage` | `pip install langchain` | Message types for conversations |
| `from langchain_core.output_parsers import StrOutputParser` | `pip install langchain` | Output parsing for string responses |
| `from langchain_core.runnables import RunnableBranch, RunnablePassthrough` | `pip install langchain` | Control flow in LangChain chains |
| `from sentence_transformers import SentenceTransformer` | `pip install sentence-transformers` | Embedding model loader |
| `from sentence_transformers import CrossEncoder` | `pip install sentence-transformers` | Scoring pairs of sentences |
| `import faiss` | `pip install faiss-cpu` | Vector similarity search |
| `import pickle` | *Standard Library* | Serialize and save objects |
| `import matplotlib.pyplot as plt` | `pip install matplotlib` | Create plots and charts |
| `import umap.umap_ as umap` | `pip install umap-learn` | Dimensionality reduction |
| `import numpy as np` | `pip install numpy` | Numeric computing and arrays |
| `import tqdm` | `pip install tqdm` | Progress bars in loops |
| `import random` | *Standard Library* | Random number utilities |
