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

This table helps you understand which `pip install` or `!pip install` command is required for each Python import used in this project.

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

## 🛠️ Troubleshooting & Forced Installation

If you encounter errors like: ModuleNotFoundError: No module named 'xyz'

Follow these steps to fix the issue:
---

### 1. Install the Missing Package

```bash
pip install xyz
```

### 2. Force Reinstallation
```bash
pip install --force-reinstall xyz
```

### 3. Upgrade the Package
```bash
pip install --upgrade xyz
```

### 4. Install a Specific Version
```bash
pip install xyz==1.2.3
```

### 5. Uninstall and Reinstall
```bash
pip uninstall xyz
pip install xyz
```
---

# .env Datei

Für API-Schlüssel usw.

```env
OPENAI_API_KEY=dein_openai_api_key
GOOGLE_API_KEY=dein_google_api_key
GROQ_API_KEY=dein_groq_api_key
```
---
## Laden der Variablen im Code

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Lädt die Variablen aus .env
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
```

# LLM_Calls

## Google
```python
import google.generativeai as genai
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel(
    "models/gemini-2.0-flash",
    system_instruction="You are a cat. Your name is Neko.",
)

chat = model.start_chat()
response = chat.send_message("Good day fine chatbot")
print(response.text)
```
---
## Openai
```python
from openai import OpenAI
client = OpenAI(api_key=openai_api_key)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)
```
---
## Groq
```python
from groq import Groq

client = Groq(api_key=groq_api_key)

llm = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI Assistant. You explain ever \
            topic the user asks as if you are explaining it to a 5 year old"
        },
        {
            "role": "user",
            "content": "What are Black Holes?",
        }
    ],
    model="mixtral-8x7b-32768",
)

print(llm.choices[0].message.content)
```
# Deepseek
https://github.com/zhaw-iwi/LLM-intro-solution/blob/main/deepseek_playground.ipynb 

## GPU in Colab aktivieren 
Menü: Laufzeit (Runtime) → Laufzeittyp ändern (Change runtime type)
Wähle GPU unter "Hardwarebeschleuniger"
---
### CUBA und ollama

![Deepseek](img/Fehler_Deepseek.png)

kann wie folgt gelöst werden:

```bash
!nvidia-smi #prüft ob GPU verfügbar ist
!apt-get install -y pciutils lshw # installiert fehlende tools
!ollama run llama2 # testen ob ollama die GPU erkennt
!curl https://ollama.ai/install.sh | sh
!sudo apt install -y cuda-drivers
```
#### cuda drivers laden & installieren

```bash
!echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
!sudo apt-get update && sudo apt-get install -y cuda-drivers

# Kontrollieren ob cuda drivers funktionieren
import os

# Set LD_LIBRARY_PATH so the system NVIDIA library
os.environ.update({'LD_LIBRARY_PATH': '/usr/lib64-nvidia'})
```
#### Start Ollama

```bash
!nohup ollama serve & #nohup is a command line expression that prevents a process from stopping after exiting the terminal.
```

#### Download Model

```bash
!ollama pull deepseek-r1:7b # Deepseek 
!ollama pull llama2 #llama2
!ollama pull deepseek-coder:6.7b # Deepseek -> wenn 7b nicht verfügbar

```
#### pip install ollama in Python
```bash
!pip install ollama
```
#### Beispiel request
```python
import ollama
response = ollama.chat(model='deepseek-r1:7b', messages=[
  {
    'role': 'user',
    'content': "How many r's are in a strawberry?",
  },
])
print(response['message']['content'])
```

# RAG
## rag with gemini
https://github.com/zhaw-iwi/rag-with-vector-solution/blob/main/rag_wit_gemini.ipynb

### Import
```python
# Standard library imports
import os  # For interacting with the operating system, e.g., file paths
import asyncio  # For managing asynchronous tasks

# Third-party library imports
from dotenv import load_dotenv  # For loading environment variables from a .env file
from PyPDF2 import PdfReader  # For reading PDF files
import tqdm  # For displaying progress bars in loops

# LangChain imports - Core functionality
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable chunks
from langchain.prompts import PromptTemplate  # For defining and managing prompt templates
from langchain.chains.combine_documents import create_stuff_documents_chain  # For combining retrieved documents into a coherent chain
from langchain.globals import set_debug  # For enabling debug mode in LangChain

# LangChain - Google Generative AI integrations
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For generating embeddings using Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat-based interactions with Google Generative AI

# LangChain - Vector store
from langchain_community.vectorstores import FAISS  # For storing and retrieving embeddings using the FAISS library

# LangChain - Advanced prompt management and messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # For creating structured chat prompts
from langchain_core.messages import HumanMessage, AIMessage  # For handling human and AI messages
from langchain_core.output_parsers import StrOutputParser  # For parsing string outputs from models
from langchain_core.runnables import RunnableBranch  # For creating branches in the chain of execution

from typing import Dict
from langchain_core.runnables import RunnablePassthrough

import nest_asyncio
import weave
nest_asyncio.apply()
```
### Pfad anpassen
```python
# Beliebieger Dateityp 
from google.colab import files
uploaded = files.upload()  # Beliebiger Dateityp z.B. PDF, CSV, JPG, PNG, TXT, JSON, etc. hochladen
# PDF
import fitz 
pdf_path = file_path
# CSV
import pandas as pd
csv_path = file_path
# JPG, PNG
from PIL import Image
import matplotlib.pyplot as plt
image_path = file_path
# TXT
txt_path = file_path 
# JSON
import json
json_path = file_path

# Direkter File import
pdf_path = "/content/*.pdf"
csv_path = "/content/*.csv"
image_path = "/content/*.jpg"
txt_path = "/content/*.txt"
json_path = "/content/*.json"
```
### Daten auslesen
```python
# PDF
from PyPDF2 import PdfReader

pdf_path = "data/nihms-1901028.pdf"

with open(pdf_path, "rb") as file:
    reader = PdfReader(file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

text

# CSV
import pandas as pd

csv_path = "data/daten.csv"

with open(csv_path, "r", encoding="utf-8") as file:
    df = pd.read_csv(file)

df.head()

# TXT
txt_path = "data/notizen.txt"

with open(txt_path, "r", encoding="utf-8") as file:
    text = file.read()

text

# JSON
import json

json_path = "data/config.json"

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

data

#PNG, JPG
from PIL import Image
import matplotlib.pyplot as plt

image_path = "data/bild.jpg"

with open(image_path, "rb") as file:
    img = Image.open(file)
    img.load()

plt.imshow(img)
plt.axis('off')
plt.show()

# WAV
from scipy.io import wavfile

audio_path = "data/audio.wav"

with open(audio_path, "rb") as file:
    rate, data = wavfile.read(file)

rate, data.shape
```
### Text aufteilen
```python
# PDF
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = "data/nihms-1901028.pdf"

with open(pdf_path, "rb") as file:
    reader = PdfReader(file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))

# CSV nur Textspalten
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

csv_path = "data/daten.csv"

with open(csv_path, "r", encoding="utf-8") as file:
    df = pd.read_csv(file)

# Kombiniere alle Textspalten zu einem großen String
text = "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1))

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))

# TXT
txt_path = "data/notizen.txt"
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open(txt_path, "r", encoding="utf-8") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))

# JSON nur textuelle Inhalte
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

json_path = "data/config.json"

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Konvertiere alles rekursiv zu String
def extract_text_from_json(obj):
    if isinstance(obj, dict):
        return " ".join([extract_text_from_json(v) for v in obj.values()])
    elif isinstance(obj, list):
        return " ".join([extract_text_from_json(i) for i in obj])
    else:
        return str(obj)

text = extract_text_from_json(data)

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))

# Bild OCR erforderlich
!apt install tesseract-ocr
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter

image_path = "data/bild.jpg"

with open(image_path, "rb") as file:
    img = Image.open(file)
    img.load()

# OCR: Text aus Bild extrahieren
text = pytesseract.image_to_string(img)

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))
```
### Speicher in Vector Store

```python
# Allgemein
!pip install langchain faiss-cpu google-generativeai PyPDF2 pytesseract Pillow pandas whisper

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings

# Google_API_KEY laden
# PDF
from PyPDF2 import PdfReader

pdf_path = "data/nihms-1901028.pdf"

with open(pdf_path, "rb") as file:
    reader = PdfReader(file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local("faiss_index_pdf")

# CSV
import pandas as pd

csv_path = "data/daten.csv"

with open(csv_path, "r", encoding="utf-8") as file:
    df = pd.read_csv(file)

text = "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1))

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local("faiss_index_csv")

# TXT
txt_path = "data/notizen.txt"

with open(txt_path, "r", encoding="utf-8") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local("faiss_index_txt")

# JSON
import json

json_path = "data/config.json"

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

def extract_text_from_json(obj):
    if isinstance(obj, dict):
        return " ".join([extract_text_from_json(v) for v in obj.values()])
    elif isinstance(obj, list):
        return " ".join([extract_text_from_json(i) for i in obj])
    else:
        return str(obj)

text = extract_text_from_json(data)

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local("faiss_index_json")

# Bild (OCR)
from PIL import Image
import pytesseract

image_path = "data/bild.jpg"

with open(image_path, "rb") as file:
    img = Image.open(file)
    img.load()

text = pytesseract.image_to_string(img)

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local("faiss_index_image")

```

### Weave Tracking
#### Vorbereitung & Login Wandb
https://wandb.ai/authorize

1. Neues Projekt erstellen -> User -> Profil -> Projects -> Create new
2. Quickstart Menu folgen 

![wandb](img/WANDB.png)

```bash
pip install wandb
wandb login
# API Key kopieren und eingeben wenn aufgefordert
```
Möglicher Fehler
```bash
weave version 0.51.44 is available! To upgrade, please run:
$ pip install weave --upgrade
```
Update
```bash
!pip install --upgrade weave

```

Erhaltenen Key in Command Line kopieren

Initallisiert Projekt

```python
weave.init("medical-data-chatbot") #Projektname ändern
```
### Retriever
```python
# PDF
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What does the PDF say about metabolic risk factors?")
print(docs)

# CSV
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What are the top 3 countries by GDP in the data?")
print(docs)

# TXT
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What key points does this text mention about nutrition?")
print(docs)

# JSON
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What config settings mention logging or error handling?")
print(docs)

# Bild
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is written on the second form page?")
print(docs)
```
Du kannst für jeden Dateityp genau diese drei Arten von Fragen ausprobieren:
- Relevante Frage → Inhalt direkt enthalten.
- Vage/unklare Frage → z. B. "How does this relate to performance?"
- Unpassende Frage → z. B. "What are the moon phases in August?"

### Question-Answering System

```python
# Define the template for answering user questions based on a provided context
system_template = """ 
Answer the users question based on the below context:
<context> {context} </context>
Say that you don't know the answer if you the context is not relevant to the question.
"""
# The system_template string specifies how the generative model should process the retrieved context.

# Create a prompt template for the question-answering system
question_answering_prompt = PromptTemplate(template=system_template, input_variables=["context"])
# The PromptTemplate wraps the system_template into a reusable object.
# The input_variables=["context"] defines which variables need to be filled in when the prompt is used.


# Initialize the generative model for question answering
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5)
# The ChatGoogleGenerativeAI class is used to instantiate a chat-based model.
#   Parameters:
#        model="gemini-1.5-pro-latest" specifies the version of the model to use.
#        temperature=0.5 controls the randomness of the responses. A value of 0.5 balances creativity and determinism.


# Create a document chain to handle the retrieval and response generation process
document_chain = create_stuff_documents_chain(llm=model, prompt=question_answering_prompt)
# The create_stuff_documents_chain() function integrates the model and the prompt into a chain.
```

### Testing
```python
document_chain.invoke(
    {
        "context": docs,
        "messages": [
            HumanMessage(content="What is the difference between high and medium protein-based diets?")
        ],
    }
)
```
### Retrieval Chain

```python
# Define a helper function to extract the latest user query from the input parameters
def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

# Create a retrieval chain with a passthrough mechanism
retrieval_chain = RunnablePassthrough.assign(
    # First step: Extract the user query and use it to retrieve relevant context
    context=parse_retriever_input | retriever,
).assign(
    # Second step: Use the retrieved context to generate an answer
    answer=document_chain,
)
```
### Testing Retrieval
```python
retrieval_chain.invoke(
    {
        "messages": [
            HumanMessage(content="What is the difference between high and medium protein-based diets?")
        ],
    }
)

# Follow Up Query
retrieval_chain.invoke(
    {
        "messages": [
            HumanMessage(content="Tell me more")
        ],
    }
)

# Testing with Vague Query
retriever.invoke("Tell me more!")

```
### Query Transformation

```python
query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
        ),
    ]
)
```

### Follow up Questions Transformation
```python
chat_history = [
    {"role": "user", "content": "What are common symptoms of long COVID?"},
    {"role": "assistant", "content": "Some symptoms include fatigue, shortness of breath, and brain fog."},
    {"role": "user", "content": "Can you provide examples?"}  # <- Hier wechselst du einfach die Follow-Up Frage
]

response = query_transform_prompt | llm
print(response.invoke({"messages": chat_history}).content)
```

### Add Model

```python
query_transformation_chain = query_transform_prompt | model
```
### Testing 
```python
query_transformation_chain.invoke(
    {
        "messages": [
            HumanMessage(content="What is the difference between high and medium protein-based diets?"),
            AIMessage(
                content="he study found that both high and normal protein diets improved body composition and glucose control in adults with type 2 diabetes. The lack of observed effects of dietary protein and red meat consumption on weight loss and improved cardiometabolic health suggest that achieved weight loss – rather than diet composition – should be the principal target of dietary interventions for T2D management."
            ),
            HumanMessage(content="Tell me more!"),
        ],
    }
)
```
### Build Query-Transforming Retriever Chain
```python
query_transforming_retriever_chain = RunnableBranch(
    (
        lambda x: len(x.get("messages", [])) == 1,
        # If only one message, then we just pass that message's content to retriever
        (lambda x: x["messages"][-1].content) | retriever,
    ),
    # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
    query_transform_prompt | model | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")
```
### Finalizing the Conversational Retrieval-Augmented Generation (RAG) Pipeline
```python
# Define the system template for generating answers
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""
# Create a prompt template for question answering (refer to Step 9 for prompt creation)
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),  # Adds conversational context (Step 9)
    ]
)

# Create a document chain for answering user questions (refer to Step 9)
document_chain = create_stuff_documents_chain(model, question_answering_prompt)

# Build the final conversational retrieval chain
# Combine the transformed query retrieval (Step 18) with the document chain (Step 9)
conversational_retrieval_chain = RunnablePassthrough.assign(
    # Assign the transformed query context to the retrieval chain (refer to Step 18)
    context=query_transforming_retriever_chain,
).assign(
    # Assign the answer generation process to the document chain (refer to Step 9)
    answer=document_chain,
)
```
### Testing the Conversational Retrieval Chain with an Unrelated Query
```python
conversational_retrieval_chain.invoke(
    {
        "messages": [
            HumanMessage(content="Can LangSmith help test my LLM applications?"),
        ]
    }
)
```
### Verifying the Conversational Retrieval Chain in the Target Use Case
```python
conversational_retrieval_chain.invoke(
    {
        "messages": [
            HumanMessage(content="What is the difference between high and medium protein-based diets?"),
            AIMessage(
                content="he study found that both high and normal protein diets improved body composition and glucose control in adults with type 2 diabetes. The lack of observed effects of dietary protein and red meat consumption on weight loss and improved cardiometabolic health suggest that achieved weight loss – rather than diet composition – should be the principal target of dietary interventions for T2D management."
            ),
            HumanMessage(content="Tell me more!"),
        ],
    }
    )
```
### Wrapping the Conversational Retrieval Pipeline in a Function

```python
@weave.op()
async def get_answer(question: str, messages: dict):
    """
    Handles user queries by appending them to the conversation history, 
    processing the query through the conversational retrieval chain, 
    and appending the AI's response back to the messages.

    Parameters:
    - question (str): The user's input question.
    - messages (dict): A dictionary containing the conversation history 
                       with a "messages" key holding a list of message objects.

    Returns:
    - str: The generated answer from the system.
    """
    # Add the user's question to the conversation history
    messages["messages"].append(HumanMessage(content=question))
    
    # Process the query through the conversational retrieval chain
    answer = conversational_retrieval_chain.invoke(messages)
    
    # Add the system's response to the conversation history
    messages["messages"].append(AIMessage(content=answer["answer"]))
    
    # Return the generated answer
    return answer["answer"]

    messages = {"messages": []} 
    answer = asyncio.get_event_loop().run_until_complete(get_answer("What is the difference between high and medium protein-based diets?", messages))
    print(answer) 
```
### Testing the Full Retrieval-Augmented Generation (RAG) Pipeline
```python
messages = {"messages": []} 
answer = asyncio.get_event_loop().run_until_complete(get_answer("how does high‑protein diet versus a low‑protein diet affect lean muscle mass retention and markers of renal function?", messages))
print(answer)
```
---
## rag and embeddings 
https://github.com/zhaw-iwi/rag-and-embeddings-solution/blob/main/RAG-and-embeddings-solution.ipynb

### Import

```python 
import tqdm
import glob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # For generating embeddings for text chunks
import faiss
import pickle
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np
from dotenv import load_dotenv
import os
from groq import Groq
```

### PDF Lesen
```python 
### load the pdf from the path
glob_path = "data/*.pdf"
text = ""
for pdf_path in tqdm.tqdm(glob.glob(glob_path)):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
         # Extract text from all pages in the PDF
        text += " ".join(page.extract_text() for page in reader.pages if page.extract_text())

text[:50]
```
### Chunks
```python
# Create a splitter: 2000 characters per chunk with an overlap of 200 characters
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# Split the extracted text into manageable chunks
chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")
print("Preview of the first chunk:", chunks[0][:200])
```
### Tokeniing the Text with Different Tokenizers
```python 
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=128, model_name="paraphrase-multilingual-MiniLM-L12-v2")

token_split_texts = []
for text in chunks:
    token_split_texts += token_splitter.split_text(text)

print(f"\nTotal chunks: {len(token_split_texts)}")
print(token_split_texts[0])
```
#### Multilingual
```python
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
tokenized_chunks = []
for i, text in enumerate(token_split_texts[:10]):
    # Tokenize each chunk
    encoded_input = model.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    # Convert token IDs back to tokens
    tokens = model.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0].tolist())
    tokenized_chunks.append(tokens)
    print(f"Chunk {i}: {tokens}")
```
#### German-semantic
```python
model_name = "Sahajtomar/German-semantic"
model = SentenceTransformer(model_name)
tokenized_chunks = []
for i, text in enumerate(token_split_texts[:10]):
    # Tokenize each chunk
    encoded_input = model.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    # Convert token IDs back to tokens
    tokens = model.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0].tolist())
    tokenized_chunks.append(tokens)
    print(f"Chunk {i}: {tokens}")
```
### Building a FAISS Vector Store
```python
d = chunk_embeddings.shape[1]
print(d)
index = faiss.IndexFlatL2(d)
index.add(chunk_embeddings)
print("Number of embeddings in FAISS index:", index.ntotal)
faiss.write_index(index, "faiss/faiss_index.index")
with open("faiss/chunks_mapping.pkl", "wb") as f:
    pickle.dump(token_split_texts, f)

index = faiss.read_index("faiss/faiss_index.index")
with open("faiss/chunks_mapping.pkl", "rb") as f:
    token_split_texts = pickle.load(f)
print(len(token_split_texts))

```
### Projecting Embeddings with UMAP
```python
# Fit UMAP on the full dataset embeddings
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(chunk_embeddings)

def project_embeddings(embeddings, umap_transform):
    """
    Project a set of embeddings using a pre-fitted UMAP transform.
    """
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm.tqdm(embeddings, desc="Projecting Embeddings")):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

# Project the entire dataset embeddings
projected_dataset_embeddings = project_embeddings(chunk_embeddings, umap_transform)
print("Projected dataset embeddings shape:", projected_dataset_embeddings.shape)
```
### Querying the Vector Store and Projecting Results
```python
def retrieve(query, k):
    """
    Retrieve the top k similar text chunks and their embeddings for a given query.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_texts = [token_split_texts[i] for i in indices[0]]
    retrieved_embeddings = np.array([chunk_embeddings[i] for i in indices[0]])
    return retrieved_texts, retrieved_embeddings, distances

query = "What is the most important factor in diagnosing asthma?"
results, result_embeddings, distances = retrieve(query, 3)
print("Retrieved document preview:")
print(results[0][:300])
print(results[1])

# Project the result embeddings
projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

# Also embed and project the original query for visualization
query_embedding = model.encode([query], convert_to_numpy=True)
project_original_query = project_embeddings(query_embedding, umap_transform)

# embed the embeddings of the entire dataset
projected_dataset_embeddings = project_embeddings(chunk_embeddings, umap_transform)
```
### Visualizing the Results
```python
def shorten_text(text, max_length=15):
    """Shortens text to max_length and adds an ellipsis if shortened."""
    return (text[:max_length] + '...') if len(text) > max_length else text

plt.figure()

# Scatter plots
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1],
            s=10, color='gray', label='Dataset')
plt.scatter(projected_result_embeddings[:, 0], projected_result_embeddings[:, 1],
            s=100, facecolors='none', edgecolors='g', label='Results')
plt.scatter(project_original_query[:, 0], project_original_query[:, 1],
            s=150, marker='X', color='r', label='Original Query')

# If results is a list of texts, iterate directly
for i, text in enumerate(results):
    if i < len(projected_result_embeddings):
        plt.annotate(shorten_text(text),
                     (projected_result_embeddings[i, 0], projected_result_embeddings[i, 1]),
                     fontsize=8)

# Annotate the original query point
original_query_text = 'Original Query Text'  # Replace with your actual query text if needed
plt.annotate(shorten_text(original_query_text),
             (project_original_query[0, 0], project_original_query[0, 1]),
             fontsize=8)

plt.gca().set_aspect('equal', 'datalim')
plt.title('Asthma')
plt.legend()
plt.show()
```
### Attach Retrieved Results to LLM
```python
load_dotenv()
# Access the API key using the variable name defined in the .env file
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

def retrieve_texts(query, k, index, token_split_texts, model):
    """
    Retrieve the top k similar text chunks and their embeddings for a given query.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_texts = [token_split_texts[i] for i in indices[0]]
    return retrieved_texts, distances

def answer_query(query, k, index,texts):
    """
    Retrieve the top k similar text chunks for the given query using the retriever,
    inject them into a prompt, and send it to the Groq LLM to obtain an answer.
    
    Parameters:
    - query (str): The user's query.
    - k (int): Number of retrieved documents to use.
    - groq_api_key (str): Your Groq API key.
    
    Returns:
    - answer (str): The answer generated by the LLM.
    """
    # Retrieve the top k documents using your retriever function.
    # This retriever uses the following definition:
    # def retrieve(query, k):
    #     query_embedding = model.encode([query], convert_to_numpy=True)
    #     distances, indices = index.search(query_embedding, k)
    #     retrieved_texts = [token_split_texts[i] for i in indices[0]]
    #     retrieved_embeddings = np.array([chunk_embeddings[i] for i in indices[0]])
    #     return retrieved_texts, retrieved_embeddings, distances
    model_name = "Sahajtomar/German-semantic"
    model = SentenceTransformer(model_name)
    retrieved_texts, _ = retrieve_texts(query, k, index, texts, model)
    
    # Combine the retrieved documents into a single context block.
    context = "\n\n".join(retrieved_texts)
    
    # Build a prompt that instructs the LLM to answer the query based on the context.
    prompt = (
        "Answer the following question using the provided context. "
        "Explain it as if you are explaining it to a 5 year old.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + query + "\n"
        "Answer:"
    )
    
    # Initialize the Groq client and send the prompt.
    client = Groq(api_key=groq_api_key)
    messages = [
        {
            "role": "system",
            "content": prompt
        }
    ]
    
    llm = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile"
    )
    
    # Extract and return the answer.
    answer = llm.choices[0].message.content
    return answer

query = "What is the most important factor in diagnosing asthma?"
answer = answer_query(query, 5, index, token_split_texts)
print("LLM Answer:", answer)
```
--- 
## rag advanced
https://github.com/zhaw-iwi/advanced-rag-solution

```bash
!pip install PyPDF2
!pip install langchain-community
!pip install faiss-cpu
!pip install groq
```
### Import
```python
import tqdm
import glob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # For generating embeddings for text chunks
import faiss
import pickle
from dotenv import load_dotenv
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import random
from sentence_transformers import CrossEncoder
import numpy as np
```
### API Key hinterlegen in Colab

![Key_in_Colab](img/Key_in_Colab.png)

### Imports dotenv nicht nötig
```python
from google.colab import userdata
google_api_key = userdata.get('GOOGLE_API_KEY')
openai_api_key = userdata.get('OPENAI_API_KEY')
groq_api_key = userdata.get('GROQ_API_KEY')
```
### Pfad anpassen
```bash
glob_path = "/content/*.pdf"
```
### RAG Pipeline
```python
# PDF
from PyPDF2 import PdfReader

glob_path = "data/*.pdf"
text = ""

for pdf_path in tqdm.tqdm(glob.glob(glob_path)):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text += " ".join(page.extract_text() for page in reader.pages if page.extract_text())

print(text[:500])

# CSV
import pandas as pd

glob_path = "data/*.csv"
text = ""

for csv_path in tqdm.tqdm(glob.glob(glob_path)):
    with open(csv_path, "r", encoding="utf-8") as file:
        df = pd.read_csv(file)
        text += "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1))

print(text[:500])

# TXT
glob_path = "data/*.txt"
text = ""

for txt_path in tqdm.tqdm(glob.glob(glob_path)):
    with open(txt_path, "r", encoding="utf-8") as file:
        text += file.read() + "\n"

print(text[:500])

# JSON
import json

def extract_text_from_json(obj):
    if isinstance(obj, dict):
        return " ".join([extract_text_from_json(v) for v in obj.values()])
    elif isinstance(obj, list):
        return " ".join([extract_text_from_json(i) for i in obj])
    else:
        return str(obj)

glob_path = "data/*.json"
text = ""

for json_path in tqdm.tqdm(glob.glob(glob_path)):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        text += extract_text_from_json(data) + "\n"

print(text[:500])

# Bilder
from PIL import Image
import pytesseract

glob_path = "data/*.[pjPJ]*"  # erfasst .jpg, .JPG, .png, .PNG usw.
text = ""

for image_path in tqdm.tqdm(glob.glob(glob_path)):
    with open(image_path, "rb") as file:
        img = Image.open(file)
        img.load()
        text += pytesseract.image_to_string(img) + "\n"

print(text[:500])
```
#### Chunks
```python
# PDF
# Angenommen: `text` enthält den zusammengeführten Text aus allen PDFs
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"Total chunks (PDF): {len(chunks)}")
print("Preview of the first PDF chunk:", chunks[0][:200])

# CSV
# text wurde aus mehreren CSV-Dateien generiert
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"Total chunks (CSV): {len(chunks)}")
print("Preview of the first CSV chunk:", chunks[0][:200])

# TXT
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"Total chunks (TXT): {len(chunks)}")
print("Preview of the first TXT chunk:", chunks[0][:200])

# JSON
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"Total chunks (JSON): {len(chunks)}")
print("Preview of the first JSON chunk:", chunks[0][:200])

# Bild
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"Total chunks (Images): {len(chunks)}")
print("Preview of the first image chunk:", chunks[0][:200])
```
#### Embedding model

```python
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
print(chunk_embeddings.dtype) # Testen welcher Datentyp mein Modell zurück gibt -> wenn Float32 alles OK, wenn nicht muss folgendes eingesetz werden:
chunk_embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")
```
#### Index
```python
# Vorbereitung einmalig
import faiss
import pickle
import numpy as np
import os

os.makedirs("faiss", exist_ok=True)

# Muster
# Schritt 1: FAISS-Index initialisieren und befüllen
d = chunk_embeddings.shape[1]
print("Dimensions:", d)

index = faiss.IndexFlatL2(d)
index.add(chunk_embeddings)
print("Number of embeddings in FAISS index:", index.ntotal)

# Schritt 2: Speichern
faiss.write_index(index, "faiss/faiss_index_TYP.index")  # <--- TYP ersetzen
with open("faiss/chunks_mapping_TYP.pkl", "wb") as f:
    pickle.dump(chunks, f)
```
Restliche Schritte im Github rag advanced