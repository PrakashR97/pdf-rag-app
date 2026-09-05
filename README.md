# 📚 Streamlit PDF RAG Application

This project is a simple **Retrieval-Augmented Generation (RAG)** application built using:

* Python
* Streamlit
* PyPDF
* Hugging Face Sentence Transformers
* FAISS
* OpenAI
* `.env` for API key management

The application allows the user to:

1. Upload a PDF.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Convert chunks into embeddings.
5. Store embeddings in FAISS.
6. Ask questions about the PDF.
7. Retrieve relevant chunks using FAISS.
8. Send the retrieved context to OpenAI.
9. Display the answer in Streamlit.
10. Save questions and answers to a text file.

---

# 1. Overall RAG Architecture

```text
                    Streamlit UI
                         |
                         v
                   Upload PDF
                         |
                         v
                    PDF Reader
                         |
                         v
                    Extract Text
                         |
                         v
                   Split into Chunks
                         |
                         v
                Hugging Face Embedding
                         |
                         v
                       FAISS
                  Vector Database
                         |
                         |
              User asks a question
                         |
                         v
                 Question Embedding
                         |
                         v
                  FAISS Similarity
                      Search
                         |
                         v
                 Top 4 Chunks
                         |
                         v
                      Context
                         |
                         v
                 OpenAI GPT-4o-mini
                         |
                         v
                       Answer
                         |
                         v
                   Streamlit UI
```

---

# 2. Install Required Libraries

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install streamlit faiss-cpu python-dotenv pypdf sentence-transformers openai
```

Or create a `requirements.txt` file:

```text
streamlit
faiss-cpu
python-dotenv
pypdf
sentence-transformers
openai
```

Then install:

```bash
pip install -r requirements.txt
```

---

# 3. Project Structure

Recommended project structure:

```text
rag_app/
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── questions_answers.txt
```

---

# 4. Import Required Libraries

```python
import os
from datetime import datetime

import streamlit as st
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
```

## Explanation

### `os`

Used to access environment variables.

```python
os.getenv()
```

We use it to read the OpenAI API key.

---

### `datetime`

Used to record the date and time when a question is asked.

```python
datetime.now()
```

---

### `streamlit`

```python
import streamlit as st
```

Streamlit is used to create the web application.

Examples:

```python
st.title()
st.write()
st.button()
st.text_input()
st.file_uploader()
st.spinner()
```

---

### `faiss`

FAISS is a vector similarity search library.

It allows us to search for the chunks that are most similar to the user's question.

---

### `dotenv`

```python
from dotenv import load_dotenv
```

Used to load variables from the `.env` file.

Example:

```text
OPENAI_API_KEY=your_api_key
```

---

### `PdfReader`

```python
from pypdf import PdfReader
```

Used to extract text from PDF files.

---

### `SentenceTransformer`

```python
from sentence_transformers import SentenceTransformer
```

Used to convert text into embeddings.

Example:

```text
"Spark is a distributed computing framework"
```

becomes something like:

```text
[0.21, -0.45, 0.78, ...]
```

These numbers are called an **embedding vector**.

---

### `OpenAI`

```python
from openai import OpenAI
```

Used to send the retrieved context and question to an OpenAI model.

---

# 5. Configuration

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4

CHUNK_SIZE = 800

CHUNK_OVERLAP = 100

QA_LOG_FILE = "questions_answers.txt"
```

## Embedding Model

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

This is the Hugging Face embedding model.

It converts text into vectors.

---

## TOP_K

```python
TOP_K = 4
```

This means FAISS retrieves the top 4 most similar chunks.

For example:

```text
Question
   |
   v
FAISS
   |
   +---- Chunk 12
   +---- Chunk 25
   +---- Chunk 7
   +---- Chunk 31
```

These 4 chunks are then sent to the LLM.

---

## Chunk Size

```python
CHUNK_SIZE = 800
```

Each chunk contains approximately 800 characters.

Example:

```text
PDF
 |
 +---- Chunk 1 → 800 characters
 +---- Chunk 2 → 800 characters
 +---- Chunk 3 → 800 characters
```

---

## Chunk Overlap

```python
CHUNK_OVERLAP = 100
```

The next chunk overlaps the previous chunk by 100 characters.

Example:

```text
Chunk 1:
AAAAAAAAAAAAAAAA [800 characters]

Chunk 2:
                  BBBBBBBBBBBBBBBB
                  ↑
                overlap
```

Overlap helps prevent important information from being split between two chunks.

---

# 6. Load Environment Variables

```python
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

The `.env` file contains:

```text
OPENAI_API_KEY=your_api_key
```

The following code checks whether the API key exists:

```python
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()
```

If the key doesn't exist, Streamlit displays an error and stops the application.

---

# 7. Create OpenAI Client

```python
client = OpenAI(api_key=OPENAI_API_KEY)
```

This creates an OpenAI client.

Later we use:

```python
client.chat.completions.create()
```

to generate the answer.

---

# 8. Load the Embedding Model

```python
@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(EMBEDDING_MODEL)

    return model
```

## What is `st.cache_resource`?

Streamlit reruns the Python script whenever the user interacts with the application.

Without caching, the embedding model could be loaded again and again.

```text
Without cache:

User clicks
   ↓
Streamlit reruns
   ↓
Load model
   ↓
User clicks
   ↓
Streamlit reruns
   ↓
Load model again
```

With:

```python
@st.cache_resource
```

Streamlit keeps the model in memory.

```text
First run
   ↓
Load model
   ↓
Cache model

Next interaction
   ↓
Reuse model
```

This makes the application much faster.

---

# 9. Load PDF

```python
def load_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            pages.append(
                f"\n--- Page {page_number + 1} ---\n{text}"
            )

    full_text = "\n".join(pages)

    return full_text
```

## Step 1

```python
reader = PdfReader(uploaded_file)
```

Reads the uploaded PDF.

---

## Step 2

```python
for page_number, page in enumerate(reader.pages):
```

Loops through every page.

For example:

```text
Page 1
Page 2
Page 3
Page 4
```

---

## Step 3

```python
text = page.extract_text()
```

Extracts text from each page.

---

## Step 4

```python
pages.append(...)
```

Stores the extracted text.

We also add the page number:

```text
--- Page 1 ---

Spark is a distributed computing framework...
```

This can later help us identify the source page.

---

# 10. Split Text into Chunks

```python
def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
```

The PDF may contain thousands of characters.

We cannot send the entire PDF to the LLM every time.

Therefore we split it into smaller pieces.

Example:

```text
PDF Text

       ↓

Chunk 1
800 characters

       ↓

Chunk 2
800 characters

       ↓

Chunk 3
800 characters

       ↓

Chunk 4
800 characters
```

---

# 11. Create FAISS Vector Database

```python
def create_vector_database(chunks, embedding_model):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index
```

This is one of the most important parts of the RAG application.

---

## Step 1: Convert chunks into embeddings

```python
embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
)
```

Example:

```text
Chunk 1
   ↓
Embedding
   ↓
[0.12, 0.45, -0.22, ...]
```

---

## Step 2: Convert to float32

```python
embeddings = embeddings.astype("float32")
```

FAISS works efficiently with `float32`.

---

## Step 3: Normalize vectors

```python
faiss.normalize_L2(embeddings)
```

Normalization allows inner product similarity to behave like cosine similarity.

---

## Step 4: Get vector dimension

```python
dimension = embeddings.shape[1]
```

The embedding model determines the vector dimension.

For:

```text
all-MiniLM-L6-v2
```

the embedding dimension is 384.

So conceptually:

```text
Chunk 1 → [384 numbers]

Chunk 2 → [384 numbers]

Chunk 3 → [384 numbers]
```

---

## Step 5: Create FAISS index

```python
index = faiss.IndexFlatIP(dimension)
```

`IP` means **Inner Product**.

Because we normalized the vectors, this can be used for cosine similarity.

---

## Step 6: Add embeddings

```python
index.add(embeddings)
```

Now FAISS contains all the document vectors.

```text
FAISS

Chunk 1 → Vector
Chunk 2 → Vector
Chunk 3 → Vector
Chunk 4 → Vector
...
```

---

# 12. Retrieve Relevant Documents

```python
def retrieve_documents(
    question,
    embedding_model,
    index,
    chunks
):
```

This function receives:

```text
Question
Embedding Model
FAISS Index
Document Chunks
```

---

## Convert Question into Embedding

```python
question_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True
)
```

Suppose the user asks:

```text
What is Spark?
```

The question becomes a vector:

```text
"What is Spark?"
       ↓
Embedding Model
       ↓
[0.12, -0.45, 0.76, ...]
```

---

## Normalize Question Vector

```python
faiss.normalize_L2(question_embedding)
```

The question vector is normalized just like the document vectors.

---

## Search FAISS

```python
scores, indexes = index.search(
    question_embedding,
    TOP_K
)
```

Since:

```python
TOP_K = 4
```

FAISS returns the 4 most similar chunks.

Example:

```text
Question
   ↓
Embedding
   ↓
FAISS
   ↓
Similarity Search
   ↓
Top 4 chunks
```

---

# 13. Build Context

```python
def build_context(retrieved_chunks):

    return "\n\n".join(retrieved_chunks)
```

The retrieved chunks are combined into one context.

Example:

```text
Chunk 1

Chunk 7

Chunk 12

Chunk 25
```

becomes:

```text
Chunk 1

Chunk 7

Chunk 12

Chunk 25
```

This context is sent to the LLM.

---

# 14. Generate Answer

```python
def generate_answer(question, context):
```

This function sends:

```text
Question
+
Retrieved Context
```

to OpenAI.

---

# 15. RAG Prompt

```python
prompt = f"""
You are a helpful assistant answering questions
about the uploaded document.

Answer the question using ONLY the provided context.

If the answer is not available in the context,
say:

"I could not find the answer in the provided document."

Do not invent information.

Context:
--------------------
{context}
--------------------

Question:
{question}

Provide a clear and concise answer.
"""
```

This is the instruction given to the LLM.

The important instruction is:

```text
Answer the question using ONLY the provided context.
```

This helps reduce hallucination.

---

# 16. Call OpenAI

```python
response = client.chat.completions.create(

    model="gpt-4o-mini",

    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful RAG assistant. "
                "Use only the provided document context."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ],

    temperature=0
)
```

The model receives:

```text
System instruction
        +
User prompt
        +
Retrieved context
```

---

## Temperature

```python
temperature=0
```

A temperature of 0 makes the response more deterministic.

For a RAG question-answering application, this is generally useful because we want factual answers based on the retrieved context.

---

# 17. Get the Answer

```python
return response.choices[0].message.content
```

The generated answer is returned to Streamlit.

Example:

```text
Question:
What is Spark?

Answer:
Apache Spark is a distributed computing framework
used for processing large datasets.
```

---

# 18. Save Question and Answer

```python
def save_question_answer(question, answer):
```

This function saves the conversation to:

```text
questions_answers.txt
```

---

## Get Current Date and Time

```python
current_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)
```

Example:

```text
2026-09-05 22:30:15
```

---

## Append to Text File

```python
with open(
    QA_LOG_FILE,
    "a",
    encoding="utf-8"
) as file:
```

`"a"` means **append**.

So new questions are added without deleting old questions.

Example:

```text
================================================

Date/Time: 2026-09-05 22:30:15

Question:
What is Spark?

Answer:
Spark is a distributed computing framework.

================================================
```

---

# 19. Configure Streamlit

```python
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚"
)
```

This configures the browser page.

---

# 20. Application Title

```python
st.title("📚 PDF RAG Assistant")
```

Displays:

```text
📚 PDF RAG Assistant
```

---

# 21. Application Description

```python
st.write(
    "Upload a PDF and ask questions about its content."
)
```

Displays a description underneath the title.

---

# 22. Load Embedding Model

```python
with st.spinner("Loading embedding model..."):

    embedding_model = load_embedding_model()
```

`st.spinner()` displays a loading message while the model is loading.

Example:

```text
⏳ Loading embedding model...
```

After loading, the message disappears.

---

# 23. PDF Upload

```python
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)
```

This creates a file upload button.

The user can select a PDF.

Example:

```text
Upload your PDF

[ Browse files ]
```

The uploaded PDF is stored in:

```python
uploaded_file
```

---

# 24. Check Whether PDF Was Uploaded

```python
if uploaded_file is not None:
```

If the user has uploaded a PDF, this condition becomes true.

---

# 25. Identify a New PDF

```python
file_id = (
    uploaded_file.name,
    uploaded_file.size
)
```

We use:

```text
filename
+
filesize
```

to identify the uploaded file.

For example:

```text
("Spark.pdf", 125000)
```

---

# 26. Streamlit Session State

```python
if st.session_state.get("file_id") != file_id:
```

Streamlit reruns the script frequently.

`st.session_state` allows us to store information between reruns.

We store:

```python
st.session_state.file_id
st.session_state.file_name
st.session_state.index
st.session_state.chunks
st.session_state.messages
```

This allows the application to remember the current PDF and FAISS database.

---

# 27. Process PDF

```python
with st.spinner("Processing PDF..."):
```

While the PDF is being processed, Streamlit shows:

```text
⏳ Processing PDF...
```

---

## Step 1: Extract Text

```python
text = load_pdf(uploaded_file)
```

Flow:

```text
PDF
 ↓
PdfReader
 ↓
Text
```

---

## Step 2: Check Text

```python
if not text.strip():
```

If no text can be extracted:

```python
st.error(
    "Could not extract text from this PDF."
)
```

This is useful for scanned PDFs where text extraction may not work.

---

# 28. Split PDF into Chunks

```python
chunks = split_text(text)
```

Example:

```text
PDF
 ↓
100,000 characters
 ↓
split_text()
 ↓
125 chunks
```

---

# 29. Create FAISS Index

```python
index = create_vector_database(
    chunks,
    embedding_model
)
```

Flow:

```text
Chunks
   ↓
Embedding Model
   ↓
Vectors
   ↓
Normalize
   ↓
FAISS
```

---

# 30. Store Data in Session State

```python
st.session_state.file_id = file_id

st.session_state.file_name = uploaded_file.name

st.session_state.index = index

st.session_state.chunks = chunks

st.session_state.messages = []
```

Now Streamlit remembers:

```text
Current PDF
Current FAISS index
Current chunks
```

---

# 31. Display Success Message

```python
st.success(
    f"PDF processed successfully: {uploaded_file.name}"
)
```

Example:

```text
✅ PDF processed successfully: Spark.pdf
```

---

# 32. Display Number of Chunks

```python
st.info(
    f"Created {len(chunks)} text chunks."
)
```

Example:

```text
Created 125 text chunks.
```

---

# 33. Ask a Question

```python
if "index" in st.session_state:
```

This checks whether a PDF has already been processed.

If an index exists, the question interface is displayed.

---

# 34. Add Divider

```python
st.divider()
```

Creates a horizontal line in the UI.

---

# 35. Question Input

```python
question = st.text_input(
    "Enter your question",
    placeholder="Example: What is Spark?"
)
```

Creates a text box.

Example:

```text
Enter your question

[ What is Spark? ]
```

---

# 36. Ask Button

```python
if st.button("Ask", type="primary"):
```

When the user clicks:

```text
[ Ask ]
```

the RAG pipeline runs.

---

# 37. Validate Question

```python
if not question.strip():

    st.warning(
        "Please enter a question."
    )
```

If the user clicks Ask without entering a question, Streamlit shows:

```text
⚠️ Please enter a question.
```

---

# 38. Retrieve Relevant Chunks

```python
retrieved_chunks = retrieve_documents(
    question,
    embedding_model,
    st.session_state.index,
    st.session_state.chunks
)
```

This is the **Retrieval** part of RAG.

Flow:

```text
User Question
      ↓
Embedding
      ↓
FAISS
      ↓
Similarity Search
      ↓
Top 4 Chunks
```

---

# 39. Generate Answer

If relevant chunks exist:

```python
context = build_context(
    retrieved_chunks
)

answer = generate_answer(
    question,
    context
)
```

Flow:

```text
Top 4 chunks
     ↓
Build Context
     ↓
OpenAI
     ↓
Answer
```

---

# 40. Display Answer

```python
st.subheader("🤖 Answer")

st.write(answer)
```

The answer appears in the browser.

Example:

```text
🤖 Answer

Apache Spark is a distributed computing
framework designed for processing large datasets.
```

---

# 41. Save Question and Answer

```python
save_question_answer(
    question,
    answer
)
```

The question and answer are saved to:

```text
questions_answers.txt
```

---

# 42. No PDF Message

If the user hasn't uploaded a PDF:

```python
else:

    st.info(
        "👆 Upload a PDF to start asking questions."
    )
```

The UI displays:

```text
ℹ️ Upload a PDF to start asking questions.
```

---

# 43. Complete RAG Flow

The entire application works like this:

```text
                 USER
                  |
                  v
            Upload PDF
                  |
                  v
             PdfReader
                  |
                  v
            Extract Text
                  |
                  v
          Split into Chunks
                  |
                  v
        Sentence Transformer
                  |
                  v
             Embeddings
                  |
                  v
               FAISS
                  |
                  |
                  |
             Ask Question
                  |
                  v
       Question Embedding
                  |
                  v
          FAISS Similarity
              Search
                  |
                  v
             Top 4 Chunks
                  |
                  v
               Context
                  |
                  v
          OpenAI GPT-4o-mini
                  |
                  v
               Answer
                  |
                  v
             Streamlit
                  |
                  v
               USER
```

---

# 44. RAG Components in This Project

| RAG Component    | Technology              |
| ---------------- | ----------------------- |
| Document         | PDF                     |
| Document Loader  | PyPDF                   |
| Text Splitter    | Custom Python function  |
| Embedding Model  | Hugging Face            |
| Vector Database  | FAISS                   |
| Retriever        | FAISS similarity search |
| LLM              | OpenAI GPT-4o-mini      |
| UI               | Streamlit               |
| Conversation Log | TXT file                |

---

# 45. Important Concepts to Understand

## Ingestion

Ingestion means preparing the document for RAG.

```text
PDF
 ↓
Extract Text
 ↓
Split Text
 ↓
Create Embeddings
 ↓
Store in Vector Database
```

---

## Embedding

Embedding converts text into numbers.

```text
"Spark is fast"
       ↓
Embedding Model
       ↓
[0.21, -0.43, 0.87, ...]
```

Similar meanings produce vectors that are close to each other.

---

## Vector Database

FAISS stores and searches these vectors.

```text
Document chunks
      ↓
   Embeddings
      ↓
     FAISS
```

---

## Retrieval

When the user asks a question:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Search
   ↓
Top 4 Similar Chunks
```

---

## Generation

The retrieved chunks are sent to the LLM.

```text
Question
   +
Retrieved Context
   ↓
OpenAI
   ↓
Answer
```

This is why it is called:

**Retrieval-Augmented Generation**

---

# 46. Why Do We Need FAISS?

Without RAG:

```text
User
 ↓
Question
 ↓
LLM
 ↓
Answer
```

The LLM may not know the information inside your private PDF.

With RAG:

```text
PDF
 ↓
FAISS
 ↓
Relevant information
 ↓
LLM
 ↓
Answer
```

The LLM receives information retrieved from your document.

---

# 47. Why Don't We Send the Entire PDF to OpenAI?

Suppose the PDF contains:

```text
500 pages
```

Sending everything for every question would be inefficient.

Instead:

```text
500-page PDF
      ↓
Chunks
      ↓
FAISS
      ↓
Top 4 relevant chunks
      ↓
OpenAI
```

Only the relevant information is sent to the LLM.

---

# 48. What Happens When a New PDF Is Uploaded?

Suppose the user first uploads:

```text
Spark.pdf
```

The application creates:

```text
Spark.pdf
   ↓
Chunks
   ↓
Embeddings
   ↓
FAISS
```

Then the user uploads:

```text
Python.pdf
```

The application detects a new file:

```python
if st.session_state.get("file_id") != file_id:
```

and creates a new FAISS index.

So the current session changes from:

```text
Spark.pdf
   ↓
Spark FAISS
```

to:

```text
Python.pdf
   ↓
Python FAISS
```

---

# 49. Run the Application

Start the virtual environment:

```bash
source venv/bin/activate
```

Run Streamlit:

```bash
streamlit run app.py
```

Streamlit will start a local web application.

Usually you will see something similar to:

```text
Local URL: http://localhost:8501
```

Open that address in your browser.

---

# 50. Final Application Flow

The user experience is very simple:

```text
📚 PDF RAG Assistant

Upload your PDF

[ Browse files ]

        ↓

Processing PDF...

        ↓

✅ PDF processed successfully

Created 125 text chunks.

        ↓

💬 Ask a Question

[ What is Spark? ]

[ Ask ]

        ↓

🤖 Answer

Spark is a distributed computing framework...
```

---

# 51. Simple Explanation for an Interview

If someone asks:

**"Explain your RAG application."**

You can say:

> I built a PDF-based RAG application using Streamlit. The user uploads a PDF, and I extract the text using PyPDF. I split the text into smaller chunks and generate embeddings using the Hugging Face `all-MiniLM-L6-v2` model. These embeddings are stored in a FAISS vector database. When the user asks a question, I convert the question into an embedding and perform similarity search in FAISS to retrieve the top four relevant chunks. I then pass those chunks as context along with the user's question to OpenAI GPT-4o-mini, which generates the final answer. The application is deployed through a Streamlit UI, and I also log the questions and answers to a text file.

---

# 52. Technologies Used

```text
Python
   |
   +-- Streamlit
   |
   +-- PyPDF
   |
   +-- Sentence Transformers
   |
   +-- FAISS
   |
   +-- OpenAI
   |
   +-- python-dotenv
```

This gives you a complete beginner-level **PDF → Embedding → FAISS → Retrieval → LLM → Answer** RAG application.
