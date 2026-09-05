
import os
from datetime import datetime

import streamlit as st
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

QA_LOG_FILE = "questions_answers.txt"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(EMBEDDING_MODEL)

    return model


# ============================================================
# LOAD PDF
# ============================================================

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


# ============================================================
# SPLIT TEXT
# ============================================================

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


# ============================================================
# CREATE FAISS DATABASE
# ============================================================

def create_vector_database(chunks, embedding_model):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    # Normalize vectors
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # FAISS cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    embedding_model,
    index,
    chunks
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    # Normalize
    faiss.normalize_L2(question_embedding)

    # Search FAISS
    scores, indexes = index.search(
        question_embedding,
        TOP_K
    )

    retrieved_chunks = []

    for index_number in indexes[0]:

        if index_number == -1:
            continue

        retrieved_chunks.append(
            chunks[index_number]
        )

    return retrieved_chunks


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(retrieved_chunks):

    return "\n\n".join(retrieved_chunks)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

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

    return response.choices[0].message.content


# ============================================================
# SAVE QUESTION AND ANSWER
# ============================================================

def save_question_answer(question, answer):

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        QA_LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write("\n")
        file.write("=" * 80)
        file.write("\n")

        file.write(
            f"Date/Time: {current_time}\n\n"
        )

        file.write(
            f"Question:\n{question}\n\n"
        )

        file.write(
            f"Answer:\n{answer}\n"
        )

        file.write("\n")


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚"
)

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

with st.spinner("Loading embedding model..."):

    embedding_model = load_embedding_model()


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS NEW PDF
# ============================================================

if uploaded_file is not None:

    # Check whether this is a new PDF
    file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    if st.session_state.get("file_id") != file_id:

        with st.spinner("Processing PDF..."):

            # -----------------------------------------------
            # 1. Load PDF
            # -----------------------------------------------

            text = load_pdf(uploaded_file)

            if not text.strip():

                st.error(
                    "Could not extract text from this PDF."
                )

                st.stop()

            # -----------------------------------------------
            # 2. Split text
            # -----------------------------------------------

            chunks = split_text(text)

            # -----------------------------------------------
            # 3. Create FAISS
            # -----------------------------------------------

            index = create_vector_database(
                chunks,
                embedding_model
            )

            # -----------------------------------------------
            # 4. Store in session
            # -----------------------------------------------

            st.session_state.file_id = file_id

            st.session_state.file_name = uploaded_file.name

            st.session_state.index = index

            st.session_state.chunks = chunks

            st.session_state.messages = []

        st.success(
            f"PDF processed successfully: {uploaded_file.name}"
        )

        st.info(
            f"Created {len(chunks)} text chunks."
        )


# ============================================================
# ASK QUESTION
# ============================================================

if "index" in st.session_state:

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What is Spark?"
    )

    if st.button("Ask", type="primary"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner("Searching document..."):

                # -------------------------------------------
                # RETRIEVAL
                # -------------------------------------------

                retrieved_chunks = retrieve_documents(
                    question,
                    embedding_model,
                    st.session_state.index,
                    st.session_state.chunks
                )

                # -------------------------------------------
                # GENERATE ANSWER
                # -------------------------------------------

                if not retrieved_chunks:

                    answer = (
                        "I could not find relevant information "
                        "in the document."
                    )

                else:

                    context = build_context(
                        retrieved_chunks
                    )

                    answer = generate_answer(
                        question,
                        context
                    )

            # -----------------------------------------------
            # DISPLAY ANSWER
            # -----------------------------------------------

            st.subheader("🤖 Answer")

            st.write(answer)

            # -----------------------------------------------
            # SAVE Q&A
            # -----------------------------------------------

            save_question_answer(
                question,
                answer
            )


# ============================================================
# NO PDF MESSAGE
# ============================================================

else:

    st.info(
        "👆 Upload a PDF to start asking questions."
    )
