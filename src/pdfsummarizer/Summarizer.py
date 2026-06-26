# Imports
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

from src.pdfsummarizer.logger import logging
from src.pdfsummarizer.utils import extract_text_from_file

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

logging.info("API Keys Loaded Successfully.")


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply:

"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
)

output_parser = StrOutputParser()


# ---------------------------------------------------------
# Build Vector Store
# ---------------------------------------------------------

def build_rag_chain(uploaded_file):

    logging.info("Processing uploaded file...")

    chunks = extract_text_from_file(uploaded_file)

    logging.info(f"Generated {len(chunks)} chunks.")

    index_name = "test"

    index = pc.Index(index_name)

    logging.info("Generating embeddings...")

    embeddings_list = embeddings.embed_documents(chunks)

    vectors = []

    for chunk, embedding in zip(chunks, embeddings_list):

        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "text": chunk
                }
            }
        )

    logging.info("Uploading vectors to Pinecone...")

    index.upsert(vectors=vectors)

    logging.info("Vectors Uploaded Successfully.")

    return index


# ---------------------------------------------------------
# Ask Question
# ---------------------------------------------------------

def ask_question(index, question):

    logging.info(f"Question Asked: {question}")

    # Create embedding for user's question
    query_embedding = embeddings.embed_query(question)

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Build context
    context = "\n\n".join(
        match["metadata"]["text"]
        for match in results["matches"]
    )

    # Create prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # LLM response
    response = llm.invoke(messages)

    answer = output_parser.invoke(response)

    logging.info("Answer Generated Successfully.")

    return answer