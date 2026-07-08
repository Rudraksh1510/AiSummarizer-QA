import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

import streamlit as st

from src.pdfsummarizer.logger import logging
from src.pdfsummarizer.utils import extract_text_from_file, extract_text_from_files

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

logging.info("API Keys Loaded Successfully.")

# LLM


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
)

# Embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

You have to understand the question very  carefully and then answer the question using the information from the document.

If the answer is not present in the context, reply:

"I couldn't find that information in the uploaded document."

Context:
{context}
""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

output_parser = StrOutputParser()

# Trim history to last 5 messages (i.e. keep window small, drop older turns)
trimmer = trim_messages(
    max_tokens=5,
    strategy="last",
    token_counter=len,   # count messages, not tokens -> "5" = last 5 messages
    include_system=False,
    start_on="human",
)

# Base chain: trims history -> formats prompt -> LLM -> string output
chain = (
    RunnablePassthrough_dict_wrapper := (
        lambda inputs: {
            **inputs,
            "history": trimmer.invoke(inputs["history"]),
        }
    )
)

base_chain = (
    RunnableLambda(chain)
    | prompt
    | llm
    | output_parser
)

# Per-session in-memory chat history store, isolated by Streamlit session
_session_store: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def get_session_id() -> str:
    """One stable session id per Streamlit session (browser tab)."""
    if "rag_session_id" not in st.session_state:
        st.session_state.rag_session_id = str(uuid.uuid4())
    return st.session_state.rag_session_id


def clear_session_history() -> None:
    """Reset the current Streamlit session's chat history."""
    _session_store[get_session_id()] = InMemoryChatMessageHistory()


rag_chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    _get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# Build Vector Store

def build_rag_chain(uploaded_files):

    logging.info("Processing uploaded file(s)...")

    # Handle both single file and multiple files for backward compatibility
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    # Seek to start for all files
    for file in uploaded_files:
        try:
            file.seek(0)
        except Exception:
            pass

    # Use extract_text_from_files for multiple files
    chunks = extract_text_from_files(uploaded_files)

    logging.info(f"Generated {len(chunks)} chunks.")

    index_name = "test"

    index = pc.Index(index_name)

    # Remove any previous vectors from the same index before uploading new content.
    try:
        index.delete(delete_all=True)
        logging.info("Cleared previous Pinecone index vectors.")
    except Exception as delete_err:
        logging.warning(f"Could not clear Pinecone index before upload: {delete_err}")

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

def get_chat_history():
    """Returns the list of messages currently held in memory for this session."""
    return _get_session_history(get_session_id()).messages

# Ask Question

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

    # LLM response, with last-5-message memory handled by RunnableWithMessageHistory
    answer = rag_chain_with_memory.invoke(
        {
            "context": context,
            "question": question,
        },
        config={"configurable": {"session_id": get_session_id()}},
    )

    logging.info("Answer Generated Successfully.")

    return answer