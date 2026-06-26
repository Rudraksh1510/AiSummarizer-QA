import streamlit as st
from src.pdfsummarizer.Summarizer import build_rag_chain
from src.pdfsummarizer.Summarizer import ask_question
uploaded_file = st.file_uploader(
    "Upload your file",
    type=["pdf", "txt", "md", "csv"]
)

if uploaded_file:

    index = build_rag_chain(uploaded_file)

    question = st.chat_input("Ask something...")

    if question:

        answer = ask_question(index, question)

        st.write(answer)