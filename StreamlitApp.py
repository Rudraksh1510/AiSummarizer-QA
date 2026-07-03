import streamlit as st
from src.pdfsummarizer.Summarizer import build_rag_chain
from src.pdfsummarizer.Summarizer import ask_question
from src.pdfsummarizer.Summarizer import get_chat_history

uploaded_file = st.file_uploader(
    "Upload your file",
    type=["pdf", "txt", "md", "csv", "pptx"]
)

if uploaded_file:

    # Build the index only once
    if "index" not in st.session_state:
        try:
            with st.spinner("Processing your document..."):
                st.session_state.index = build_rag_chain(uploaded_file)

            st.success("✅ Document processed successfully!")

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                st.warning(
                    "⚠️ Gemini API rate limit reached.\n\n"
                    "Please wait a few seconds and upload the document again."
                )
                st.stop()
            else:
                st.error(f"Error while processing document:\n\n{e}")
                st.stop()

    # Render the full chat history (last 5 turns held in memory) on every rerun
    for message in get_chat_history():
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(message.content)

    question = st.chat_input("Ask something...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        try:
            with st.spinner("Searching..."):
                answer = ask_question(st.session_state.index, question)

            with st.chat_message("assistant"):
                st.write(answer)

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                st.warning(
                    "⏳ Gemini API limit reached.\n\n"
                    "Please wait 10 seconds and ask your question again."
                )
            else:
                st.error(f"An unexpected error occurred:\n\n{e}")