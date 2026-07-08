import hashlib
import streamlit as st
from src.pdfsummarizer.Summarizer import build_rag_chain
from src.pdfsummarizer.Summarizer import ask_question
from src.pdfsummarizer.Summarizer import get_chat_history
from src.pdfsummarizer.Summarizer import clear_session_history

uploaded_files = st.file_uploader(
    "Upload your file(s)",
    type=["pdf", "txt", "md", "csv", "pptx"],
    accept_multiple_files=True
)

current_file_key = None
if uploaded_files:
    # Create a hash of all uploaded files for change detection
    file_info_list = []
    for file in uploaded_files:
        file_bytes = file.read()
        file.seek(0)
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        file_info_list.append((file.name, file.type, len(file_bytes), content_hash))
    current_file_key = tuple(file_info_list)

previous_file_key = st.session_state.get("uploaded_file_key")

# if the user clears the file uploader, reset the index and chat history
if not uploaded_files and previous_file_key is not None:
    st.session_state.pop("uploaded_file_key", None)
    st.session_state.pop("index", None)
    clear_session_history()

# if different files are uploaded, discard the previous file's index and history
if uploaded_files and current_file_key != previous_file_key:
    st.session_state["uploaded_file_key"] = current_file_key
    st.session_state.pop("index", None)
    clear_session_history()

if uploaded_files:

    # Build the index only once
    if "index" not in st.session_state:
        try:
            with st.spinner("Processing your document(s)..."):
                st.session_state.index = build_rag_chain(uploaded_files)

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