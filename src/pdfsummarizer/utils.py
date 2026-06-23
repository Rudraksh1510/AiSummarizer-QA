from pathlib import Path
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Extracting text from Pdf using PyPDF2 and as well as making chunks of it. 

def extract_text_from_file(file):
    try:
        file_extension = Path(file.name).suffix.lower()

        text = ""

        # PDF Files
        if file_extension == ".pdf":
            reader = PdfReader(file)

            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"

        # Plain Text Files
        elif file_extension in [".txt", ".md", ".csv"]:
            text = file.read().decode("utf-8")

        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                "Supported formats are PDF, TXT, MD, and CSV."
            )

        if not text.strip():
            raise ValueError("The file contains no readable text.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

        chunks = text_splitter.split_text(text)
        logging.info("Text Extraction was successful!")
        return chunks

    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")