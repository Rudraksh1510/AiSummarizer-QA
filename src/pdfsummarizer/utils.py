from pathlib import Path
from io import BytesIO
import pymupdf
import fitz
import pytesseract
from PIL import Image
from pptx import Presentation
from src.pdfsummarizer.logger import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Extracting text from PDF/PPTX using OCR (PyMuPDF + pytesseract, no poppler dependency)
# and as well as making chunks of it.

def extract_text_from_file(file):
    try:
        file_extension = Path(file.name).suffix.lower()

        text = ""

        # PDF Files (OCR-based extraction via PyMuPDF, no poppler needed)
        if file_extension == ".pdf":
            file_bytes = file.read()
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")

            for page_number, page in enumerate(pdf_doc, start=1):
                pix = page.get_pixmap(dpi=300)
                image = Image.open(BytesIO(pix.tobytes("png")))
                page_text = pytesseract.image_to_string(image)
                text += (page_text or "") + "\n"
                logging.info(f"OCR completed for PDF page {page_number}.")

            pdf_doc.close()

        # PowerPoint Files (text extraction + OCR on embedded images)
        elif file_extension in [".ppt", ".pptx"]:
            file_bytes = file.read()
            presentation = Presentation(BytesIO(file_bytes))

            for slide_number, slide in enumerate(presentation.slides, start=1):
                for shape in slide.shapes:
                    # Native text on the slide (text boxes, titles, etc.)
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                text += run.text + " "
                        text += "\n"

                    # Tables
                    if shape.has_table:
                        for row in shape.table.rows:
                            for cell in row.cells:
                                text += cell.text + " "
                            text += "\n"

                    # OCR on embedded images (screenshots, diagrams, etc.)
                    if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                        try:
                            image_bytes = shape.image.blob
                            image = Image.open(BytesIO(image_bytes))
                            ocr_text = pytesseract.image_to_string(image)
                            text += (ocr_text or "") + "\n"
                        except Exception as img_err:
                            logging.warning(f"Could not OCR image on slide {slide_number}: {img_err}")

                logging.info(f"Processed slide {slide_number}.")

        # Plain Text Files
        elif file_extension in [".txt", ".md", ".csv"]:
            text = file.read().decode("utf-8")

        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                "Supported formats are PDF, PPT, PPTX, TXT, MD, and CSV."
            )

        if not text.strip():
            raise ValueError("The file contains no readable text.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

        chunks = text_splitter.split_text(text)
        logging.info(f"Extracted {len(chunks)} chunks successfully.")
        return chunks

    except Exception as e:
        logging.exception(e)
        raise