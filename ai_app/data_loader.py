# /home/kamal/doc_finder/ai_app/data_loader.py
import os
from typing import List, Optional
import logging # Import logging
from langchain_community.document_loaders import PyPDFLoader
# Removed UnstructuredFileLoader and OCR dependencies for this simplified approach
# If you need OCR for image-based PDFs, that logic would need refinement here.
# from langchain_community.document_loaders import UnstructuredFileLoader
# import pytesseract
# import cv2
# from pdf2image import convert_from_path
# import numpy as np
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # Keep splitter for potential summarization later if needed
from config import config

logger = logging.getLogger(__name__) # Get logger

class DataLoader:
    def __init__(self, pdf_directory: str = config.PDF_DIRECTORY):
        self.pdf_directory = pdf_directory
        # Optional: Initialize a text splitter if you want to chunk *within* the combined content later
        # self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def load_pdfs(self) -> List[Document]:
        """
        Loads all PDFs from the specified directory.
        Creates ONE Document per PDF file, containing concatenated text.
        """
        all_docs_per_file: List[Document] = []
        logger.info(f"Scanning directory '{self.pdf_directory}' for PDF files...")
        pdf_files_found = 0

        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith(".pdf"):
                pdf_files_found += 1
                filepath = os.path.join(self.pdf_directory, filename)
                logger.debug(f"Attempting to load PDF: {filename}")
                try:
                    # Use PyPDFLoader to get pages
                    loader = PyPDFLoader(filepath)
                    pages = loader.load() # Loads pages as separate Document objects initially

                    if not pages:
                        logger.warning(f"PyPDFLoader returned no pages for {filename}. Skipping.")
                        continue

                    # Concatenate page content
                    full_text = "\n".join([page.page_content for page in pages if page.page_content])

                    if not full_text.strip():
                        logger.warning(f"No text content extracted from {filename} after concatenation. Skipping.")
                        continue

                    # Create a single Document for the entire PDF
                    # Metadata now clearly links to the source file
                    file_document = Document(
                        page_content=full_text,
                        metadata={"source": filename} # Essential metadata: the filename
                    )
                    all_docs_per_file.append(file_document)
                    logger.info(f"Successfully processed '{filename}' into a single document.")

                # Simplified error handling for this approach. Add OCR/Unstructured back if needed.
                except Exception as e:
                    logger.error(f"Failed to load or process {filename} with PyPDFLoader: {e}", exc_info=True)
                    # Optionally, try other loaders here if PyPDFLoader fails consistently

        logger.info(f"Found {pdf_files_found} PDF files. Successfully processed {len(all_docs_per_file)} files into documents.")
        if pdf_files_found > 0 and not all_docs_per_file:
             logger.error("Found PDF files but failed to process any into documents. Check PDF content and loader errors.")

        return all_docs_per_file
