# /home/kamal/doc_finder/ai_app/indexer.py
import os
import logging
import shutil # Import shutil for removing files/directories
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from embedder import Embedder
from config import config
# Import DataLoader type hint for the rebuild method
from data_loader import DataLoader

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, embedder: Embedder, index_directory: str = config.INDEX_DIRECTORY):
        self.embedder = embedder
        self.index_directory = index_directory
        self.index_name = "faiss_index"
        self.vectorstore: FAISS | None = None

    def _get_index_path(self) -> str:
        """Helper to get the base path for FAISS files."""
        return os.path.join(self.index_directory, self.index_name)

    def index_exists(self) -> bool:
        """Checks if the FAISS index files exist."""
        faiss_file = f"{self._get_index_path()}.faiss"
        pkl_file = f"{self._get_index_path()}.pkl"
        # Check if the directory exists first
        return os.path.isdir(self.index_directory) and \
               os.path.exists(faiss_file) and \
               os.path.exists(pkl_file)

    def _remove_existing_index(self):
        """Removes the existing FAISS index files."""
        faiss_file = f"{self._get_index_path()}.faiss"
        pkl_file = f"{self._get_index_path()}.pkl"
        removed = False
        if os.path.exists(faiss_file):
            try:
                os.remove(faiss_file)
                logger.debug(f"Removed existing index file: {faiss_file}")
                removed = True
            except OSError as e:
                logger.error(f"Failed to remove index file {faiss_file}: {e}", exc_info=True)
                raise # Re-raise to indicate failure
        if os.path.exists(pkl_file):
            try:
                os.remove(pkl_file)
                logger.debug(f"Removed existing index file: {pkl_file}")
                removed = True
            except OSError as e:
                logger.error(f"Failed to remove index file {pkl_file}: {e}", exc_info=True)
                raise # Re-raise to indicate failure
        if removed:
             logger.info(f"Successfully removed existing index files from '{self.index_directory}'.")
        else:
             logger.info(f"No existing index files found at '{self._get_index_path()}' to remove.")


    def create_n_save_index(self, documents: List[Document]):
        """Creates a FAISS vector index from a list of documents and saves it."""
        if not documents:
            logger.error("Cannot create index: No documents provided.")
            raise ValueError("No documents provided to create index.")
        if not self.embedder or not self.embedder.model:
             logger.error("Cannot create index: Embedder or embedding model not initialized.")
             raise ValueError("Embedder not properly initialized.")

        logger.info(f"Creating FAISS index from {len(documents)} documents...")
        try:
            # Ensure the index directory exists before saving
            if not os.path.exists(self.index_directory):
                logger.info(f"Creating index directory: {self.index_directory}")
                os.makedirs(self.index_directory)

            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embedder.model
            )
            logger.info("FAISS index created in memory.")
            self.save_index() # Save immediately after creation
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}", exc_info=True)
            raise

    def save_index(self):
        """Saves the FAISS index to disk."""
        if self.vectorstore is None:
            logger.error("Cannot save index: Vectorstore not created.")
            raise ValueError("Vectorstore not created, cannot save.")

        index_path = self._get_index_path()
        logger.info(f"Saving FAISS index to: {index_path}.faiss / .pkl")
        try:
            # Ensure the directory exists just before saving
            if not os.path.exists(self.index_directory):
                 logger.warning(f"Index directory '{self.index_directory}' did not exist. Creating it now.")
                 os.makedirs(self.index_directory)
            self.vectorstore.save_local(folder_path=self.index_directory, index_name=self.index_name)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index to {index_path}: {e}", exc_info=True)
            raise

    def load_index(self):
        """Loads an existing FAISS vector index."""
        if not self.index_exists():
            logger.error(f"FAISS index files not found in '{self.index_directory}' with base name '{self.index_name}'.")
            raise FileNotFoundError(f"FAISS index not found at {self._get_index_path()}")
        if not self.embedder or not self.embedder.model:
             logger.error("Cannot load index: Embedder or embedding model not initialized.")
             raise ValueError("Embedder not properly initialized for loading FAISS index.")

        index_path = self._get_index_path()
        logger.info(f"Loading FAISS index from: {index_path}.faiss / .pkl")
        try:
            self.vectorstore = FAISS.load_local(
                folder_path=self.index_directory,
                embeddings=self.embedder.model,
                index_name=self.index_name,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_path}: {e}", exc_info=True)
            raise

    def search(self, query: str, k: int = 2) -> List[Document]:
        """Searches the FAISS index for similar documents."""
        if self.vectorstore is None:
            logger.error("Cannot search: FAISS index not loaded or created.")
            raise ValueError("Index not loaded or created.")
        logger.debug(f"Performing FAISS similarity search for query: '{query}' with k={k}")
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.debug(f"FAISS search returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during FAISS similarity search: {e}", exc_info=True)
            return []

    def setup_index(self, data_loader: DataLoader) -> bool:
        """Sets up the FAISS index by loading or creating it if it doesn't exist."""
        try:
            if self.index_exists():
                logger.info(f"Existing FAISS index found in '{self.index_directory}'. Loading index...")
                self.load_index()
                logger.info("FAISS index loaded successfully.")
                return True
            else:
                logger.info(f"FAISS index not found in '{self.index_directory}'.")
                logger.info(f"Attempting to create a new index from PDFs in '{config.PDF_DIRECTORY}'...")
                # Call the rebuild logic, which handles loading and creation
                return self.rebuild_index(data_loader)

        except Exception as e:
            logger.error(f"Error during index setup: {e}", exc_info=True)
            # Don't re-raise here, allow main to handle the False return
            return False # Indicate failure

    def rebuild_index(self, data_loader: DataLoader) -> bool:
        """
        Forces a rebuild of the index: removes existing index files,
        loads all current PDFs, and creates/saves a new index.
        Returns True on success, False on failure.
        """
        logger.info(f"Starting index rebuild process for directory '{self.index_directory}'...")
        try:
            # 1. Remove existing index files (if they exist)
            self._remove_existing_index()

            # 2. Check if PDF directory exists and has PDFs
            if not os.path.isdir(config.PDF_DIRECTORY) or not any(f.lower().endswith(".pdf") for f in os.listdir(config.PDF_DIRECTORY)):
                logger.error(f"PDF directory '{config.PDF_DIRECTORY}' is empty or does not exist. Cannot rebuild index.")
                return False # Cannot proceed without PDFs

            # 3. Load documents from all PDFs in the directory
            logger.info(f"Loading documents from PDFs in '{config.PDF_DIRECTORY}'...")
            documents_per_file = data_loader.load_pdfs()

            # 4. Create and save the new index
            if documents_per_file:
                logger.info(f"Successfully processed {len(documents_per_file)} PDFs. Creating and saving new FAISS index...")
                self.create_n_save_index(documents_per_file) # This handles creation and saving
                logger.info(f"FAISS index rebuilt and saved successfully in '{self.index_directory}'.")
                return True
            else:
                logger.error("No documents were processed from PDFs. Index cannot be rebuilt.")
                return False # Failed to load any documents

        except FileNotFoundError as e:
             logger.error(f"Error during index rebuild (file not found): {e}", exc_info=True)
             return False
        except ValueError as e:
             logger.error(f"Error during index rebuild (value error): {e}", exc_info=True)
             return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during index rebuild: {e}", exc_info=True)
            return False # Indicate failure
