# /home/kamal/doc_finder/ai_app/main.py
import os
import logging
import argparse # Keep argparse for potential future flags
import sys # Import sys for exiting cleanly
from config import config
from data_loader import DataLoader
from embedder import Embedder
from indexer import Indexer # Make sure Indexer has the rebuild_index method
from llm_handler import LLMHandler
from query_engine import QueryEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory(directory: str):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)
            raise

def initialize_components():
    """Initializes and returns the system components."""
    logger.info("Initializing components...")
    try:
        # Ensure necessary directories exist before initializing components that might use them
        create_directory(config.PDF_DIRECTORY)
        create_directory(config.INDEX_DIRECTORY)

        data_loader = DataLoader(pdf_directory=config.PDF_DIRECTORY)
        embedder = Embedder()
        # Pass allow_dangerous_deserialization=True if needed by your FAISS load_local
        indexer = Indexer(embedder=embedder, index_directory=config.INDEX_DIRECTORY)
        llm_handler = LLMHandler()
        query_engine = QueryEngine(indexer=indexer, llm_handler=llm_handler)
        logger.info("Components initialized successfully.")
        return data_loader, indexer, query_engine
    except Exception as e:
        logger.error(f"Error during component initialization: {e}", exc_info=True)
        # Re-raise to be caught by the main try-except block
        raise

def query_loop(query_engine):
    """Starts the main query loop for user interaction."""
    logger.info("Starting query loop. Type 'exit' to quit.")
    while True:
        try:
            user_query = input("\nEnter your query (or 'exit' to quit): ")
            if user_query.strip().lower() == "exit":
                logger.info("Exit command received.")
                break
            if not user_query.strip():
                continue

            logger.info(f"User query received: '{user_query}'")
            logger.info("\nSearching for relevant PDFs and generating suggestions...")
            # Use the query method from QueryEngine which handles the LLM interaction
            response = query_engine.query(user_query)
            # logger.info("\n--- Suggested PDFs ---") # Original header
            logger.info("\n--- Response ---") # Changed header for clarity
            logger.info(response)
            logger.info("----------------\n") # Match closing dashes
        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt received. Exiting.")
            break
        except Exception as e:
            logger.error(f"Unexpected error during query loop: {e}", exc_info=True)
            logger.info("An unexpected error occurred. Please check logs.")

def main():
    """Main function to orchestrate the RAG system."""
    # --- Argument Parsing (Optional: Keep if you might add other flags later) ---
    parser = argparse.ArgumentParser(description="Doc Finder: Query PDFs using RAG.")
    # Example: Add back --rebuild-index if you want both options
    # parser.add_argument(
    #     '--rebuild-index',
    #     action='store_true',
    #     help='Force rebuild the FAISS index non-interactively before starting.'
    # )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    try:
        data_loader, indexer, query_engine = initialize_components()

        index_ready = False
        rebuild_requested = False

        # Check if index exists
        if indexer.index_exists():
            logger.info("Existing FAISS index found.")
            # Prompt user, default to 'no' if they just press Enter
            choice = input("Rebuild index with current PDFs? (yes/[no]): ").strip().lower()

            if choice in ['yes', 'y']:
                rebuild_requested = True
            # Treat empty input or anything other than 'yes'/'y' as 'no'
            else:
                rebuild_requested = False
                if choice not in ['no', 'n', '']: # Log if they typed something other than no/n/empty
                     logger.info(f"Input '{choice}' treated as 'no'. Proceeding without rebuild.")

            if rebuild_requested:
                logger.info("User requested index rebuild. Starting rebuild process...")
                if indexer.rebuild_index(data_loader): # Assumes rebuild_index exists in Indexer
                    logger.info("Index rebuilt successfully.")
                    index_ready = True
                else:
                    logger.error("Index rebuild failed. Please check logs. Exiting.")
                    sys.exit(1) # Exit if rebuild fails
            else:
                # Only log loading message if we didn't just rebuild
                if not rebuild_requested:
                    logger.info("Loading existing index...")
                try:
                    indexer.load_index() # Directly load since we know it exists
                    logger.info("Existing index loaded successfully.")
                    index_ready = True
                except Exception as e:
                    logger.error(f"Failed to load existing index: {e}", exc_info=True)
                    logger.error("Exiting due to index load failure.")
                    sys.exit(1) # Exit if loading fails

        else:
            # Index does not exist, attempt to create it
            logger.info("No existing index found. Attempting to create a new index...")
            # setup_index should handle the creation process if index doesn't exist
            if indexer.setup_index(data_loader):
                # setup_index logs success/failure internally
                index_ready = True
            else:
                 logger.error("Index creation failed. Please check logs. Exiting.")
                 sys.exit(1) # Exit if initial creation fails

        # Start query loop only if the index is ready (loaded, rebuilt, or created)
        if index_ready:
            query_loop(query_engine)
        else:
             # This case should ideally be covered by the sys.exit calls above,
             # but added as a safeguard.
             logger.error("Index is not ready. Cannot start query loop.")

    except Exception as e:
        logger.error(f"Fatal error during application startup or execution: {e}", exc_info=True)
        sys.exit(1) # Exit on fatal errors

if __name__ == "__main__":
    main()
