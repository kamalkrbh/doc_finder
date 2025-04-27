# /home/kamal/doc_finder/ai_app/query_engine.py
import logging
from typing import List, Optional
from langchain.docstore.document import Document
from indexer import Indexer
from llm_handler import LLMHandler

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self, indexer: Indexer, llm_handler: LLMHandler):
        self.indexer = indexer
        self.llm_handler = llm_handler

    # get_relevant_docs remains structurally similar, but the 'docs' it returns
    # now represent whole PDFs based on the modified DataLoader.
    def get_relevant_docs(self, query: str) -> List[Document]:
        """
        Queries the index for documents (representing PDFs) most similar to the query.
        Returns a list of Document objects, where each Document's metadata contains the source PDF filename.
        """
        logger.debug(f"Searching index for PDFs relevant to query: '{query}'")
        if self.indexer.vectorstore is None:
             logger.error("Cannot search: Index is not loaded in the Indexer.")
             raise ValueError("Index not loaded or created.")

        try:
            # Search returns Document objects whose embeddings (based on full PDF text) are similar
            relevant_pdf_docs: List[Document] = self.indexer.search(query)
            logger.info(f"Indexer search returned {len(relevant_pdf_docs)} potentially relevant PDF documents for query: '{query}'.")
            if not relevant_pdf_docs:
                 logger.warning(f"Indexer search returned no relevant PDF documents for query: '{query}'")
            else:
                 # Log the filenames found
                 sources = [doc.metadata.get('source', 'Unknown Source') for doc in relevant_pdf_docs]
                 logger.debug(f"Found potentially relevant PDF sources: {sources}")
            return relevant_pdf_docs
        except Exception as e:
            logger.error(f"An unexpected error occurred during indexer.search for query '{query}': {e}", exc_info=True)
            return [] # Return empty list on unexpected search errors


    # Modified query method to suggest PDF filenames
    def query(self, user_query: str) -> str:
        """
        Retrieves relevant PDF documents based on the query, then asks the LLM
        to suggest which PDF filenames the user should check.
        """
        logger.info(f"Processing query to suggest relevant PDF files: '{user_query}'")
        try:
            # 1. Retrieve relevant PDF documents (each doc represents one PDF)
            # We call get_relevant_docs internally now, no need to pass docs in.
            relevant_pdf_docs = self.get_relevant_docs(user_query)

            # 2. Extract filenames and prepare prompt for LLM
            if not relevant_pdf_docs:
                logger.warning(f"No relevant PDF documents found for query: '{user_query}'")
                # Respond directly that no relevant files were found
                return "I could not find any PDF files in the index that seem relevant to your query."
            else:
                # Get the list of source filenames from the metadata
                relevant_filenames = [doc.metadata.get('source', 'Unknown Source') for doc in relevant_pdf_docs]
                # Remove duplicates and sort for consistent prompting
                unique_filenames = sorted(list(set(relevant_filenames)))

                logger.info(f"Suggesting the following PDF files based on relevance: {unique_filenames}")

                # 3. Construct the prompt for the LLM
                # Ask the LLM to suggest files based on the query and the list of potentially relevant filenames
                prompt = f"""Based on the user's query and an analysis of the content of available PDF documents, the following PDF files were identified as potentially relevant:
                        Potentially Relevant PDF Filenames:
                        - {"\n- ".join(unique_filenames)}
                        User Query: "{user_query}"
                        Please analyze the user's query and the list of filenames. Suggest which of these PDF files the user should look into to find the information they are seeking. Explain briefly why each suggested file might be relevant, if possible. If none seem particularly relevant despite being listed, state that. Be concise.
                        Suggested files to check:"""
                logger.info("Sending prompt to LLM to get PDF suggestions.")
                # Handle potential object response from LLM (like ChatGroq)
                response_obj = self.llm_handler.generate_response(prompt)
                response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

                logger.info("Received PDF suggestion response from LLM.")
                # Prepend the list of found files for clarity before the LLM's suggestion
                final_response = (f"Based on your query, the following PDF files might contain relevant information:\n"
                                  f"- {'\n- '.join(unique_filenames)}\n\n"
                                  f"LLM Suggestion:\n{response_text.strip()}")
                return final_response

        except ValueError as e: # Catch error if index not loaded during get_relevant_docs
             logger.error(f"Error during index search: {e}")
             return "Error: The document index is not available. Please ensure it has been created or loaded."
        except Exception as e:
            logger.error(f"An unexpected error occurred during query processing for query '{user_query}': {e}", exc_info=True)
            return "An unexpected error occurred while processing your query."

