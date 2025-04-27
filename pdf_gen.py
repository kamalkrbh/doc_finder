import os
import random
import time
from fpdf import FPDF
import unicodedata
from fpdf.enums import XPos, YPos
# Assuming this script is in the root 'doc_finder' directory
# Adjust imports if you place the script elsewhere
try:
    from config import config
    from llm_handler import LLMHandler
except ModuleNotFoundError:
    print("Error: Make sure this script is run from the 'doc_finder' parent directory,")
    print("or adjust the import paths for 'ai_app.config' and 'ai_app.llm_handler'.")
    exit()


# --- Configuration ---
PDF_DIRECTORY = config.PDF_DIRECTORY # Use directory from config
NUM_PDFS_TO_CREATE = 50      # How many dummy PDFs to create
STORY_WORD_COUNT_TARGET = 1000     # Aim for stories around this length (LLM dependent)
RETRY_DELAY_SECONDS = 8            # Delay before retrying LLM call on failure
MAX_RETRIES = 3                    # Max retries for LLM generation
# --- End Configuration ---

def get_prompt_for_pdf_gen():
    doc_type = random.choice(["story", "personal"])
    if doc_type == "story":
        return get_prompt_for_story_generation(), doc_type
    else:
        return get_personal_doc_prompt(), doc_type

def get_prompt_for_pdf_title(doc_type, pdf_content):
    if doc_type == "story":
        return f"Generate a creative and engaging title with max 10 words for the following {doc_type} content:\n\n" \
                "if more than one titles generated choose first one from the suggested titles:\n\n" \
                "no filler text, no explanation, no extra text, just the title\n\n" \
                        f"{pdf_content[:500]}"
    else:
        return f"Generate only one concise and relevant title with max 10 words for the following {doc_type} content, " \
                "if more than one titles generated choose first one from the suggested titles:\n\n" \
                "no filler text, no explanation, no extra text, just the title\n\n" \
                        f"{pdf_content[:500]}"
                        
def get_prompt_for_story_generation():
    """Creates a random prompt for a short story."""
    theme = random.choice([
        "a curious squirrel discovering a hidden map",
        "a lonely lighthouse keeper receiving a mysterious message in a bottle",
        "a group of kids building a time machine in their treehouse",
        "an old bookstore where characters step out of their books at night",
        "a chef who can cook emotions into food",
        "an alien trying to understand human holidays",
        "a musician whose songs can alter reality",
        "a detective investigating strange occurrences in a quiet town",
        "a gardener growing plants that bloom with light",
        "a robot learning about friendship from a child",
        "a painter whose artwork comes to life",
        "a librarian who can travel through stories",
        "a cat that can talk to ghosts",
        "a child who can see the future in their dreams",
        "a baker who creates pastries that grant wishes",
        "a scientist discovering a portal to another dimension",
        "a time traveler visiting ancient civilizations",
        "a dragon who loves to paint landscapes",
    ])
    location = random.choice([
        "in a bustling futuristic city of india",
        "on a remote, mist-covered island in america",
        "deep within an enchanted forest in african real life",
        "aboard a generation starship",
        "in a steampunk-inspired Victorian London",
        "in a hidden underwater kingdom",
        "in a quaint village during a festival",
        "in a magical library that changes its layout every night",
        "in a world where dreams and reality intertwine",
        "in a desert where time stands still",
    ])
    
    country = random.choice([
        "India",
        "USA",
        "UK",
        "France",
        "Germany",
        "Japan",
        "Brazil",
        "Australia",
        "South Africa",
        "Canada"
    ])
    
    timeline = random.choice([
        "future in the year 2050",
        "1800s",
        "1700s",
        "2000s",
        'medieval times',
    ])
    
    return (f"Write a fictional story approximately {STORY_WORD_COUNT_TARGET} words long "
            f"about {theme} {location} {country} during {timeline}. The story should have a clear beginning, "
            f"middle, and end. Be creative and engaging.")

def get_personal_doc_prompt():
    """Creates a random prompt for a personal document."""

    document_types = [
        "medical report",
        "travel plan",
        "financial statements",
        "personal diary entry",
        "resume",
        "cover letter",
        "business proposal",
        "project report",
        "meeting minutes",
        "email correspondence",
        "research paper",
        "blog post",
        "social media post",
        "newsletter",
        "presentation slides",
        "work contract",
        "salary slip",
        "salary statement",
        "ID card",
        "passport",
        "birth certificate",
        "marriage certificate",
    ]
    
    type = random.choice(document_types)
    return (f"Generate a {type} of a dummy person (dont use realworld names instead use xyz or abc like text to replace the person name) with dummy details "
            f"like age city and country, "
            f"dont use real street names or house numbers while generating addresses for example. "
            f"Make it detailed and engaging.")

def normalize_text(text: str) -> str:
    """
    Replaces common problematic characters with simpler equivalents
    and normalizes Unicode to potentially reduce unsupported characters.
    """
    replacements = {
        '–': '-',  # En dash to hyphen
        '—': '-',  # Em dash to hyphen
        '‘': "'",  # Left single quote
        '’': "'",  # Right single quote / apostrophe
        '“': '"',  # Left double quote
        '”': '"',  # Right double quote
        '…': '...', # Ellipsis
        '₹': 'Rs.', # Indian Rupee Sign to "Rs." (or "INR", or "" to remove) <-- ADDED
        # Add more specific symbol replacements as needed
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # Keep normalization for other potential issues (like combined characters)
    text = unicodedata.normalize('NFKC', text)

    # Avoid the aggressive ASCII ignore unless absolutely necessary,
    # as it removes characters like the original '₹' if not replaced above.
    # text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    return text

def create_pdf(filename: str, title: str, story_text: str):
    """Creates a PDF document, attempting to normalize text for basic fonts."""
    if not story_text or not story_text.strip():
        print(f"Skipping PDF creation for {filename} due to empty story content.")
        return False

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Use Basic Font ---
    pdf.set_font("Helvetica", size=12)
    title_font = "Helvetica"
    # --- End Font Setup ---

    # Normalize Title and Story Text
    normalized_title = normalize_text(title)
    normalized_story_text = normalize_text(story_text)

    # Add Title
    pdf.set_font(title_font, 'B', 16)
    try:
        # Use normalized text and updated multi_cell call
        pdf.multi_cell(0, 10, normalized_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    except Exception as e: # Catch potential errors even after normalization
        print(f"Warning: Error writing normalized title '{normalized_title[:30]}...' to PDF. Error: {e}")
        # Fallback using latin-1 replace on the *normalized* text
        pdf.multi_cell(0, 10, normalized_title.encode('latin-1', 'replace').decode('latin-1'), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    # Add Story Text
    pdf.set_font(title_font, size=12)
    try:
        # Use normalized text
        pdf.multi_cell(0, 5, normalized_story_text)
    except Exception as e:
         print(f"Error during PDF multi_cell generation for {filename} even after normalization: {e}")
         # Attempt fallback encoding as a last resort
         try:
             cleaned_text = normalized_story_text.encode('latin-1', 'replace').decode('latin-1')
             pdf.multi_cell(0, 5, cleaned_text)
             print(f"Warning: Used latin-1 fallback encoding for story text in {filename} due to error.")
         except Exception as fallback_err:
              print(f"ERROR: Could not write story text even with fallback for {filename}. Error: {fallback_err}")
              return False

    # Save the PDF
    try:
        pdf.output(filename)
        print(f"Successfully created PDF: {filename}")
        return True
    except Exception as e:
        print(f"Error saving PDF {filename}: {e}")
        return False
    
def generate_title_from_content(llm_handler, title_prompt: str) -> str:
    """Generates a title for the pdf based on its content using the LLM."""
    try:
        title_response = llm_handler.generate_response(title_prompt)
        if title_response:
            return title_response.strip()
        else:
            print("LLM returned an empty title. Falling back to default title.")
            return "Untitled PDF"
    except ValueError as e:
        print(f"ValueError during title generation: {e}")
        return "Untitled PDF"

if __name__ == "__main__":
    
    if os.path.exists(PDF_DIRECTORY):
        print(f"Warning: Directory '{PDF_DIRECTORY}' already exists. PDFs will be created here.")
        ## delete existing PDFs
        for file in os.listdir(PDF_DIRECTORY):
            if file.endswith(".pdf"):
                os.remove(os.path.join(PDF_DIRECTORY, file))
            print(f"Deleted existing PDF: {file}")
    else:
        print(f"Creating directory for PDFs: {PDF_DIRECTORY}")
        try:
            os.makedirs(PDF_DIRECTORY)
            print(f"Created directory: {PDF_DIRECTORY}")
        except OSError as e:
            print(f"Error creating directory {PDF_DIRECTORY}: {e}")
            exit()

    # Initialize LLM Handler
    llm_handler = None
    try:
        llm_handler = LLMHandler()
        print(f"Using LLM Provider: {config.LLM_PROVIDER}")
        # Quick test call if needed (optional, might incur cost)
        # llm_handler.generate_response("Test prompt")
        # print("LLM Handler initialized successfully.")
    except ValueError as e:
        print(f"Error initializing LLM Handler: {e}")
        print("Please ensure your LLM provider and API key (if needed) are configured correctly in environment variables or .env file.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred initializing LLM Handler: {e}")
        exit()

    print(f"\nGenerating {NUM_PDFS_TO_CREATE} dummy PDFs in '{PDF_DIRECTORY}'...")

    pdfs_created_count = 0
    for i in range(1, NUM_PDFS_TO_CREATE + 1):
        # story_title = f"The Tale of {random.choice(['Whispering Woods', 'Starlight City', 'Clockwork Dreams', 'Oceanic Secrets'])} - Part {i}"
        # #pdf_filename = os.path.join(PDF_DIRECTORY, f"dummy_story_{i:03d}.pdf")
        # pdf_filename=story_title.replace(" ", "_") + ".pdf"
        # print(f"\n--- Generating Story {i}/{NUM_PDFS_TO_CREATE} for {pdf_filename} ---")
        prompt, doc_type = get_prompt_for_pdf_gen()
        print(f"Prompt: {prompt[:100]}...") # Print start of prompt

        pdf_content = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                pdf_content = llm_handler.generate_response(prompt)
                if pdf_content and pdf_content:
                     print(f"Content generated (length: {len(pdf_content)} chars).")
                else:
                    print("LLM returned empty content.")
                    pdf_content = None # Ensure it's None for retry logic
            except Exception as e:
                print(f"Error generating content (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                if attempt < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print("Max retries reached. Skipping this PDF.")

        if pdf_content:
            print("Creating PDF...")
            print("Generating title from pdf content...")
            story_title = generate_title_from_content(llm_handler, get_prompt_for_pdf_title(doc_type, pdf_content))
            pdf_filename = story_title.replace(" ", "_").replace("\"", "") + ".pdf"
            print(f"Generated Title: {story_title}")
            if create_pdf(PDF_DIRECTORY+'/'+pdf_filename, story_title, pdf_content):
                pdfs_created_count += 1
        else:
            print(f"Failed to generate pdf content for after retries.")

    print(f"\n--- Generation Complete ---")
    print(f"Successfully created {pdfs_created_count} out of {NUM_PDFS_TO_CREATE} requested PDFs in '{PDF_DIRECTORY}'.")

