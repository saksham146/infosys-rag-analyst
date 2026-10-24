# ==========================================
# BIG STEP 1: IMPORTING REQUIRED LIBRARIES
# ==========================================
from pathlib import Path  # Safely handles file paths across Windows, Mac, and Linux
import pymupdf  # The core PDF reading engine (PyMuPDF / fitz)

# ==========================================
# BIG STEP 2: DEFINING FILE PATHS
# ==========================================
# 'r' stands for raw string, preventing Python from breaking on Windows backslashes (\)
PDF_PATH = Path(r"E:\AnnualRepot_Ai_project\Data\annual_report.pdf")


# ==========================================
# BIG STEP 3: THE MAIN INGESTION FUNCTION
# ==========================================
def ingest_annual_report(pdf_path, chunk_size=500, overlap=50):
  """Opens the PDF, extracts text page-by-page in reading order,

  and splits it into structured chunks with page metadata.
  """

  # --- Micro-Step 3.1: Safety Check & Opening ---
  # Check if the file actually exists before trying to open it
  if not pdf_path.exists():
    print("❌ Error: PDF not found at:", pdf_path)
    return []  # Return an empty list if file is missing

  # Load the 352-page PDF into Python's memory as a document object
  doc = pymupdf.open(pdf_path)
  all_chunks = []  # Empty list to hold our dictionary chunks later

  print(f"🚀 Processing {pdf_path.name} ({len(doc)} pages). Please wait...")

  # ==========================================
  # BIG STEP 4: LOOPING THROUGH EVERY PAGE
  # ==========================================
  # len(doc) = 352 pages. range() loops from index 0 to 351.
  for page_num in range(len(doc)):
    page = doc[page_num]  # Grab one single page at a time

    # ==========================================
    # BIG STEP 5: TEXT EXTRACTION & CLEANING
    # ==========================================

    # --- Micro-Step 5.1: Layout-Aware Extraction ---
    # sort=True forces PyMuPDF to arrange text logically top-to-bottom, left-to-right
    text = page.get_text("text", sort=True)

    # --- Micro-Step 5.2: Whitespace Normalization ---
    # text.split() breaks text by spaces/newlines; " ".join() squashes them into clean single spaces
    text = " ".join(text.split())

    # --- Micro-Step 5.3: Blank Page Filter ---
    # If a page has no text (e.g., blank cover/transition page), skip it
    if not text.strip():
      continue

    # ==========================================
    # BIG STEP 6: SLIDING WINDOW CHUNKING LOGIC
    # ==========================================
    # We chop the long page text into bite-sized pieces (default: 500 characters)
    start = 0
    while start < len(text):
      # Define the end boundary of the current chunk
      end = start + chunk_size

      # Slice the text from 'start' to 'end'
      chunk_text = text[start:end]

      # --- Micro-Step 6.1: Storing with Metadata ---
      # Save as a dictionary so we never lose track of which page it came from!
      all_chunks.append({
          "page": page_num + 1,  # Convert 0-index to human-readable 1-indexed page number
          "text": chunk_text,
      })

      # --- Micro-Step 6.2: Shifting the Window with Overlap ---
      # We step forward by (chunk_size - overlap). Example: 500 - 50 = 450 steps forward.
      # This ensures sentences aren't cut weirdly in half between chunks.
      start += chunk_size - overlap

  # ==========================================
  # BIG STEP 7: CLOSING & RETURNING
  # ==========================================
  doc.close()  # Always close the PDF file to free up computer memory (RAM)
  return all_chunks  # Send the full list of 2,652 chunks back to the program


# ==========================================
# BIG STEP 8: SCRIPT EXECUTION & VERIFICATION
# ==========================================
# This block runs only when you execute this script directly from your terminal/IDE
if __name__ == "__main__":
  print("📂 PDF file exists check:", PDF_PATH.exists())

  # --- Micro-Step 8.1: Run the Function ---
  chunks = ingest_annual_report(PDF_PATH)

  print(f"\n✅ Successfully ingested {len(chunks)} total chunks!")

  # --- Micro-Step 8.2: Print Previews to Verify Quality ---
  if chunks:
    print("\n--- Preview of Chunk #1 ---")
    print(f"Page number : {chunks[0]['page']}")
    print(f"Chunk text  : {chunks[0]['text']}")

    print("\n--- Preview of Chunk #50 ---")
    if len(chunks) > 50:
      print(f"Page number : {chunks[49]['page']}")  # Index 49 = Chunk #50
      print(f"Chunk text  : {chunks[49]['text']}")


#=========================================
#Big Step 9: Embeddings & Vector Storage
#=========================================







