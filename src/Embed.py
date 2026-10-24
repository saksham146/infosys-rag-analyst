# ==========================================
# STEP 3: EMBEDDINGS & VECTOR GENERATION
# ==========================================
from pathlib import Path
import time
from ingest import ingest_annual_report  # Import your existing ingestion script!
from sentence_transformers import SentenceTransformer

# Path to PDF
PDF_PATH = Path(r"E:\AnnualRepot_Ai_project\Data\annual_report.pdf")


def generate_embeddings():
  # 1. Run your ingestion function to get all chunks
  print("📥 Loading and chunking PDF...")
  chunks = ingest_annual_report(PDF_PATH)

  if not chunks:
    print("❌ No chunks found to embed.")
    return

  # 2. Load the lightweight local embedding model
  print("\n🤖 Loading local embedding model ('all-MiniLM-L6-v2')...")
  model = SentenceTransformer("all-MiniLM-L6-v2")

  # 3. Extract just the text from all chunks so the model can process them in bulk
  print(f"\n🔄 Generating vectors for {len(chunks)} chunks... (Please wait)")
  start_time = time.time()

  # We pull out the 'text' property from each dictionary
  texts_to_embed = [chunk["text"] for chunk in chunks]

  # model.encode converts all text strings into numerical vectors simultaneously (batch processing)
  embeddings = model.encode(texts_to_embed, show_progress_bar=True)

  end_time = time.time()

  print(f"\n✅ Successfully generated embeddings!")
  print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
  print(f"📊 Total vectors created: {len(embeddings)}")
  print(f"📐 Shape/Dimensions of each vector: {embeddings[0].shape}")  # Should be (384,)

  # Let's attach the vectors back to our chunk dictionaries for safe keeping
  for i, chunk in enumerate(chunks):
    chunk["vector"] = embeddings[i]

  return chunks


if __name__ == "__main__":
  embedded_chunks = generate_embeddings()

  if embedded_chunks:
    print("\n--- Verification of Embedded Chunk #1 ---")
    print(f"Page Number : {embedded_chunks[0]['page']}")
    print(f"Chunk Text  : {embedded_chunks[0]['text'][:100]}...")
    print(
        f"Vector Preview (First 5 numbers): {embedded_chunks[0]['vector'][:5]}"
    )



