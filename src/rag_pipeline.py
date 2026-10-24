# ==========================================
# STEP 5: ENTERPRISE RAG QA PIPELINE
# ==========================================
import os
import time
from pathlib import Path
import chromadb
from google import genai
from google.genai.errors import ServerError
from sentence_transformers import SentenceTransformer

# Paths and configuration
DB_PATH = Path(r"E:\AnnualRepot_Ai_project\Data\chroma_db")
COLLECTION_NAME = "infosys_annual_report_2024"


def initialize_rag_system():
  """Initializes the ChromaDB client and embedding model."""
  print("🔌 Connecting to Vector Database...")
  client = chromadb.PersistentClient(path=str(DB_PATH))
  collection = client.get_collection(name=COLLECTION_NAME)

  print("🤖 Loading embedding model for queries...")
  embed_model = SentenceTransformer("all-MiniLM-L6-v2")

  return collection, embed_model


def generate_with_retry(ai_client, prompt, primary_model="gemini-3.8-flash"):
  """Attempts generation with retries on 503 error, falling back if necessary."""
  models_to_try = [primary_model, "gemini-2.5-flash"]

  for model_id in models_to_try:
    print(f"🧠 Generating response with model: {model_id}...")
    for attempt in range(1, 4):  # Try up to 3 times per model
      try:
        response = ai_client.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        return response.text
      except ServerError as e:
        if "503" in str(e) or "high demand" in str(e):
          print(
              f"⚠️ High demand on {model_id} (Attempt {attempt}/3). Retrying in"
              " 3 seconds..."
          )
          time.sleep(3)
        else:
          raise e
      except Exception as e:
        print(f"⚠️ Error with {model_id}: {e}")
        break  # Fall back to next model

  raise RuntimeError(
      "❌ All model generation attempts failed due to server demand."
  )


def ask_annual_report(query_text, top_k=3):
  """Retrieves relevant chunks from ChromaDB and generates a cited answer using Gemini."""
  # 1. Initialize vector store and embedding model
  collection, embed_model = initialize_rag_system()

  # 2. Convert query to vector and search ChromaDB
  print(f"\n🔍 Searching annual report for: '{query_text}'")
  query_vector = embed_model.encode(query_text).tolist()

  results = collection.query(query_embeddings=[query_vector], n_results=top_k)

  # 3. Extract retrieved documents and page metadata
  retrieved_docs = results["documents"][0]
  retrieved_pages = results["metadatas"][0]

  # 4. Build context string with explicit page citations for the LLM
  context_blocks = []
  sources_used = set()

  for i, doc in enumerate(retrieved_docs):
    page_num = retrieved_pages[i]["page"]
    sources_used.add(int(page_num))
    context_blocks.append(f"[Page {page_num}]\n{doc}")

  combined_context = "\n\n---\n\n".join(context_blocks)

  # 5. Construct prompt for Gemini
  prompt = f"""
You are an expert Financial Analyst assistant specializing in corporate annual reports. 
Answer the user's question accurately using ONLY the provided context from the Infosys Annual Report 2023-24. 
Do not assume or make up facts. If the answer is not present in the context, state clearly that it is not available in the report.
Always cite the specific page numbers (e.g., [Page X]) where you found the information.

=== CONTEXT ===
{combined_context}

=== USER QUESTION ===
{query_text}

=== FINANCIAL ANSWER (WITH PAGE CITATIONS) ===
"""

  # 6. Initialize Gemini Client safely
  api_key = os.environ.get(
      "GEMINI_API_KEY", "AIzaSyB0plS25t8YjFPJSiQuMSRPK7o_9n9DB4c"
  )
  ai_client = genai.Client(api_key=api_key)

  # Generate content with built-in retry and model fallback
  answer_text = generate_with_retry(ai_client, prompt)

  return answer_text, sorted(list(sources_used))


if __name__ == "__main__":
  question = (
      "What was the revenue growth and overall financial performance of Infosys"
      " this year?"
  )

  answer, source_pages = ask_annual_report(question, top_k=3)

  print("\n" + "=" * 60)
  print("💡 RAG SYSTEM ANSWER:")
  print("=" * 60)
  print(answer)
  print("=" * 60)
  print(f"📖 Reference Pages Cited: {source_pages}")