# ==========================================
# STEP 4: VECTOR DATABASE & SEMANTIC SEARCH
# ==========================================
from pathlib import Path
import chromadb
from Embed import generate_embeddings

# Path for persistent vector database storage on your disk
DB_PATH = Path(r"E:\AnnualRepot_Ai_project\Data\chroma_db")


def setup_vector_database():
    print("🔄 Generating/Loading embeddings for vector database...")
    chunks = generate_embeddings()

    if not chunks:
        print("❌ No chunks available to store.")
        return None

    # 1. Initialize ChromaDB client (stores data persistently on your disk)
    print(f"\n💾 Initializing ChromaDB at: {DB_PATH}")
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # 2. Create or reset a collection (our vector table)
    collection_name = "infosys_annual_report_2024"

    # Delete collection if it already exists to avoid duplication errors during testing
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    # 3. Prepare data in batches for ChromaDB
    print(f"📦 Storing {len(chunks)} chunks into ChromaDB...")

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({"page": int(chunk["page"])})
        embeddings.append(chunk["vector"].tolist())  # ChromaDB expects standard python lists

    # Add everything to the vector database in one go
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("✅ Successfully stored all vectors into ChromaDB!")
    return collection


def search_vector_database(collection, query_text, top_k=3):
    """Searches the vector database using semantic similarity."""
    from sentence_transformers import SentenceTransformer

    print(f"\n🔍 Searching for: '{query_text}'")

    # Load the same model to convert the user's text question into a vector
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(query_text).tolist()

    # Query ChromaDB for the closest matching vectors
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    return results


if __name__ == "__main__":
    # 1. Build and populate the database
    col = setup_vector_database()

    if col:
        # 2. Test a sample semantic search query
        sample_query = "What is the revenue growth and financial performance of Infosys?"
        search_results = search_vector_database(col, sample_query, top_k=3)

        print("\n" + "=" * 60)
        print("🎯 TOP SEARCH RESULTS:")
        print("=" * 60)

        for i in range(len(search_results['documents'][0])):
            doc_text = search_results['documents'][0][i]
            page_num = search_results['metadatas'][0][i]['page']
            distance = search_results['distances'][0][i]

            print(f"\n--- Result #{i + 1} (Source Page: {page_num}) ---")
            print(f"Distance Score (lower means closer match): {distance:.4f}")
            print(f"Text Snippet: {doc_text}")
            print("-" * 60)