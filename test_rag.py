# ==========================================
# TEST RUNNER FOR ENTERPRISE RAG PIPELINE
# ==========================================
from src.rag_pipeline import ask_annual_report


def run_test():
  # Highly specific financial statement query with a wider search window (top_k=12)
  question = (
      "What are the figures for revenue from operations, total income, and net"
      " profit in the Consolidated Statement of Profit and Loss for FY 2024?"
  )

  print(f"🚀 Running precise financial query with top_k=12...")
  answer, source_pages = ask_annual_report(question, top_k=12)

  print("\n" + "=" * 60)
  print("💡 RAG SYSTEM ANSWER:")
  print("=" * 60)
  print(answer)
  print("=" * 60)
  print(f"📖 Reference Pages Cited: {source_pages}")


if __name__ == "__main__":
  run_test()