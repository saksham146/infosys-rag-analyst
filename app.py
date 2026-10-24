# ==========================================
# STREAMLIT WEB UI FOR INFOSYS RAG SYSTEM
# ==========================================
import os
import streamlit as st
from src.rag_pipeline import ask_annual_report

# Page configuration
st.set_page_config(
    page_title="Infosys Annual Report AI Analyst",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Infosys FY 2024 Annual Report AI Analyst")
st.markdown(
    "Ask detailed financial and operational questions about the Infosys"
    " 2023-24 Annual Report, powered by local ChromaDB and Google Gemini."
)

# Sidebar settings for advanced control
with st.sidebar:
  st.header("⚙️ RAG Configuration")
  top_k_val = st.slider(
      "Chunks to Retrieve (top_k)",
      min_value=3,
      max_value=15,
      value=8,
      help=(
          "Number of text chunks fetched from ChromaDB before sending to"
          " Gemini."
      ),
  )

  st.markdown("---")
  st.markdown("### 💡 Sample Questions:")
  st.markdown(
      "- *What are the figures for revenue from operations and net profit?*"
  )
  st.markdown("- *What is the total employee count and attrition rate?*")
  st.markdown("- *What are the major risks discussed by management?*")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
    if "sources" in message and message["sources"]:
      st.caption(f"📖 Cited Pages: {message['sources']}")

# Accept user input
if prompt := st.chat_input(
    "Ask a question about the Infosys Annual Report..."
):
  # Append user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Generate assistant response
  with st.chat_message("assistant"):
    with st.status(
        "🔍 Searching vector database and consulting Gemini...", expanded=True
    ) as status:
      try:
        # Call your existing RAG pipeline backend function
        answer, source_pages = ask_annual_report(prompt, top_k=top_k_val)
        status.update(label="✅ Analysis complete!", state="complete")
      except Exception as e:
        status.update(label="❌ Error generating response.", state="error")
        answer = f"An error occurred: {e}"
        source_pages = []

    # Display the answer
    st.markdown(answer)
    if source_pages:
      st.caption(f"📖 Cited Pages: {source_pages}")

    # Save assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_pages,
    })