# 🔧 Streamlit interface for testing LLM entity extraction

import os
import sys
import streamlit as st

# Add path to src folder for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "nlp")))

from llm_ner import extract_entities_with_openai

# -----------------------------
# 🚀 Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI Travel Assistant", page_icon="🧠")

st.title("🧳 AI Travel Assistant")
st.write("This assistant extracts structured travel information from your query using OpenAI's LLM.")

user_query = st.text_input("✈️ Enter your travel request:")

if user_query:
    with st.spinner("🧠 Analyzing your request..."):
        result = extract_entities_with_openai(user_query)
    st.success("✅ Extraction complete!")
    st.subheader("📦 Extracted Entities:")
    st.json(result)
