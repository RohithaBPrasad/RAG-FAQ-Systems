# app.py
import sys
import os
import streamlit as st
import pandas as pd

# ensure Python can import package files in src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import classes (not nonexistent functions)
from backend import RAGBackend
from generator import GroqGenerator

# Streamlit page config
st.set_page_config(page_title="RAG FAQ Chat", layout="wide")

st.title("📚 RAG FAQ Assistant — Online Course Platform")
st.markdown("Ask anything about the platform; answers are generated from your FAQ dataset.")

# Initialize backend and generator once and keep in session_state
if "backend" not in st.session_state:
    with st.spinner("Initializing backend (loading embeddings / FAISS)..."):
        st.session_state.backend = RAGBackend(data_csv=os.path.join("data", "faq.csv"))
if "generator" not in st.session_state:
    with st.spinner("Initializing LLM generator..."):
        st.session_state.generator = GroqGenerator()  # will read GROQ_API_KEY from env

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: number of top results
top_k = st.sidebar.slider("Number of retrieved FAQs (top_k)", 1, 5, 3)

# User input
user_q = st.text_input("Type your question here:", placeholder="How do I enroll in a course?")

# Send button handling
if st.button("Send") and user_q.strip():
    with st.spinner("Retrieving and generating answer..."):
        backend = st.session_state.backend
        generator = st.session_state.generator

        faqs = backend.retrieve(user_q, top_k=top_k)
        answer = generator.generate(user_q, faqs)

    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "faqs": faqs})

# Display chat history in reverse (latest first)
for chat in reversed(st.session_state.chat_history):
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**RAG Assistant:** {chat['content']}")
        st.markdown("**Context retrieved:**")
        for i, f in enumerate(chat["faqs"]):
            st.markdown(f"{i+1}. **Q:** {f['question']}")
            st.markdown(f"   **A:** {f['answer']}")
        st.write("---")

# Optional: show dataset
if st.checkbox("Show sample FAQs (first 10)"):
    df = pd.read_csv(os.path.join("data", "faq.csv"))
    st.dataframe(df.head(10))
