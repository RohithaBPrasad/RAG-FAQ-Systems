# app.py
import sys
import os
import streamlit as st
import pandas as pd

# Ensure Python can find your src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_backend import init_backend, generate_answer_with_groq

# Streamlit page config
st.set_page_config(page_title="RAG FAQ Chat", layout="wide")

st.title("📚 RAG FAQ Assistant — Online Course Platform")
st.markdown("Ask anything about the platform; answers are generated from your FAQ dataset.")

# Initialize backend only once
if "initialized" not in st.session_state:
    with st.spinner("Initializing backend (loading embeddings / FAISS)..."):
        init_backend(csv_path=os.path.join("data", "faq.csv"))
    st.session_state.initialized = True

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: number of top results
top_k = st.sidebar.slider("Number of retrieved FAQs (top_k)", 1, 5, 3)

# User input
user_q = st.text_input("Type your question here:", placeholder="How do I enroll in a course?")

# When user presses Enter or clicks button
if st.button("Send") and user_q.strip():
    with st.spinner("Retrieving and generating answer..."):
        answer, faqs = generate_answer_with_groq(user_q, top_k=top_k)
    
    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "faqs": faqs})

# Display chat history in reverse (latest on top)
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


