import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from backend import RAGBackend
from generator import GroqGenerator

st.set_page_config(page_title="RAG FAQ Chat", layout="wide")

# ------------------ LOAD BACKEND & GENERATOR ------------------
if "backend" not in st.session_state:
    with st.spinner("Loading embeddings & FAISS..."):
        st.session_state.backend = RAGBackend(data_csv=os.path.join("data", "faq.csv"))

if "generator" not in st.session_state:
    with st.spinner("Starting LLM generator..."):
        st.session_state.generator = GroqGenerator()

# ------------------ SESSION STATE ------------------
st.session_state.setdefault("all_chats", [])
st.session_state.setdefault("current_chat", [])

# ------------------ LIGHT THEME CSS & SCROLL ------------------
LIGHT_THEME_CSS = """
body { background-color: #ffffff; color: #000000; }
[data-testid="stSidebar"] { background-color: #f0f2f6; color: #000000; padding: 10px; }
h1, h2, h3, h4, h5, h6 { color: #000000; }

/* Main chat container */
.chat-container { 
    max-height: 500px; 
    overflow-y: auto; 
    padding: 10px; 
    border: 1px solid #ccc; 
    border-radius: 10px;
    margin-bottom: 10px;
}

/* Sidebar chat history */
.sidebar-chat-history {
    max-height: 300px;
    overflow-y: auto;
}

/* Chat bubbles */
.user-bubble { background-color: #DCF8C6; color: #000000; padding: 10px 14px; border-radius: 14px; margin: 8px 0; margin-left: 25%; width: fit-content; max-width: 70%; }
.bot-bubble { background-color: #E6E6E6; color: #000000; padding: 10px 14px; border-radius: 14px; margin: 8px 0; margin-right: 25%; width: fit-content; max-width: 70%; }
.context-box { background-color: #FAFAFA; border: 1px solid #CCC; padding: 8px; border-radius: 10px; margin: 6px 0 10px 0; }

/* Input & button */
.stTextInput>div>div>input { background-color: #ffffff; color: #000000; }
.stButton>button { background-color: #E6E6E6; color: #000000; }

.title-text { color: #000000; }
.user-emoji { color: #000000; }
.bot-emoji { color: #3333FF; }
"""
st.markdown(f"<style>{LIGHT_THEME_CSS}</style>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("💬 Chat Menu")

    # New Chat
    if st.button("➕ New Chat"):
        if st.session_state.current_chat:
            st.session_state.all_chats.append(st.session_state.current_chat.copy())
        st.session_state.current_chat = []

    # Dataset preview
    with st.expander("📌 View sample FAQ dataset"):
        df = pd.read_csv(os.path.join("data", "faq.csv"))
        st.dataframe(df.head(5))

    st.markdown("---")
    st.subheader("📁 Your Chats")
    
    # Scrollable chat history container
    st.markdown('<div class="sidebar-chat-history">', unsafe_allow_html=True)
    if st.session_state.all_chats:
        for idx, chat in enumerate(st.session_state.all_chats):
            # Show first 2 queries as summary
            summary = [msg["content"] for msg in chat if msg["role"] == "user"]
            summary_text = " | ".join(summary[:2]) + (" ..." if len(summary) > 2 else "")
            if st.button(f"Chat {idx+1}: {summary_text}"):
                st.session_state.current_chat = chat.copy()
    else:
        st.write("No previous chats yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ MAIN PAGE ------------------
st.markdown('<h1 class="title-text">📚 RAG FAQ Assistant — Online Course Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="title-text">Ask anything about the platform. The system uses your FAQ dataset to respond intelligently.</p>', unsafe_allow_html=True)

# Form for input to avoid session_state conflicts
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("💬 Your Question:", placeholder="Example: How do I enroll in a course?")
    submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        backend = st.session_state.backend
        generator = st.session_state.generator
        faqs = backend.retrieve(user_query)
        answer = generator.generate(user_query, faqs)

        st.session_state.current_chat.append({"role": "user", "content": user_query})
        st.session_state.current_chat.append({"role": "assistant", "content": answer, "faqs": faqs})

# ------------------ CHAT DISPLAY ------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.current_chat:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble"><span class="user-emoji">🧑💻</span> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble"><span class="bot-emoji">🤖</span> {msg["content"]}</div>', unsafe_allow_html=True)
        with st.expander("🔍 FAQ Context Used"):
            for i, f in enumerate(msg["faqs"]):
                st.markdown(f'<div class="context-box">**Q{i+1}.** {f["question"]}<br>➡️ **A:** {f["answer"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
