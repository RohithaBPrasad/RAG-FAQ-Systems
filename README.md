# RAG FAQ System — Online Course Platform

This repository contains a Retrieval-Augmented Generation (RAG) FAQ chatbot:
- Embeddings: Sentence-BERT (all-MPNet-base-v2)
- Retrieval: FAISS
- Generation: Groq AI
- UI: Streamlit

## How to run locally (developer)
1. Create virtual env and install:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
