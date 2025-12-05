# src/backend.py
import pickle
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = os.getenv("EMBED_MODEL", "all-MPNet-base-v2")

class RAGBackend:
    def __init__(self, data_csv='data/faq.csv', embeddings_npy='data/embeddings.npy', faiss_idx='data/faiss_index.idx'):
        self.df = pd.read_csv(data_csv)
        # try to load saved embeddings & index; if not present, compute
        self.embed_model = SentenceTransformer(MODEL_NAME)
        try:
            self.embedding_matrix = np.load(embeddings_npy)
            self.index = faiss.read_index(faiss_idx)
        except Exception as e:
            # fallback: compute embeddings and create index (slower)
            print("Saved embeddings/index not found, computing embeddings... ", e)
            self._compute_embeddings_and_index(embeddings_npy, faiss_idx)

    def _compute_embeddings_and_index(self, embeddings_npy, faiss_idx):
        texts = self.df['question'].tolist()
        emb = self.embed_model.encode(texts, show_progress_bar=True)
        self.embedding_matrix = np.vstack([e for e in emb]).astype('float32')
        np.save(embeddings_npy, self.embedding_matrix)
        dim = self.embedding_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embedding_matrix)
        faiss.write_index(self.index, faiss_idx)

    def retrieve(self, query, top_k=3):
        qv = self.embed_model.encode([query]).astype('float32')
        distances, indices = self.index.search(qv, top_k)
        results = []
        for i in indices[0]:
            results.append({
                "question": self.df.iloc[i]['question'],
                "answer": self.df.iloc[i]['answer'],
                "idx": int(i)
            })
        return results
