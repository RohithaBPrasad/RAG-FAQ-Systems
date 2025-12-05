# src/generator.py
import os
import groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class GroqGenerator:
    def __init__(self, api_key=None, max_tokens=200):
        if api_key is None:
            api_key = GROQ_API_KEY
        if api_key is None:
            raise ValueError("Set GROQ_API_KEY as environment variable")
        self.client = groq.Client(api_key=api_key)
        self.max_tokens = max_tokens

    def generate(self, query, context_faqs):
        # context_faqs: list of {question, answer}
        context_text = ""
        for faq in context_faqs:
            context_text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
        prompt = f"""You are an expert support assistant for an online course platform.
Use the following FAQs to answer the user's question. If the answer is not in the FAQs, be honest and say you don't know, and suggest contacting support.

Context:
{context_text}

User Question: {query}
Answer:"""
        resp = self.client.generate(prompt=prompt, max_tokens=self.max_tokens)
        return resp.text
