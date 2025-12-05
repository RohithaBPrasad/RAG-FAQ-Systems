# src/generator.py
import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class GroqGenerator:
    def __init__(self, api_key=None, max_tokens=200, model="llama-3.1-8b-instant"):
        if api_key is None:
            api_key = GROQ_API_KEY
        if api_key is None:
            raise ValueError("Set GROQ_API_KEY as environment variable")
        self.client = Groq(api_key=api_key)
        self.max_tokens = max_tokens
        self.model = model

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens
        )

        # return the assistant text
        return response.choices[0].message.content.strip()
