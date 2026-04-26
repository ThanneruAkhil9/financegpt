"""
rag.py - Retrieval Augmented Generation pipeline for FinanceGPT
Uses Groq for LLM + HuggingFace Inference API for embeddings (no torch, no fastembed)
"""

import os
import requests
import numpy as np
from groq import Groq
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Constants
COLLECTION  = "financegpt"
TOP_K       = 5
GROQ_MODEL  = "llama-3.3-70b-versatile"

HF_EMBED_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

SEBI_DISCLAIMER = (
    "\n\n---\n WARNING: This is for educational purposes only and does not "
    "constitute investment advice. Mutual fund investments are subject to market risks. "
    "Please consult a SEBI-registered Investment Adviser for personalized advice."
)

SYSTEM_PROMPT = """You are FinanceGPT - an India-first multilingual finance assistant.

Your role:
- Answer questions about Indian personal finance, mutual funds, stocks, taxes, banking, crypto, regulations, macroeconomics, loans, and financial planning.
- Be clear, accurate, and helpful to both beginners and experienced investors.
- Respond in the SAME LANGUAGE the user writes in (Hindi, Telugu, Tamil, English, etc.)
- Use simple language; avoid unnecessary jargon unless explaining technical terms.
- Always base your answer on the provided context documents.

What you MUST NOT do:
- Give stock tips or say "buy this stock"
- Provide F&O (futures/options) signals
- Act as a portfolio manager
- Give personalized investment advice ("you should invest X in Y")
- Make up numbers - if you don't know, say so clearly

Confidence levels to use:
- High Confidence: For well-established facts (tax slabs, how SIPs work, SEBI rules)
- Verify Recommended: For numbers that may have changed (interest rates, tax rates)
- Consult a CA/Expert: For complex personal tax situations

Format your answers clearly with bullet points or numbered steps when explaining processes.
"""

# Clients initialized once at startup
print("Connecting to Qdrant...")
_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print("Connecting to Groq...")
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

_hf_token = os.getenv("HF_TOKEN", "")
_hf_headers = {"Authorization": f"Bearer {_hf_token}"} if _hf_token else {}

print("FinanceGPT RAG pipeline ready (lightweight mode)")


def get_embedding(text: str) -> list:
    """Get embedding vector via HuggingFace Inference API."""
    try:
        response = requests.post(
            HF_EMBED_URL,
            headers=_hf_headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result[0], list):
            vector = np.mean(result, axis=0).tolist()
        else:
            vector = result

        return vector

    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 384


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """Embed the query and fetch top_k matching documents from Qdrant."""
    vector = get_embedding(query)
    results = _qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results]


def build_context(docs: list) -> str:
    """Format retrieved docs into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Doc {i} | {doc.get('domain', '')} | {doc.get('title', '')}]\n{doc.get('content', '')}"
        )
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    conversation_history: list,
    use_fallback: bool = False,
) -> tuple:
    """
    Main RAG function.
    Returns: (answer_text, retrieved_docs)
    """
    docs = retrieve(query)
    context = build_context(docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in conversation_history[-6:]:
        messages.append(turn)

    user_message = f"""Context from FinanceGPT knowledge base:
{context}

User Question: {query}

Answer based on the context above. If the context does not fully cover the question, say so clearly and answer from general knowledge. Add confidence level."""

    messages.append({"role": "user", "content": user_message})

    try:
        model = GROQ_MODEL
        if use_fallback:
            model = "gemma2-9b-it"

        response = _groq.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"I encountered an error: {str(e)}. Please try again."

    final_answer = answer + SEBI_DISCLAIMER

    return final_answer, docs
