"""
rag.py — Retrieval Augmented Generation pipeline for FinanceGPT
"""
 
import os
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from dotenv import load_dotenv
 
load_dotenv()
 
# ─── Constants ───────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION   = "financegpt"
TOP_K        = 5          # how many docs to retrieve
GROQ_MODEL   = "llama-3.3-70b-versatile"
 
SEBI_DISCLAIMER = (
    "\n\n---\n⚠️ **SEBI Disclaimer:** This is for educational purposes only and does not "
    "constitute investment advice. Mutual fund investments are subject to market risks. "
    "Please consult a SEBI-registered Investment Adviser for personalized advice."
)
 
SYSTEM_PROMPT = """You are FinanceGPT — an India-first multilingual finance assistant.
 
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
- Make up numbers — if you don't know, say so clearly
 
Confidence levels to use:
- ✅ High Confidence: For well-established facts (tax slabs, how SIPs work, SEBI rules)
- ⚠️ Verify Recommended: For numbers that may have changed (interest rates, tax rates)  
- 📞 Consult a CA/Expert: For complex personal tax situations
 
Format your answers clearly with bullet points or numbered steps when explaining processes.
"""
 
# ─── Load models once at startup (not on every request) ──────
print("Loading embedding model...")
_embed_model = TextEmbedding(model_name=EMBED_MODEL)
 
print("Connecting to Qdrant...")
_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
 
print("Connecting to Groq...")
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
 
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and fetch top_k matching documents from Qdrant."""
    vector = list(_embed_model.embed([query]))[0].tolist()
    results = _qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results]
 
 
def build_context(docs: list[dict]) -> str:
    """Format retrieved docs into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Doc {i} | {doc['domain']} | {doc['title']}]\n{doc['content']}"
        )
    return "\n\n".join(parts)
 
 
def generate_answer(
    query: str,
    conversation_history: list[dict],
    use_fallback: bool = False,
) -> tuple[str, list[dict]]:
    """
    Main RAG function.
    Returns: (answer_text, retrieved_docs)
    """
    # Step 1: Retrieve relevant docs
    docs = retrieve(query)
 
    # Step 2: Build context
    context = build_context(docs)
 
    # Step 3: Build messages for Groq
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    # Add conversation history (last 6 turns to stay within token limits)
    for turn in conversation_history[-6:]:
        messages.append(turn)
 
    # Add current query with context
    user_message = f"""Context from FinanceGPT knowledge base:
{context}
 
User Question: {query}
 
Answer based on the context above. If the context doesn't fully cover the question, say so clearly and answer from general knowledge. Add confidence level."""
 
    messages.append({"role": "user", "content": user_message})
 
    # Step 4: Call Groq (with fallback model if requested)
    try:
        model = GROQ_MODEL
        if use_fallback:
            model = "gemma2-9b-it"  # Groq's Gemma as fallback
 
        response = _groq.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,  # low temperature = more factual
        )
        answer = response.choices[0].message.content
 
    except Exception as e:
        answer = f"I encountered an error: {str(e)}. Please try again."
 
    # Step 5: Append SEBI disclaimer
    final_answer = answer + SEBI_DISCLAIMER
 
    return final_answer, docs
 
