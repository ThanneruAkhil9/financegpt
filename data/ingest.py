"""
ingest.py — Load finance data into Qdrant vector database
Run this ONCE before starting the app: python data/ingest.py
"""

import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─── Config ──────────────────────────────────────────────────
QDRANT_URL    = os.getenv("QDRANT_URL")       # from Qdrant Cloud dashboard
QDRANT_API    = os.getenv("QDRANT_API_KEY")   # from Qdrant Cloud dashboard
COLLECTION    = "financegpt"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM    = 384   # this model gives 384-dimensional vectors

def main():
    print("📦 Loading finance data...")
    with open("data/finance_data.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"   Loaded {len(docs)} documents")

    print("🤖 Loading embedding model (downloads once, ~120MB)...")
    model = SentenceTransformer(EMBED_MODEL)

    print("☁️  Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)

    # Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"   ✅ Created collection '{COLLECTION}'")
    else:
        print(f"   ℹ️  Collection '{COLLECTION}' already exists")

    # Build text for embedding: title + content + keywords
    print("🔢 Creating embeddings...")
    texts = []
    for doc in docs:
        text = f"{doc['title']}. {doc['content']} Keywords: {', '.join(doc.get('keywords', []))}"
        texts.append(text)

    vectors = model.encode(texts, show_progress_bar=True)

    # Upload to Qdrant
    print("⬆️  Uploading to Qdrant...")
    points = []
    for i, (doc, vector) in enumerate(zip(docs, vectors)):
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "id":       doc["id"],
                "domain":   doc["domain"],
                "language": doc["language"],
                "title":    doc["title"],
                "content":  doc["content"],
                "keywords": doc.get("keywords", []),
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"✅ Done! {len(points)} documents uploaded to Qdrant.")
    print("   You can now start the backend: uvicorn backend.main:app --reload")

if __name__ == "__main__":
    main()
