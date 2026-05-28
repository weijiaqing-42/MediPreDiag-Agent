"""
Medical Knowledge Base Ingestion Script
Reads data/medical_knowledge.json, generates embeddings via DashScope,
and inserts into Milvus for RAG retrieval.
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import AsyncOpenAI
from src.config import settings
from src.db.milvus_client import milvus_client

BATCH_SIZE = 10
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "medical_knowledge.json")


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    client = AsyncOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1} "
              f"({len(batch)} texts)...")
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=1024,
        )
        for item in resp.data:
            embeddings.append(item.embedding)
        await asyncio.sleep(0.5)
    return embeddings


async def main():
    print("=" * 60)
    print("Medical Knowledge Base Ingestion")
    print("=" * 60)

    if not os.path.isfile(DATA_FILE):
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"\nLoaded {len(documents)} documents from {DATA_FILE}")
    print(f"Source type distribution:")
    types = {}
    for doc in documents:
        t = doc.get("source_type", "unknown")
        types[t] = types.get(t, 0) + 1
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")

    print("\n[1] Connecting to Milvus...")
    try:
        milvus_client.connect()
        milvus_client.init_collection()
        print("  Milvus connected, collection ready.")
    except Exception as e:
        print(f"  [ERROR] Milvus connection failed: {e}")
        print("  Make sure 'docker compose up -d' has been run first.")
        sys.exit(1)

    print("\n[2] Generating embeddings...")
    texts = [doc["content"] for doc in documents]
    try:
        embeddings = await get_embeddings(texts)
        print(f"  Generated {len(embeddings)} embeddings (dim={len(embeddings[0])}).")
    except Exception as e:
        print(f"  [ERROR] Embedding generation failed: {e}")
        sys.exit(1)

    print("\n[3] Inserting into Milvus...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_embs = embeddings[i : i + BATCH_SIZE]
        payloads = [
            {
                "content": d["content"],
                "source_type": d.get("source_type", ""),
                "disease_tag": d.get("disease_tag", []),
            }
            for d in batch_docs
        ]
        try:
            milvus_client.insert(payloads, batch_embs)
            print(f"  Inserted batch {i // BATCH_SIZE + 1}/"
                  f"{(len(documents) - 1) // BATCH_SIZE + 1} "
                  f"({len(batch_docs)} docs)")
        except Exception as e:
            print(f"  [ERROR] Insert failed at batch {i // BATCH_SIZE + 1}: {e}")
            sys.exit(1)

    print(f"\n[DONE] Successfully ingested {len(documents)} documents.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())