import asyncio
from typing import List, Optional
import hashlib
import jieba
import httpx
from openai import AsyncOpenAI
from src.config import settings
from src.db.milvus_client import milvus_client, DIMENSION

embedding_client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,
)


class BM25Searcher:
    def __init__(self):
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []

    def build_index(self, documents: List[str]):
        self.documents = documents
        self.tokenized_docs = [list(jieba.cut(doc)) for doc in documents]

    def search(self, query: str, top_k: int = 20) -> List[tuple[int, float]]:
        query_tokens = list(jieba.cut(query))
        scores = []
        for idx, doc_tokens in enumerate(self.tokenized_docs):
            score = self._bm25_score(query_tokens, doc_tokens, idx)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str], doc_idx: int) -> float:
        k1, b = 1.5, 0.75
        avgdl = sum(len(d) for d in self.tokenized_docs) / max(len(self.tokenized_docs), 1)
        doc_len = len(doc_tokens)
        score = 0.0
        doc_counts = {}
        for token in doc_tokens:
            doc_counts[token] = doc_counts.get(token, 0) + 1
        N = len(self.tokenized_docs)
        for token in set(query_tokens):
            df = sum(1 for d in self.tokenized_docs if token in d)
            if df == 0:
                continue
            idf = max(0, ((N - df + 0.5) / (df + 0.5)) + 1)
            tf = doc_counts.get(token, 0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avgdl, 1)))
        return score


class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Searcher()

    async def get_embedding(self, text: str) -> List[float]:
        resp = await embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=DIMENSION,
        )
        return resp.data[0].embedding

    def _rrf_fusion(self, vector_results: List[dict], bm25_results: List[tuple], top_k: int = 30) -> List[dict]:
        k = 60
        scores = {}
        content_map = {}

        for rank, hit in enumerate(vector_results):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            content_map[doc_id] = hit

        for rank, (bm25_idx, _) in enumerate(bm25_results):
            doc_id = f"bm25_{bm25_idx}"
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in content_map:
                if bm25_idx < len(self.bm25.documents):
                    content_map[doc_id] = {
                        "id": doc_id,
                        "content": self.bm25.documents[bm25_idx],
                        "source_type": "bm25",
                        "disease_tag": [],
                        "score": 0.0,
                    }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        seen_hashes: set[str] = set()
        for doc_id, rrf_score in ranked[:top_k]:
            item = content_map.get(doc_id)
            if item:
                content_hash = hashlib.md5(item.get("content", "").encode()).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
                item = dict(item)
                item["rrf_score"] = rrf_score
                results.append(item)
        return results

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        try:
            query_embedding = await self.get_embedding(query)
        except Exception:
            return []

        try:
            vector_results = milvus_client.search(
                query_embedding=query_embedding,
                top_k=top_k,
                expr=f'source_type == "{source_filter}"' if source_filter else None,
            )
        except Exception:
            vector_results = []

        all_docs_for_bm25 = []
        for r in vector_results:
            all_docs_for_bm25.append(r.get("content", ""))
        self.bm25.build_index(all_docs_for_bm25)

        try:
            bm25_results = self.bm25.search(query, top_k=top_k)
        except Exception:
            bm25_results = []

        fused = self._rrf_fusion(vector_results, bm25_results, top_k=30)

        return await reranker.rerank(query, fused, top_k=min(top_k, 5))


class Reranker:
    """DashScope qwen3-rerank API. Sortes documents by semantic relevance."""

    async def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        if not candidates:
            return []

        documents = [c["content"] for c in candidates]

        payload = {
            "model": settings.rerank_model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": False,
                "top_n": min(top_k, len(documents)),
            },
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    settings.rerank_api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {settings.dashscope_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return candidates[:top_k]

        results = data.get("output", {}).get("results", [])
        if not results:
            return candidates[:top_k]

        ranked = []
        for item in results:
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if 0 <= idx < len(candidates):
                entry = dict(candidates[idx])
                entry["rerank_score"] = score
                ranked.append(entry)
        return ranked if ranked else candidates[:top_k]


reranker = Reranker()
hybrid_retriever = HybridRetriever()