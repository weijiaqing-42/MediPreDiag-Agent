import json
from datetime import datetime
from openai import AsyncOpenAI
from src.config import settings
from src.db.milvus_client import milvus_client
from src.rag.retriever import HybridRetriever

llm_client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,
)

SUMMARY_COLLECTION = "user_long_term_memory"
SUMMARY_DIM = 1024


class LongTermMemory:
    async def generate_summary(self, session_history: list[dict]) -> str:
        if not session_history:
            return ""

        history_text = "\n".join([
            f"{'用户' if h.get('role') == 'user' else '助手'}: {h.get('content', '')}"
            for h in session_history
        ])

        prompt = f"""请对以下医疗预诊对话生成一份简要摘要，包含：
1. 用户描述的主要症状
2. 系统的诊断推断
3. 用户可能的过敏史或慢性病史（如有提及）
4. 用户偏好（如对某类药物、某类医院的偏好）

对话记录：
{history_text}

请用中文简要总结，200字以内。"""

        response = await llm_client.chat.completions.create(
            model=settings.llm_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    async def save_summary(self, user_id: str, session_id: str, summary: str):
        try:
            from src.rag.retriever import HybridRetriever
            retriever = HybridRetriever()
            embedding = await retriever.get_embedding(summary)
            milvus_client.connect()

            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

            if not utility.has_collection(SUMMARY_COLLECTION):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=SUMMARY_DIM),
                ]
                schema = CollectionSchema(fields, description="User Long-Term Memory")
                col = Collection(SUMMARY_COLLECTION, schema)
                col.create_index("embedding", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})
                col.load()
            else:
                col = Collection(SUMMARY_COLLECTION)

            col.insert([
                [user_id],
                [session_id],
                [summary],
                [datetime.now().isoformat()],
                [embedding],
            ])
        except Exception:
            pass

    async def retrieve_summaries(self, user_id: str, top_k: int = 3) -> str:
        try:
            milvus_client.connect()
            from pymilvus import Collection, utility

            if not utility.has_collection(SUMMARY_COLLECTION):
                return ""

            col = Collection(SUMMARY_COLLECTION)
            results = col.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["summary", "created_at"],
                limit=top_k,
            )
            summaries = [r.get("summary", "") for r in results]
            return "\n".join(summaries)
        except Exception:
            return ""


long_term_memory = LongTermMemory()