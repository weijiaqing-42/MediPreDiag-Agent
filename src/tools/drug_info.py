from typing import List, Optional
from src.db.milvus_client import milvus_client
from src.rag.retriever import hybrid_retriever


class DrugInfoLookupTool:
    async def search(
        self,
        drug_name: str,
        query: str = "",
        top_k: int = 5,
    ) -> str:
        search_text = f"药品名称：{drug_name}\n查询内容：{query}" if query else drug_name
        results = await hybrid_retriever.retrieve(
            query=search_text,
            top_k=top_k,
            source_filter="drug_manual",
        )
        if not results:
            return f"未找到关于「{drug_name}」的药品说明信息。"

        contexts = [r["content"] for r in results if r.get("content")]
        return "\n\n".join(contexts)


drug_tool = DrugInfoLookupTool()