import json
import logging
from openai import AsyncOpenAI
from src.config import settings
from src.agents.state import MediState
from src.tools.drug_info import drug_tool
from src.rag.retriever import hybrid_retriever

logger = logging.getLogger(__name__)


async def drug_qa_agent_node(state: MediState) -> MediState:
    user_message = state.get("user_message", "")

    client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)

    extract_prompt = f"""请从用户查询中提取药品名称和具体问题类型。

用户查询：{user_message}

返回JSON：
{{
    "drug_name": "药品名称，未知则填unknown",
    "query_type": "usage/contraindication/side_effect/dosage/interaction/general",
    "specific_question": "用户的具体问题"
}}"""

    drug_name = "unknown"
    query_type = "general"
    specific_question = user_message

    try:
        response = await client.chat.completions.create(
            model=settings.llm_fast_model,
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        drug_name = data.get("drug_name", "unknown")
        query_type = data.get("query_type", "general")
        specific_question = data.get("specific_question", user_message)
    except Exception as e:
        logger.error(f"Drug name extraction failed: {e}")

    try:
        drug_context = await drug_tool.search(drug_name, specific_question)
    except Exception as e:
        logger.error(f"Drug info search failed: {e}")
        drug_context = ""

    try:
        rag_results = await hybrid_retriever.retrieve(
            query=f"{drug_name} {specific_question}",
            top_k=3,
            source_filter="drug_manual",
        )
        rag_context = "\n\n".join([r["content"] for r in rag_results if r.get("content")])
    except Exception:
        rag_context = ""

    combined_context = f"{drug_context}\n\n{rag_context}" if rag_context else drug_context

    prompt = f"""你是一位专业药师。请回答以下药品相关问题。

用户问题：{user_message}
药品名称：{drug_name}
问题类型：{query_type}

{"参考药品信息：" + combined_context if combined_context else "未找到相关药品信息，请基于你的专业知识回答。"}

请提供：
1. 针对用户问题的直接回答
2. 必要的注意事项和警告
3. 免责声明：本回答仅供参考，请遵医嘱用药

请用专业但易懂的语言回答。"""

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        state["drug_info"] = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Drug QA failed: {e}")
        state["drug_info"] = "抱歉，药品咨询服务暂时不可用。"

    return state