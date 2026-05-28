import logging
from openai import AsyncOpenAI
from src.config import settings
from src.agents.state import MediState
from src.rag.retriever import hybrid_retriever

logger = logging.getLogger(__name__)


async def rag_retriever_node(state: MediState) -> MediState:
    symptoms = state.get("extracted_symptoms", [])
    user_message = state.get("user_message", "")
    query = user_message
    if symptoms:
        query = f"{user_message}\n症状：{', '.join(symptoms)}"

    try:
        results = await hybrid_retriever.retrieve(query, top_k=5)
        contexts = [r["content"] for r in results if r.get("content")]
        state["rag_context"] = "\n\n".join(contexts)
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        state["rag_context"] = ""

    return state


async def medical_advisor_node(state: MediState) -> MediState:
    symptoms = state.get("extracted_symptoms", [])
    diseases = state.get("possible_diseases", [])
    severity = state.get("severity_level", "unknown")
    rag_context = state.get("rag_context", "")
    user_message = state.get("user_message", "")
    long_term = state.get("long_term_summary", "")
    history_context = ""
    for entry in state.get("short_term_history", []):
        role = "用户" if entry.get("role") == "user" else "助手"
        history_context += f"{role}: {entry.get('content', '')}\n"

    prompt = f"""你是一位专业医疗顾问。请根据以下信息为用户提供医疗建议。

用户当前描述：{user_message}

提取的症状：{symptoms}
可能的疾病：{diseases}
严重程度：{severity}

历史对话摘要：{history_context[-500:] if history_context else "无"}

{"用户长期健康摘要：" + long_term if long_term else ""}

{"参考医学知识：" + rag_context if rag_context else ""}

请提供：
1. 症状分析与可能的疾病方向
2. 建议的措施（自我观察/药店购药/门诊就医/急诊就医）
3. 需要注意的危险信号
4. 免责声明：本建议仅供参考，不能替代专业医疗诊断

请用清晰、温暖、专业的语气回答。"""

    try:
        client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
        )
        state["medical_advice"] = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Medical advisor failed: {e}")
        state["medical_advice"] = "抱歉，医疗建议生成暂时不可用。"

    return state