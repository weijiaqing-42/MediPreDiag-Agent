import logging
from openai import AsyncOpenAI
from src.config import settings
from src.agents.state import MediState

logger = logging.getLogger(__name__)


async def severity_evaluator_node(state: MediState) -> MediState:
    symptoms = state.get("extracted_symptoms", [])
    diseases = state.get("possible_diseases", [])
    user_message = state.get("user_message", "")

    emergency_keywords = [
        "胸痛", "大出血", "呼吸困难", "意识丧失", "抽搐", "窒息",
        "剧烈头痛", "突然视力丧失", "咳血", "吐血", "便血",
        "严重外伤", "骨折", "烧伤", "中毒", "过敏性休克",
        "心脏骤停", "呼吸停止", "严重过敏", "高热惊厥",
    ]

    is_emergency = any(kw in user_message for kw in emergency_keywords)
    if is_emergency:
        state["severity_level"] = "severe"
        return state

    client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)

    prompt = f"""请评估以下症状的严重程度。

用户描述：{user_message}
提取的症状：{symptoms}
可能的疾病：{diseases}

请返回JSON：
{{
    "severity": "mild/moderate/severe/unknown",
    "reason": "判断理由",
    "urgency_advice": "紧急程度建议"
}}"""

    try:
        import json
        response = await client.chat.completions.create(
            model=settings.llm_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        state["severity_level"] = data.get("severity", "unknown")
    except Exception as e:
        logger.error(f"Severity evaluation failed: {e}")

    return state