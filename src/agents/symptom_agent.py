import logging
from openai import AsyncOpenAI
from src.config import settings
from src.agents.state import MediState

logger = logging.getLogger(__name__)


async def image_analyzer_node(state: MediState) -> MediState:
    image_url = state.get("image_url", "")
    user_message = state.get("user_message", "")

    if not image_url:
        return state

    client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
    prompt = "请仔细观察这张图片，描述其中可能与医学症状相关的内容。如果图中没有明显的医学症状信息，请如实说明。"
    if user_message:
        prompt = f"用户描述：{user_message}\n\n请结合以上用户描述，仔细观察这张图片，提取医学症状相关信息。"

    try:
        response = await client.chat.completions.create(
            model=settings.llm_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1024,
        )
        state["user_message"] = state["user_message"] + "\n[图片分析结果]: " + response.choices[0].message.content
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")

    return state


async def symptom_extractor_node(state: MediState) -> MediState:
    user_message = state.get("user_message", "")
    rag_context = state.get("rag_context", "")

    client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
    prompt = f"""你是一位经验丰富的医生。请根据以下用户描述，提取并结构化症状信息。

用户描述：{user_message}

{f"相关医学知识：{rag_context}" if rag_context else ""}

请以JSON格式返回：
{{
    "symptoms": ["症状1", "症状2"],
    "possible_diseases": [
        {{"name": "疾病名", "confidence": 0.0-1.0, "description": "简要说明"}}
    ],
    "severity_hint": "mild/moderate/severe/unknown"
}}

仅返回JSON。"""

    try:
        import json
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)

        state["extracted_symptoms"] = data.get("symptoms", [])
        state["possible_diseases"] = data.get("possible_diseases", [])
        state["severity_level"] = data.get("severity_hint", "unknown")
    except Exception as e:
        logger.error(f"Symptom extraction failed: {e}")
        state["extracted_symptoms"] = []
        state["possible_diseases"] = []
        state["severity_level"] = "unknown"

    return state


async def symptom_analysis_entry(state: MediState) -> MediState:
    state = await image_analyzer_node(state)
    state = await symptom_extractor_node(state)
    return state