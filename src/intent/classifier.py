import json
import re
import logging
from typing import Any
from openai import AsyncOpenAI
from src.config import settings

logger = logging.getLogger(__name__)

INTENT_LABELS = [
    "symptom_query",
    "drug_query",
    "location_search",
    "emergency",
    "chitchat",
    "unknown",
]

INTENT_SYSTEM_PROMPT = """你是医疗预诊系统的意图分类器。

将用户输入精确分类为以下 6 种类别之一：
- symptom_query: 用户描述身体不适、症状、疼痛、或上传身体图片
- drug_query: 用户询问药品名称、用法、禁忌、副作用、剂量
- location_search: 用户询问附近医院、诊所、药店位置
- emergency: 用户描述胸痛、大出血、呼吸困难、意识丧失等可能危及生命的紧急情况
- chitchat: 问候、闲聊、感谢、与医疗无关的对话
- unknown: 无法明确判断意图

【严格要求】
1. 只输出一行纯 JSON 字符串，不得包含任何其他文字。
2. 不得使用 markdown 代码块标记（如 ``` 或 ```json）。
3. JSON 必须且仅包含两个字段：intent（字符串）和 confidence（浮点数 0.0~1.0）。
4. 禁止输出注释、解释、换行、前导或尾随空格以外的字符。

【输出格式】
{"intent":"symptom_query","confidence":0.95}"""


def _extract_json(raw: str) -> dict | None:
    stripped = raw.strip()

    candidates: list[str] = []

    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if code_block:
        candidates.append(code_block.group(1).strip())

    candidates.append(stripped)

    brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _validate_result(parsed: dict[str, Any]) -> dict[str, Any]:
    intent = parsed.get("intent", "unknown")
    if not isinstance(intent, str) or intent not in INTENT_LABELS:
        logger.warning(f"Invalid intent label '{intent}', falling back to 'unknown'")
        intent = "unknown"

    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    return {"intent": intent, "confidence": confidence}


async def classify_intent(user_message: str) -> dict[str, Any]:
    request_payload = json.dumps(
        {"role": "user", "message": user_message},
        ensure_ascii=False,
    )

    client = AsyncOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )

    response = await client.chat.completions.create(
        model=settings.llm_fast_model,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": request_payload},
        ],
        temperature=0.0,
        max_tokens=64,
    )

    raw = response.choices[0].message.content

    parsed = _extract_json(raw)
    if parsed is None:
        logger.error(
            f"Intent classifier returned unparseable response: {raw[:200]}"
        )
        return {"intent": "unknown", "confidence": 0.0}

    return _validate_result(parsed)