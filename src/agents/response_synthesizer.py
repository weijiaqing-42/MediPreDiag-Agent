from openai import AsyncOpenAI
from src.config import settings
from src.agents.state import MediState


async def response_synthesizer_node(state: MediState) -> MediState:
    intent = state.get("intent", "unknown")
    interrupt = state.get("interrupt_flag", False)
    timeout = state.get("timeout_flag", False)

    if interrupt or timeout:
        state["final_response"] = "抱歉，当前服务遇到了一些问题。请重新描述您的情况，我会尽力帮助您。"
        return state

    if intent == "emergency":
        state["final_response"] = _build_emergency_response(state)
    elif intent == "symptom_query":
        state["final_response"] = await _build_symptom_response(state)
    elif intent == "drug_query":
        state["final_response"] = state.get("drug_info", "抱歉，药品咨询暂不可用。")
    elif intent == "location_search":
        state["final_response"] = _build_location_response(state)
    elif intent == "chitchat":
        state["final_response"] = await _build_chitchat_response(state)
    else:
        state["final_response"] = await _build_fallback_response(state)

    return state


def _build_emergency_response(state: MediState) -> str:
    parts = ["🚨 紧急情况提醒 🚨\n"]
    parts.append("根据您的描述，您的情况可能属于紧急情况。")
    parts.append("请立即拨打 120 急救电话或前往最近的急诊科！\n")

    severity = state.get("severity_level", "severe")
    if severity == "severe":
        places = state.get("nearby_places", [])
        if places:
            parts.append("附近的急救中心/医院：")
            for p in places[:3]:
                parts.append(f"  • {p.get('name', '')} - {p.get('address', '')} (距离: {p.get('distance', '')}米)")

    parts.append("\n在等待急救期间：")
    parts.append("  • 保持冷静，尽量保持舒适姿势")
    parts.append("  • 如有已知疾病，告知身边人或急救人员")
    parts.append("  • 不要自行驾车去医院")
    return "\n".join(parts)


async def _build_symptom_response(state: MediState) -> str:
    advice = state.get("medical_advice", "")
    severity = state.get("severity_level", "unknown")
    places = state.get("nearby_places", [])
    symptoms = state.get("extracted_symptoms", [])
    diseases = state.get("possible_diseases", [])

    parts = []
    if advice:
        parts.append(advice)
    else:
        parts.append("根据您的描述，我为您分析如下：\n")
        if symptoms:
            parts.append(f"识别到的症状：{', '.join(symptoms)}")
        if diseases:
            parts.append("\n可能的疾病方向：")
            for d in diseases[:3]:
                parts.append(f"  • {d.get('name', '')} (可能性: {d.get('confidence', 0):.0%})")

    if places:
        parts.append(f"\n🏥 附近的医疗机构（{severity}级别推荐）：")
        for p in places[:5]:
            parts.append(f"  • {p.get('name', '')} - {p.get('address', '')} (距离: {p.get('distance', '')}米)")

    return "\n".join(parts)


def _build_location_response(state: MediState) -> str:
    places = state.get("nearby_places", [])
    if not places:
        return "抱歉，暂时无法查询到附近的医疗机构。请使用地图软件手动搜索。"

    parts = ["以下是您附近的医疗机构："]
    for p in places:
        parts.append(f"  • {p.get('name', '')} - {p.get('address', '')}")
        if p.get("tel"):
            parts.append(f"    电话: {p.get('tel')}")
        parts.append(f"    距离: {p.get('distance', '')}米")
    return "\n".join(parts)


async def _build_chitchat_response(state: MediState) -> str:
    try:
        client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
        response = await client.chat.completions.create(
            model=settings.llm_fast_model,
            messages=[{
                "role": "system",
                "content": "你是医疗预诊助手的问候回复生成器。简洁友好地回复用户，并引导用户描述健康问题。",
            }, {
                "role": "user",
                "content": state.get("user_message", ""),
            }],
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "您好！我是医疗预诊助手。请告诉我您哪里不舒服，我会尽力帮助您。"


async def _build_fallback_response(state: MediState) -> str:
    try:
        client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
        response = await client.chat.completions.create(
            model=settings.llm_fast_model,
            messages=[{
                "role": "system",
                "content": "你是医疗预诊助手。用户的问题不属于症状/药品/地点查询，用友好方式引导用户说明健康问题。",
            }, {
                "role": "user",
                "content": state.get("user_message", ""),
            }],
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "我不太理解您的意思。您可以描述您的症状、询问药品信息或查找附近医院。"