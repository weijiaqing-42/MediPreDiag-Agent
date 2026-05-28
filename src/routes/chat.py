import uuid
import json
import asyncio
from fastapi import APIRouter, HTTPException
from src.schemas import ChatRequest, ChatResponse
from src.agents.state import MediState
from src.agents.graph import graph
from src.handlers.cancel_registry import cancel_registry
from src.db.redis_client import redis_client

router = APIRouter(prefix="/api/v1", tags=["chat"])

CANCEL_RESPONSE_TEXT = "⏹ 操作已被用户取消。请告诉我您的新问题。"


async def _run_graph_with_cancel(initial_state: MediState) -> dict:
    session_id = initial_state.get("session_id", "")

    task = asyncio.current_task()
    if task is not None and session_id:
        cancel_registry.register_task(session_id, task)

    try:
        result = await graph.ainvoke(initial_state)
        return result
    except asyncio.CancelledError:
        return {
            "final_response": CANCEL_RESPONSE_TEXT,
            "intent": "unknown",
            "session_id": session_id,
        }
    finally:
        if session_id:
            cancel_registry.unregister_task(session_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if cancel_registry.is_cancelled(session_id):
        cancel_registry.clear(session_id)

    location = None
    if req.user_location and isinstance(req.user_location, dict):
        lng = req.user_location.get("lng") or req.user_location.get("longitude")
        lat = req.user_location.get("lat") or req.user_location.get("latitude")
        if lng is not None and lat is not None:
            location = (float(lng), float(lat))

    initial_state: MediState = {
        "user_id": req.user_id,
        "session_id": session_id,
        "user_message": req.message,
        "image_url": req.image_url,
        "user_location": location,
        "intent": "unknown",
        "intent_confidence": 0.0,
        "interrupt_flag": False,
        "retry_count": 0,
        "timeout_flag": False,
        "rollback_target": None,
        "extracted_symptoms": [],
        "possible_diseases": [],
        "severity_level": "unknown",
        "rag_context": "",
        "medical_advice": "",
        "nearby_places": [],
        "drug_info": "",
        "final_response": "",
        "short_term_history": [],
        "long_term_summary": "",
    }

    try:
        result = await _run_graph_with_cancel(initial_state)
    except Exception as e:
        cancel_registry.clear(session_id)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    cancel_registry.clear(session_id)

    return ChatResponse(
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        response=result.get("final_response", "抱歉，服务暂时不可用。"),
        possible_diseases=result.get("possible_diseases"),
        severity_level=result.get("severity_level"),
        nearby_places=result.get("nearby_places"),
        drug_info=result.get("drug_info"),
    )


@router.post("/cancel/{session_id}")
async def cancel_session(session_id: str):
    cancelled = cancel_registry.cancel_session(session_id)
    await redis_client.set_cancel_flag(session_id)
    return {
        "status": "cancelled",
        "session_id": session_id,
        "immediate": cancelled,
        "message": "已发送取消信号" if not cancelled else "已立即取消正在执行的任务",
    }


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    from src.memory.short_term import short_term_memory
    history = await short_term_memory.load(session_id)
    return {"session_id": session_id, "history": history}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    from src.memory.short_term import short_term_memory
    await short_term_memory.clear(session_id)
    return {"status": "ok", "message": f"Session {session_id} cleared"}