import uuid
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.agents.state import MediState
from src.agents.graph import graph
from src.handlers.cancel_registry import cancel_registry
from src.db.redis_client import redis_client

router = APIRouter()

CANCEL_RESPONSE_TEXT = "⏹ 操作已被用户取消。请告诉我您的新问题。"


async def _run_graph_ws(initial_state: MediState) -> dict:
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


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            msg_type = data.get("type", "message")

            if msg_type == "cancel":
                session_id = data.get("session_id", "")
                if session_id:
                    cancelled = cancel_registry.cancel_session(session_id)
                    await redis_client.set_cancel_flag(session_id)
                    await websocket.send_json({
                        "type": "cancelled",
                        "session_id": session_id,
                        "immediate": cancelled,
                        "message": "已立即取消正在执行的任务" if cancelled else "已发送取消信号",
                    })
                continue

            session_id = data.get("session_id") or str(uuid.uuid4())
            user_id = data.get("user_id", "anonymous")
            message = data.get("message", "")
            image_url = data.get("image_url")
            location_dict = data.get("user_location")

            if cancel_registry.is_cancelled(session_id):
                cancel_registry.clear(session_id)

            location = None
            if location_dict and isinstance(location_dict, dict):
                lng = location_dict.get("lng") or location_dict.get("longitude")
                lat = location_dict.get("lat") or location_dict.get("latitude")
                if lng is not None and lat is not None:
                    location = (float(lng), float(lat))

            await websocket.send_json({
                "type": "status",
                "session_id": session_id,
                "message": "Processing...",
            })

            initial_state: MediState = {
                "user_id": user_id,
                "session_id": session_id,
                "user_message": message,
                "image_url": image_url,
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
                result = await _run_graph_ws(initial_state)
            except Exception as e:
                cancel_registry.clear(session_id)
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
                continue

            cancel_registry.clear(session_id)

            response_text = result.get("final_response", "")
            is_cancelled = CANCEL_RESPONSE_TEXT in response_text

            await websocket.send_json({
                "type": "response",
                "session_id": session_id,
                "intent": result.get("intent", "unknown"),
                "response": response_text,
                "cancelled": is_cancelled,
                "possible_diseases": result.get("possible_diseases"),
                "severity_level": result.get("severity_level"),
                "nearby_places": result.get("nearby_places"),
                "drug_info": result.get("drug_info"),
            })

    except WebSocketDisconnect:
        pass