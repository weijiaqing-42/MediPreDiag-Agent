import logging
from src.agents.state import MediState
from src.memory.short_term import short_term_memory
from src.memory.long_term import long_term_memory
from src.db.mysql import SessionLocal

logger = logging.getLogger(__name__)


async def memory_loader_node(state: MediState) -> MediState:
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")

    if session_id:
        try:
            history = await short_term_memory.load(session_id)
            state["short_term_history"] = history
        except Exception as e:
            logger.error(f"Short-term memory load failed: {e}")

    if user_id:
        try:
            summary = await long_term_memory.retrieve_summaries(user_id)
            state["long_term_summary"] = summary
        except Exception as e:
            logger.error(f"Long-term memory load failed: {e}")

    return state


async def memory_updater_node(state: MediState) -> MediState:
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    user_message = state.get("user_message", "")
    final_response = state.get("final_response", "")

    if session_id:
        try:
            await short_term_memory.save(session_id, {
                "role": "user",
                "content": user_message,
            })
            await short_term_memory.save(session_id, {
                "role": "assistant",
                "content": final_response,
            })
        except Exception as e:
            logger.error(f"Short-term memory save failed: {e}")

    if session_id and user_id:
        try:
            history = await short_term_memory.load(session_id)
            if len(history) >= 6:
                summary = await long_term_memory.generate_summary(history)
                await long_term_memory.save_summary(user_id, session_id, summary)
        except Exception as e:
            logger.error(f"Long-term memory save failed: {e}")

    return state


def _save_diagnosis_sync(session_id: str, user_id: str, state: MediState):
    try:
        import json
        from sqlalchemy import text

        db = SessionLocal()
        db.execute(text("""
            INSERT INTO diagnoses (session_id, user_id, extracted_symptoms, possible_diseases, severity_level, medical_advice)
            VALUES (:session_id, :user_id, :symptoms, :diseases, :severity, :advice)
        """), {
            "session_id": session_id,
            "user_id": user_id,
            "symptoms": json.dumps(state.get("extracted_symptoms", []), ensure_ascii=False),
            "diseases": json.dumps(state.get("possible_diseases", []), ensure_ascii=False),
            "severity": state.get("severity_level", "unknown"),
            "advice": state.get("medical_advice", ""),
        })
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"MySQL diagnosis save failed: {e}")