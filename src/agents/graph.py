from typing import Literal
import logging
from langgraph.graph import StateGraph, END
from src.agents.state import MediState
from src.agents.symptom_agent import symptom_analysis_entry
from src.agents.severity_evaluator import severity_evaluator_node
from src.agents.medical_advisor import rag_retriever_node, medical_advisor_node
from src.agents.location_agent import location_recommender_node
from src.agents.drug_qa_agent import drug_qa_agent_node
from src.agents.response_synthesizer import response_synthesizer_node
from src.agents.memory_nodes import memory_loader_node, memory_updater_node
from src.intent.classifier import classify_intent
from src.handlers.interrupt import interrupt_handler
from src.handlers.cancel_registry import cancel_registry
from src.db.redis_client import redis_client

logger = logging.getLogger(__name__)

CANCEL_RESPONSE = "⏹ 操作已被用户取消。请告诉我您的新问题。"


def _check_external_cancel(state: MediState) -> bool:
    session_id = state.get("session_id", "")
    if session_id and cancel_registry.is_cancelled(session_id):
        state["interrupt_flag"] = True
        state["final_response"] = CANCEL_RESPONSE
        cancel_registry.clear(session_id)
        return True
    return False


async def intent_classifier_node(state: MediState) -> MediState:
    user_message = state.get("user_message", "")

    if not user_message:
        state["intent"] = "chitchat"
        state["intent_confidence"] = 1.0
        return state

    if interrupt_handler.check_cancellation(user_message):
        state["interrupt_flag"] = True
        state["intent"] = "chitchat"
        state["intent_confidence"] = 1.0
        state["final_response"] = "好的，已取消当前操作。请告诉我您的新问题。"
        return state

    try:
        result = await interrupt_handler.execute_with_retry(
            classify_intent, user_message
        )
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.0)
        logger.info(f"Intent classified: {intent} (confidence={confidence:.3f})")
        state["intent"] = intent
        state["intent_confidence"] = confidence
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        state["intent"] = "unknown"
        state["intent_confidence"] = 0.0
        state["timeout_flag"] = True

    return state


async def interrupt_handler_node(state: MediState) -> MediState:
    if state.get("interrupt_flag"):
        if state.get("final_response"):
            return state
        state["final_response"] = CANCEL_RESPONSE
        return state

    rollback_target = state.get("rollback_target", "")
    if interrupt_handler.check_cancellation(state.get("user_message", "")):
        state["final_response"] = "好的，已取消当前操作。请告诉我您的新问题。"
        return state

    if state.get("timeout_flag"):
        state["final_response"] = interrupt_handler.build_fallback_response(
            rollback_target or "intent_classifier"
        )
        return state

    state["final_response"] = interrupt_handler.build_fallback_response(
        rollback_target or "unknown"
    )
    return state


def route_by_intent(state: MediState) -> str:
    if _check_external_cancel(state) or state.get("interrupt_flag"):
        return "interrupt_handler"
    if state.get("timeout_flag"):
        return "interrupt_handler"

    intent = state.get("intent", "unknown")

    if intent == "emergency":
        return "severity_evaluator"

    if intent == "symptom_query":
        return "symptom_analysis"

    if intent == "drug_query":
        return "drug_qa_agent"

    if intent == "location_search":
        return "location_recommender"

    if intent == "chitchat":
        return "response_synthesizer"

    return "response_synthesizer"


def route_after_symptom(state: MediState) -> str:
    if _check_external_cancel(state) or state.get("interrupt_flag") or state.get("timeout_flag"):
        return "interrupt_handler"
    return "rag_retriever"


def route_after_rag(state: MediState) -> str:
    if _check_external_cancel(state) or state.get("interrupt_flag") or state.get("timeout_flag"):
        return "interrupt_handler"
    return "severity_evaluator"


def route_after_severity(state: MediState) -> str:
    if _check_external_cancel(state) or state.get("interrupt_flag") or state.get("timeout_flag"):
        return "interrupt_handler"

    severity = state.get("severity_level", "unknown")
    intent = state.get("intent", "unknown")

    if severity == "severe" or intent == "emergency":
        return "location_recommender"

    return "medical_advisor"


def route_after_medical_advisor(state: MediState) -> str:
    if _check_external_cancel(state) or state.get("interrupt_flag"):
        return "interrupt_handler"
    severity = state.get("severity_level", "unknown")
    if severity in ("moderate", "severe"):
        return "location_recommender"
    return "response_synthesizer"


def route_after_location(state: MediState) -> str:
    if _check_external_cancel(state) or state.get("interrupt_flag"):
        return "interrupt_handler"
    return "response_synthesizer"


def build_graph() -> StateGraph:
    workflow = StateGraph(MediState)

    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("memory_loader", memory_loader_node)
    workflow.add_node("symptom_analysis", symptom_analysis_entry)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("severity_evaluator", severity_evaluator_node)
    workflow.add_node("medical_advisor", medical_advisor_node)
    workflow.add_node("location_recommender", location_recommender_node)
    workflow.add_node("drug_qa_agent", drug_qa_agent_node)
    workflow.add_node("response_synthesizer", response_synthesizer_node)
    workflow.add_node("memory_updater", memory_updater_node)
    workflow.add_node("interrupt_handler", interrupt_handler_node)

    workflow.set_entry_point("memory_loader")

    workflow.add_edge("memory_loader", "intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "symptom_analysis": "symptom_analysis",
            "severity_evaluator": "severity_evaluator",
            "drug_qa_agent": "drug_qa_agent",
            "location_recommender": "location_recommender",
            "response_synthesizer": "response_synthesizer",
            "interrupt_handler": "interrupt_handler",
        },
    )

    workflow.add_conditional_edges(
        "symptom_analysis",
        route_after_symptom,
        {
            "rag_retriever": "rag_retriever",
            "interrupt_handler": "interrupt_handler",
        },
    )

    workflow.add_conditional_edges(
        "rag_retriever",
        route_after_rag,
        {
            "severity_evaluator": "severity_evaluator",
            "interrupt_handler": "interrupt_handler",
        },
    )

    workflow.add_conditional_edges(
        "severity_evaluator",
        route_after_severity,
        {
            "location_recommender": "location_recommender",
            "medical_advisor": "medical_advisor",
            "interrupt_handler": "interrupt_handler",
        },
    )

    workflow.add_conditional_edges(
        "medical_advisor",
        route_after_medical_advisor,
        {
            "location_recommender": "location_recommender",
            "response_synthesizer": "response_synthesizer",
        },
    )

    workflow.add_conditional_edges(
        "location_recommender",
        route_after_location,
        {
            "response_synthesizer": "response_synthesizer",
            "medical_advisor": "medical_advisor",
        },
    )

    workflow.add_edge("drug_qa_agent", "response_synthesizer")
    workflow.add_edge("response_synthesizer", "memory_updater")
    workflow.add_edge("memory_updater", END)

    workflow.add_edge("interrupt_handler", "response_synthesizer")

    return workflow.compile()


graph = build_graph()