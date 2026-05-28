import logging
from src.agents.state import MediState
from src.tools.amap_poi import amap_tool

logger = logging.getLogger(__name__)


async def location_recommender_node(state: MediState) -> MediState:
    user_location = state.get("user_location")
    severity = state.get("severity_level", "unknown")

    if not user_location:
        state["nearby_places"] = []
        return state

    try:
        results = await amap_tool.search(
            location=(user_location[0], user_location[1]),
            severity=severity,
        )
        state["nearby_places"] = results
    except Exception as e:
        logger.error(f"Location search failed: {e}")
        state["nearby_places"] = []

    return state