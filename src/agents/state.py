from typing import TypedDict, Literal, Optional, List, Annotated
import operator


class MediState(TypedDict):
    user_id: str
    session_id: str
    user_message: str
    image_url: Optional[str]
    user_location: Optional[tuple]

    intent: str
    intent_confidence: float
    interrupt_flag: bool
    retry_count: int
    timeout_flag: bool
    rollback_target: Optional[str]

    extracted_symptoms: List[str]
    possible_diseases: List[dict]
    severity_level: str
    rag_context: str

    medical_advice: str
    nearby_places: List[dict]
    drug_info: str
    final_response: str

    short_term_history: Annotated[List[dict], operator.add]
    long_term_summary: str