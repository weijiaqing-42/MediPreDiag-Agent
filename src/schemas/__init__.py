from __future__ import annotations

from typing import Optional, List, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    message: str
    image_url: Optional[str] = None
    user_location: Optional[dict] = None


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    response: str
    possible_diseases: Optional[List[dict]] = None
    severity_level: Optional[str] = None
    nearby_places: Optional[List[dict]] = None
    drug_info: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    status: str
    intent_path: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None


class DiagnosisRecord(BaseModel):
    id: Optional[int] = None
    session_id: str
    user_id: str
    extracted_symptoms: List[str] = Field(default_factory=list)
    possible_diseases: List[dict] = Field(default_factory=list)
    severity_level: Optional[str] = None
    medical_advice: Optional[str] = None
    created_at: Optional[datetime] = None