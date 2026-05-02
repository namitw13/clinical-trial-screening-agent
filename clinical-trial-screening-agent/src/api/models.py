from pydantic import BaseModel
from typing import Optional


class ClinicalNote(BaseModel):
    """Request model — what the API accepts."""
    note_id: str
    patient_id: str
    note_type: Optional[str] = "Discharge Summary"
    text: str


class ScreeningResponse(BaseModel):
    """Response model — what the API returns."""
    note_id: str
    patient_id: str
    trial_id: str
    trial_name: str
    decision: str          # ELIGIBLE / INELIGIBLE / UNCERTAIN
    confidence: str        # HIGH / MEDIUM / LOW
    justification: str
    entities_found: dict
    status: str            # success / error


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    trial_loaded: bool


class ErrorResponse(BaseModel):
    """Error response."""
    status: str
    error: str
    note_id: Optional[str] = None