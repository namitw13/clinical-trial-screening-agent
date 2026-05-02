import os
import sys
import json
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add paths so imports work correctly
sys.path.append(str(Path(__file__).parent.parent / "nlp"))
sys.path.append(str(Path(__file__).parent.parent / "agent"))

from src.api.models import ClinicalNote, ScreeningResponse, HealthResponse, ErrorResponse
from src.agent.screening_agent import screen_patient, TRIAL_CRITERIA, llm

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical Trial Screening Agent API",
    description="Multi-step AI agent for automated clinical trial eligibility screening.",
    version="1.0.0"
)

# Allow cross-origin requests (needed if you add a frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and model are running correctly."""
    return HealthResponse(
        status="healthy",
        model="gemini-2.5-flash",
        trial_loaded=bool(TRIAL_CRITERIA)
    )


@app.get("/trial")
def get_trial_criteria():
    """Return the current clinical trial criteria."""
    return {
        "status": "success",
        "trial": TRIAL_CRITERIA
    }


@app.post("/screen", response_model=ScreeningResponse)
def screen_patient_endpoint(note: ClinicalNote):
    """
    Screen a patient clinical note against the trial criteria.
    
    - Runs NER to extract medical entities
    - Uses LangGraph agent to check inclusion/exclusion criteria
    - Returns decision: ELIGIBLE / INELIGIBLE / UNCERTAIN
    """
    try:
        # Convert Pydantic model to dict for the agent
        note_dict = {
            "note_id": note.note_id,
            "patient_id": note.patient_id,
            "note_type": note.note_type,
            "text": note.text
        }

        # Run the screening agent
        result = screen_patient(note_dict)

        return ScreeningResponse(
            note_id=result["note_id"],
            patient_id=result["patient_id"],
            trial_id=TRIAL_CRITERIA["trial_id"],
            trial_name=TRIAL_CRITERIA["trial_name"],
            decision=result.get("decision", "UNCERTAIN"),
            confidence=result.get("confidence", "LOW"),
            justification=result.get("justification", ""),
            entities_found={
                "diseases": result["entities"].get("diseases", []),
                "medications": result["entities"].get("medications", []),
                "lab_values": result["entities"].get("lab_values", [])
            },
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Screening failed: {str(e)}"
        )


@app.post("/screen/batch")
def screen_batch(notes: list[ClinicalNote]):
    """
    Screen multiple patients in one request.
    Returns a list of screening decisions.
    """
    if len(notes) > 10:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 10 notes per request."
        )

    results = []
    for note in notes:
        try:
            note_dict = {
                "note_id": note.note_id,
                "patient_id": note.patient_id,
                "note_type": note.note_type,
                "text": note.text
            }
            result = screen_patient(note_dict)
            results.append({
                "patient_id": result["patient_id"],
                "decision": result.get("decision", "UNCERTAIN"),
                "confidence": result.get("confidence", "LOW"),
                "status": "success"
            })
            time.sleep(3)  # Rate limit buffer between patients

        except Exception as e:
            results.append({
                "patient_id": note.patient_id,
                "decision": "ERROR",
                "confidence": "NONE",
                "status": f"error: {str(e)}"
            })

    return {"status": "success", "results": results}