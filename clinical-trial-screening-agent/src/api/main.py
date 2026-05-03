import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.models import ClinicalNote, ScreeningResponse, HealthResponse
from src.agent.screening_agent import screen_patient, TRIAL_CRITERIA
from src.db.models import init_db, get_db
from src.db.crud import (
    save_screening_result,
    get_all_results,
    get_results_by_patient,
    get_results_by_decision,
    get_screening_stats
)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical Trial Screening Agent API",
    description="Multi-step AI agent for automated clinical trial eligibility screening.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    init_db()
    print("✅ API startup complete.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model="gemini-2.5-flash",
        trial_loaded=bool(TRIAL_CRITERIA)
    )


@app.get("/trial")
def get_trial_criteria():
    return {"status": "success", "trial": TRIAL_CRITERIA}


@app.post("/screen", response_model=ScreeningResponse)
def screen_patient_endpoint(note: ClinicalNote, db: Session = Depends(get_db)):
    """Screen a patient and save result to database."""
    try:
        note_dict = {
            "note_id": note.note_id,
            "patient_id": note.patient_id,
            "note_type": note.note_type,
            "text": note.text
        }

        result = screen_patient(note_dict)

        response_data = {
            "note_id": result["note_id"],
            "patient_id": result["patient_id"],
            "trial_id": TRIAL_CRITERIA["trial_id"],
            "trial_name": TRIAL_CRITERIA["trial_name"],
            "decision": result.get("decision", "UNCERTAIN"),
            "confidence": result.get("confidence", "LOW"),
            "justification": result.get("justification", ""),
            "entities_found": {
                "diseases": result["entities"].get("diseases", []),
                "medications": result["entities"].get("medications", []),
                "lab_values": result["entities"].get("lab_values", [])
            },
            "status": "success"
        }

        # Save to database
        save_screening_result(db, response_data)

        return ScreeningResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@app.post("/screen/batch")
def screen_batch(notes: list[ClinicalNote], db: Session = Depends(get_db)):
    """Screen multiple patients and save all results."""
    if len(notes) > 10:
        raise HTTPException(status_code=400, detail="Batch size limited to 10.")

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
            response_data = {
                "note_id": result["note_id"],
                "patient_id": result["patient_id"],
                "trial_id": TRIAL_CRITERIA["trial_id"],
                "trial_name": TRIAL_CRITERIA["trial_name"],
                "decision": result.get("decision", "UNCERTAIN"),
                "confidence": result.get("confidence", "LOW"),
                "justification": result.get("justification", ""),
                "entities_found": {
                    "diseases": result["entities"].get("diseases", []),
                    "medications": result["entities"].get("medications", []),
                    "lab_values": result["entities"].get("lab_values", [])
                },
                "status": "success"
            }
            save_screening_result(db, response_data)
            results.append({
                "patient_id": result["patient_id"],
                "decision": result.get("decision", "UNCERTAIN"),
                "confidence": result.get("confidence", "LOW"),
                "status": "success"
            })
            time.sleep(3)

        except Exception as e:
            results.append({
                "patient_id": note.patient_id,
                "decision": "ERROR",
                "confidence": "NONE",
                "status": f"error: {str(e)}"
            })

    return {"status": "success", "results": results}


@app.get("/results")
def get_results(db: Session = Depends(get_db)):
    """Get all screening results from database."""
    results = get_all_results(db)
    return {
        "status": "success",
        "count": len(results),
        "results": [
            {
                "id": r.id,
                "patient_id": r.patient_id,
                "decision": r.decision,
                "confidence": r.confidence,
                "created_at": str(r.created_at)
            } for r in results
        ]
    }


@app.get("/results/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get summary statistics of all screening decisions."""
    return get_screening_stats(db)


@app.get("/results/patient/{patient_id}")
def get_patient_results(patient_id: str, db: Session = Depends(get_db)):
    """Get all screening results for a specific patient."""
    results = get_results_by_patient(db, patient_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"No results found for {patient_id}")
    return {
        "status": "success",
        "patient_id": patient_id,
        "count": len(results),
        "results": [
            {
                "decision": r.decision,
                "confidence": r.confidence,
                "justification": r.justification,
                "created_at": str(r.created_at)
            } for r in results
        ]
    }