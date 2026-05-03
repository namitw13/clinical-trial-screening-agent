from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from src.db.models import ScreeningResult
from datetime import datetime


def save_screening_result(db: Session, result: dict) -> ScreeningResult:
    """Save a screening decision to the database."""
    db_result = ScreeningResult(
        note_id=result["note_id"],
        patient_id=result["patient_id"],
        trial_id=result["trial_id"],
        trial_name=result["trial_name"],
        decision=result["decision"],
        confidence=result["confidence"],
        justification=result["justification"],
        entities_found=result["entities_found"],
        created_at=datetime.utcnow()
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    print(f"✅ Saved screening result for {result['patient_id']} to database.")
    return db_result


def get_all_results(db: Session) -> List[ScreeningResult]:
    """Retrieve all screening results, newest first."""
    return db.query(ScreeningResult).order_by(desc(ScreeningResult.created_at)).all()


def get_results_by_patient(db: Session, patient_id: str) -> List[ScreeningResult]:
    """Retrieve all screening results for a specific patient."""
    return db.query(ScreeningResult).filter(
        ScreeningResult.patient_id == patient_id
    ).order_by(desc(ScreeningResult.created_at)).all()


def get_results_by_decision(db: Session, decision: str) -> List[ScreeningResult]:
    """Retrieve all results filtered by decision type."""
    return db.query(ScreeningResult).filter(
        ScreeningResult.decision == decision.upper()
    ).order_by(desc(ScreeningResult.created_at)).all()


def get_screening_stats(db: Session) -> dict:
    """Return summary statistics of all screening decisions."""
    total = db.query(ScreeningResult).count()
    eligible = db.query(ScreeningResult).filter(
        ScreeningResult.decision == "ELIGIBLE"
    ).count()
    ineligible = db.query(ScreeningResult).filter(
        ScreeningResult.decision == "INELIGIBLE"
    ).count()
    uncertain = db.query(ScreeningResult).filter(
        ScreeningResult.decision == "UNCERTAIN"
    ).count()

    return {
        "total_screened": total,
        "eligible": eligible,
        "ineligible": ineligible,
        "uncertain": uncertain,
        "auto_approval_rate": round((eligible / total * 100), 1) if total > 0 else 0
    }