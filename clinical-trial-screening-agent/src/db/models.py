from sqlalchemy import (
    Column, String, Text, DateTime, Integer, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent.parent / ".env")

Base = declarative_base()


class ScreeningResult(Base):
    """Stores every screening decision made by the agent."""
    __tablename__ = "screening_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    note_id = Column(String(50), nullable=False)
    patient_id = Column(String(50), nullable=False)
    trial_id = Column(String(50), nullable=False)
    trial_name = Column(String(200), nullable=False)
    decision = Column(String(20), nullable=False)       # ELIGIBLE/INELIGIBLE/UNCERTAIN
    confidence = Column(String(10), nullable=False)     # HIGH/MEDIUM/LOW
    justification = Column(Text, nullable=True)
    entities_found = Column(JSON, nullable=True)        # stores dict as JSON
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Database Connection ───────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://screening_user:screening_pass@localhost:5432/clinical_screening"
)

# Render uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created.")


def get_db():
    """Dependency for FastAPI — yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()