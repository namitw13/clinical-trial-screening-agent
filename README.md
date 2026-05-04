# 🏥 Clinical Trial Screening Agent

> Multi-step AI agent for automated clinical trial eligibility screening using LangGraph, Gemini, FastAPI, and PostgreSQL.

🔗 **Live Demo:** [clinical-screening-ui.vercel.app](https://clinical-screening-ui.vercel.app)
📖 **API Docs:** [clinical-trial-screening-agent.onrender.com/docs](https://clinical-trial-screening-agent.onrender.com/docs)

---

## 🎯 Problem Statement

Manually screening patients for clinical trial eligibility takes physicians hours per trial. This system automates the process using a multi-step AI agent that reads unstructured clinical notes, extracts medical entities, reasons over eligibility criteria, and returns a structured decision with justification — in seconds.

---

## 🏗️ Architecture

Clinical Note (unstructured text)
↓
[Step 1] Medical NER — extract diseases, medications, lab values
↓
[Step 2] Inclusion Criteria Check — Gemini reasons over extracted entities
↓
[Step 3] Exclusion Criteria Check — flags disqualifying conditions
↓
[Step 4] Decision Engine — ELIGIBLE / INELIGIBLE / UNCERTAIN + confidence
↓
[Step 5] Justification Report — clinical-grade explanation with evidence
↓
FastAPI REST endpoint → PostgreSQL audit log → React frontend

---

## 🛠️ Tech Stack

| Layer           | Technology                                  |
| --------------- | ------------------------------------------- |
| AI Agent        | LangGraph, Gemini 2.5 Flash                 |
| NER Pipeline    | spaCy, rule-based medical entity extraction |
| Backend         | FastAPI, Python 3.12                        |
| Database        | PostgreSQL, SQLAlchemy                      |
| Frontend        | React, Axios                                |
| Backend Deploy  | Render                                      |
| Frontend Deploy | Vercel                                      |

---

## ✨ Key Features

* **Multi-step reasoning** — 5-node LangGraph pipeline, not a single prompt
* **Medical entity extraction** — diseases, medications, lab values from unstructured text
* **Human-in-the-loop design** — UNCERTAIN decisions escalated for physician review
* **Confidence scoring** — HIGH / MEDIUM / LOW with evidence-based justification
* **Production audit trail** — every decision logged to PostgreSQL with timestamp
* **REST API** — fully documented, Pydantic-validated endpoints
* **Live web app** — anyone can test it at the demo link above

---

## 🚀 Local Setup

```bash
# Clone the repo
git clone https://github.com/namitw13/clinical-trial-screening-agent.git
cd clinical-trial-screening-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NER model
python3 -m spacy download en_core_web_sm

# Set environment variables
cp .env.example .env
# Add your GEMINI_API_KEY and DATABASE_URL

# Run the API
uvicorn src.api.main:app --reload --port 8000
```

---

## 📡 API Endpoints

| Method | Endpoint                | Description              |
| ------ | ----------------------- | ------------------------ |
| GET    | `/health`               | Service health check     |
| GET    | `/trial`                | Current trial criteria   |
| POST   | `/screen`               | Screen a single patient  |
| POST   | `/screen/batch`         | Screen multiple patients |
| GET    | `/results`              | All screening results    |
| GET    | `/results/stats`        | Decision statistics      |
| GET    | `/results/patient/{id}` | Patient history          |

---

## 📊 Sample Output

```json
{
  "patient_id": "P1001",
  "decision": "ELIGIBLE",
  "confidence": "HIGH",
  "justification": "Patient meets all inclusion criteria including Type 2 Diabetes Mellitus diagnosis, CKD stage 3 (eGFR 42), antidiabetic medication (metformin), and age within range. No exclusion criteria triggered.",
  "entities_found": {
    "diseases": ["diabetes mellitus", "hypertension", "chronic kidney disease"],
    "medications": ["metformin", "lisinopril"],
    "lab_values": ["eGFR 42", "HbA1c 9.2%"]
  }
}
```

---

## 🎓 Resume Bullets

* Built a multi-step LangGraph agent for clinical trial eligibility screening; combined spaCy NER with Gemini 2.5 Flash reasoning to achieve structured ELIGIBLE/INELIGIBLE/UNCERTAIN decisions with evidence-based justification; deployed as FastAPI service on Render with PostgreSQL audit logging.
* Designed human-in-the-loop escalation system — agent auto-decides HIGH/MEDIUM confidence cases, flags UNCERTAIN cases for physician review; built React frontend deployed on Vercel with real-time agent step visualization.

---

## 📁 Project Structure

clinical-trial-screening-agent/
├── data/
│   ├── raw/                  # Clinical notes
│   └── trial_criteria.json   # Trial eligibility criteria
├── src/
│   ├── agent/                # LangGraph multi-step agent
│   ├── api/                  # FastAPI routes and Pydantic models
│   ├── nlp/                  # NER pipeline and data loader
│   └── db/                   # PostgreSQL models and CRUD operations
├── requirements.txt
├── render.yaml
├── Procfile
└── README.md

---

## 👤 Author

Built by [@namitw13](https://github.com/namitw13) as a portfolio project demonstrating production-grade AI engineering skills including LLM orchestration, agentic systems, MLOps, and full-stack deployment.
