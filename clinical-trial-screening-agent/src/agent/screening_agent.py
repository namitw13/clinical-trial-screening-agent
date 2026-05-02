import os
import json
import sys
import time
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv


# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Add nlp directory to path
sys.path.append(str(Path(__file__).parent.parent / "nlp"))
from ner_pipeline import extract_entities
from data_loader import load_and_clean

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1  # Low temperature for consistent medical reasoning
)

# Load trial criteria
CRITERIA_PATH = Path(__file__).parent.parent.parent / "data" / "trial_criteria.json"
with open(CRITERIA_PATH) as f:
    TRIAL_CRITERIA = json.load(f)


# ── Agent State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    note_id: str
    patient_id: str
    clinical_text: str
    entities: Dict[str, Any]
    trial_criteria: Dict[str, Any]
    inclusion_check: str
    exclusion_check: str
    decision: str
    confidence: str
    justification: str


# ── Node 1: Extract Entities ─────────────────────────────────────────────────
def extract_node(state: AgentState) -> AgentState:
    """Extract medical entities from clinical note."""
    print(f"\n🔍 [Step 1] Extracting entities for {state['note_id']}...")
    entities = extract_entities(state["clinical_text"])
    state["entities"] = entities
    print(f"   Found: {len(entities['diseases'])} diseases, "
          f"{len(entities['medications'])} medications, "
          f"{len(entities['lab_values'])} lab values")
    return state


# ── Node 2: Check Inclusion Criteria ─────────────────────────────────────────
def inclusion_check_node(state: AgentState) -> AgentState:
    """Use Gemini to check inclusion criteria against extracted entities."""
    print(f"✅ [Step 2] Checking inclusion criteria...")
    time.sleep(3)
    prompt = f"""You are a clinical trial screening assistant.

PATIENT INFORMATION:
- Diseases/Conditions: {state['entities']['diseases']}
- Medications: {state['entities']['medications']}
- Lab Values: {state['entities']['lab_values']}
- Full Clinical Note: {state['clinical_text'][:500]}

INCLUSION CRITERIA (patient must meet ALL):
{json.dumps(state['trial_criteria']['inclusion_criteria'], indent=2)}

For each inclusion criterion, state:
- CRITERION: [criterion text]
- MET: Yes / No / Uncertain
- EVIDENCE: [what in the patient data supports this]

Be concise and precise."""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["inclusion_check"] = response.content
    print(f"   Inclusion check complete.")
    return state


# ── Node 3: Check Exclusion Criteria ─────────────────────────────────────────
def exclusion_check_node(state: AgentState) -> AgentState:
    """Use Gemini to check exclusion criteria against extracted entities."""
    print(f"🚫 [Step 3] Checking exclusion criteria...")
    time.sleep(3)
    prompt = f"""You are a clinical trial screening assistant.

PATIENT INFORMATION:
- Diseases/Conditions: {state['entities']['diseases']}
- Medications: {state['entities']['medications']}
- Lab Values: {state['entities']['lab_values']}
- Full Clinical Note: {state['clinical_text'][:500]}

EXCLUSION CRITERIA (patient must meet NONE of these):
{json.dumps(state['trial_criteria']['exclusion_criteria'], indent=2)}

For each exclusion criterion, state:
- CRITERION: [criterion text]
- TRIGGERED: Yes / No / Uncertain
- EVIDENCE: [what in the patient data supports this]

Be concise and precise."""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["exclusion_check"] = response.content
    print(f"   Exclusion check complete.")
    return state


# ── Node 4: Make Final Decision ───────────────────────────────────────────────
def decision_node(state: AgentState) -> AgentState:
    """Synthesize inclusion and exclusion checks into a final decision."""
    print(f"⚖️  [Step 4] Making final decision...")
    time.sleep(3)
    prompt = f"""You are a senior clinical trial coordinator making a final eligibility decision.

INCLUSION CRITERIA ANALYSIS:
{state['inclusion_check']}

EXCLUSION CRITERIA ANALYSIS:
{state['exclusion_check']}

Based on the above analysis, provide:
1. DECISION: ELIGIBLE / INELIGIBLE / UNCERTAIN
2. CONFIDENCE: HIGH / MEDIUM / LOW
3. PRIMARY REASON: One sentence explaining the key factor in your decision
4. RECOMMENDATION: What should happen next (enroll / reject / physician review)

Format your response exactly as:
DECISION: [value]
CONFIDENCE: [value]
PRIMARY REASON: [one sentence]
RECOMMENDATION: [one sentence]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Parse the structured response
    lines = response.content.strip().split('\n')
    for line in lines:
        if line.startswith("DECISION:"):
            state["decision"] = line.replace("DECISION:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            state["confidence"] = line.replace("CONFIDENCE:", "").strip()
        elif line.startswith("PRIMARY REASON:"):
            state["justification"] = line.replace("PRIMARY REASON:", "").strip()

    print(f"   Decision: {state.get('decision', 'Unknown')} "
          f"(Confidence: {state.get('confidence', 'Unknown')})")
    return state


# ── Node 5: Generate Justification Report ────────────────────────────────────
def justification_node(state: AgentState) -> AgentState:
    """Generate a human-readable justification report."""
    print(f"📋 [Step 5] Generating justification report...")

    prompt = f"""You are a clinical trial screening assistant writing a brief report.

PATIENT: {state['patient_id']}
TRIAL: {state['trial_criteria']['trial_name']}
DECISION: {state.get('decision', 'UNCERTAIN')}
CONFIDENCE: {state.get('confidence', 'LOW')}

INCLUSION ANALYSIS:
{state['inclusion_check']}

EXCLUSION ANALYSIS:
{state['exclusion_check']}

Write a concise 3-4 sentence screening report that:
1. States the decision clearly
2. Lists the key supporting evidence
3. Flags any uncertainties
4. Gives a clear next step

Write in professional clinical language."""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["justification"] = response.content
    print(f"   Report generated.")
    return state


# ── Build the LangGraph ───────────────────────────────────────────────────────
def build_agent():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("extract", extract_node)
    workflow.add_node("inclusion_check", inclusion_check_node)
    workflow.add_node("exclusion_check", exclusion_check_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("justification", justification_node)

    # Define edges (sequential flow)
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "inclusion_check")
    workflow.add_edge("inclusion_check", "exclusion_check")
    workflow.add_edge("exclusion_check", "decision")
    workflow.add_edge("decision", "justification")
    workflow.add_edge("justification", END)

    return workflow.compile()


# ── Run the Agent ─────────────────────────────────────────────────────────────
def screen_patient(note: Dict) -> Dict:
    """Screen a single patient note against the trial criteria."""
    agent = build_agent()

    initial_state = AgentState(
        note_id=note["note_id"],
        patient_id=note["patient_id"],
        clinical_text=note["text"],
        entities={},
        trial_criteria=TRIAL_CRITERIA,
        inclusion_check="",
        exclusion_check="",
        decision="",
        confidence="",
        justification=""
    )

    result = agent.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Load notes and screen the first two patients
    df = load_and_clean()
    notes = df.to_dict(orient="records")

    print("=" * 60)
    print(f"TRIAL: {TRIAL_CRITERIA['trial_name']}")
    print("=" * 60)

    for note in notes[:2]:  # Test with first 2 notes
        print(f"\n{'='*60}")
        print(f"SCREENING PATIENT: {note['patient_id']}")
        print(f"{'='*60}")

        result = screen_patient(note)

        print(f"\n{'─'*60}")
        print("FINAL SCREENING REPORT")
        print(f"{'─'*60}")
        print(f"Patient    : {result['patient_id']}")
        print(f"Decision   : {result['decision']}")
        print(f"Confidence : {result['confidence']}")
        print(f"\nJustification:\n{result['justification']}")
        print(f"{'='*60}\n")