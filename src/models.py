"""
src/models.py
─────────────
All typed data models for the Clinical Triage OpenEnv environment.

Pydantic models define exactly what an AI agent SEES (Observation),
what it DOES (Action), and what SCORE it receives (Reward).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# ENUMS  (controlled vocabularies)
# ──────────────────────────────────────────────

class UrgencyLevel(str, Enum):
    """How quickly the patient needs to be seen."""
    LOW      = "low"       # Non-urgent, routine appointment fine
    MEDIUM   = "medium"    # Needs attention within hours
    HIGH     = "high"      # Needs attention within the hour
    CRITICAL = "critical"  # Life-threatening, immediate intervention


class SpecialistType(str, Enum):
    """Medical specialist categories available for referral."""
    GENERAL            = "general_practitioner"
    CARDIOLOGIST       = "cardiologist"
    NEUROLOGIST        = "neurologist"
    ORTHOPEDIC         = "orthopedic"
    PSYCHIATRIST       = "psychiatrist"
    PULMONOLOGIST      = "pulmonologist"
    GASTROENTEROLOGIST = "gastroenterologist"
    ENDOCRINOLOGIST    = "endocrinologist"
    EMERGENCY          = "emergency_medicine"


# ──────────────────────────────────────────────
# PATIENT INTAKE FORM  (the raw clinical data)
# ──────────────────────────────────────────────

class PatientVitals(BaseModel):
    """Vital signs measured during intake."""
    blood_pressure_systolic:  Optional[int]   = Field(None, description="Systolic BP in mmHg (top number)")
    blood_pressure_diastolic: Optional[int]   = Field(None, description="Diastolic BP in mmHg (bottom number)")
    heart_rate:               Optional[int]   = Field(None, description="Beats per minute")
    temperature_celsius:      Optional[float] = Field(None, description="Body temperature in Celsius")
    oxygen_saturation:        Optional[float] = Field(None, description="SpO2 percentage (normal ≥ 95%)")
    respiratory_rate:         Optional[int]   = Field(None, description="Breaths per minute (normal 12-20)")
    pain_scale:               Optional[int]   = Field(None, ge=0, le=10, description="Self-reported pain 0-10")


class PatientIntakeForm(BaseModel):
    """Complete patient intake form presented to the triage agent."""
    patient_id:              str
    age:                     int
    gender:                  str
    chief_complaint:         str                    = Field(..., description="Primary reason for visit in patient's words")
    symptoms:                List[str]              = Field(..., description="Reported symptoms")
    symptom_duration_hours:  Optional[int]          = Field(None, description="How long symptoms have been present")
    vitals:                  Optional[PatientVitals] = None
    current_medications:     Optional[List[str]]    = Field(None, description="Medications patient is currently taking")
    proposed_medications:    Optional[List[str]]    = Field(None, description="Medications being considered for this visit (Task 3)")
    allergies:               Optional[List[str]]    = Field(None, description="Known allergies")
    medical_history:         Optional[List[str]]    = Field(None, description="Significant past medical conditions")


# ──────────────────────────────────────────────
# OBSERVATION  (what the agent receives each step)
# ──────────────────────────────────────────────

class Observation(BaseModel):
    """
    Everything the AI agent can see at any given step.
    The agent reads this and then decides its Action.
    """
    task_id:          str
    task_name:        str
    task_description: str                = Field(..., description="Instructions telling the agent what it must do")
    patient_form:     PatientIntakeForm  = Field(..., description="The patient intake data to triage")
    step_number:      int                = Field(..., description="Current step within episode (1-indexed)")
    max_steps:        int                = Field(..., description="Total steps in this episode")
    previous_reward:  Optional[float]   = Field(None, description="Score received on the previous step")


# ──────────────────────────────────────────────
# ACTION  (what the agent decides to do)
# ──────────────────────────────────────────────

class TriageDecision(BaseModel):
    """
    The agent's clinical triage decision for one patient.
    All tasks use the same action schema — later tasks just
    require more fields to be correct.
    """
    urgency_level:      UrgencyLevel              = Field(..., description="How urgently this patient needs care")
    specialist_referral: SpecialistType           = Field(..., description="Which specialist to route the patient to")
    medication_flags:   Optional[List[str]]       = Field(default_factory=list, description="Medications the agent considers dangerous for this patient")
    contraindications:  Optional[List[str]]       = Field(default_factory=list, description="Described contraindications or drug interactions found")
    clinical_notes:     Optional[str]             = Field(None, description="Agent's brief clinical reasoning (optional but rewarded in Task 3)")


class Action(BaseModel):
    """Wrapper for the agent's action — matches OpenEnv Action spec."""
    action_type: str          = Field(default="triage_decision", description="Always 'triage_decision' for this env")
    triage:      TriageDecision


# ──────────────────────────────────────────────
# REWARD  (detailed scoring breakdown)
# ──────────────────────────────────────────────

class Reward(BaseModel):
    """
    Detailed breakdown of how the agent scored on a single step.
    The top-level `score` is the number that matters (0.0–1.0).
    Sub-scores help the agent (and developer) understand what to improve.
    """
    score:                  float = Field(..., ge=0.0, le=1.0, description="Overall score for this step")
    urgency_score:          float = Field(..., ge=0.0, le=1.0, description="Score for urgency level accuracy")
    specialist_score:       float = Field(..., ge=0.0, le=1.0, description="Score for specialist routing accuracy")
    medication_score:       float = Field(..., ge=0.0, le=1.0, description="Score for medication flag detection")
    contraindication_score: float = Field(..., ge=0.0, le=1.0, description="Score for contraindication identification")
    feedback:               str   = Field(..., description="Human-readable explanation of the score")


# ──────────────────────────────────────────────
# STEP RESULT  (what step() returns)
# ──────────────────────────────────────────────

class StepResult(BaseModel):
    """The full return value from env.step(action)."""
    observation: Observation          = Field(..., description="Next observation (next patient, or final state if done)")
    reward:      float                = Field(..., description="Score for the action just taken (0.0–1.0)")
    done:        bool                 = Field(..., description="True when all patients in episode have been triaged")
    info:        Dict[str, Any]       = Field(default_factory=dict, description="Extra diagnostic info (scores breakdown, etc.)")


# ──────────────────────────────────────────────
# ENVIRONMENT STATE  (what state() returns)
# ──────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Snapshot of the environment — returned by env.state()."""
    session_id:      str
    task_id:         str
    step_number:     int
    max_steps:       int
    done:            bool
    current_score:   float                 = Field(..., description="Running average score across completed steps")
    step_scores:     List[float]           = Field(default_factory=list, description="Score for each completed step")
    episode_history: List[Dict[str, Any]]  = Field(default_factory=list, description="Log of all actions and scores this episode")


# ──────────────────────────────────────────────
# API REQUEST BODIES
# ──────────────────────────────────────────────

task_id:    Optional[str] = Field("task_1", description="Which task to start: 'task_1', 'task_2', or 'task_3'")
session_id: Optional[str] = Field(None, description="Provide to resume a session, or omit for a new one")


class StepRequest(BaseModel):
    session_id: str    = Field(..., description="Session ID returned by /reset")
    action:     Action = Field(..., description="The agent's triage decision")