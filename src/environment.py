"""
src/environment.py
──────────────────
ClinicalTriageEnv — the main OpenEnv environment class.

Implements the full OpenEnv interface:
  reset(task_id)   → Observation       (starts a fresh episode)
  step(action)     → StepResult        (agent acts; env responds)
  state()          → EnvironmentState  (snapshot of current state)

Each episode = one task (3 patient cases = 3 steps).
Reward is given at every step (partial progress signal).
Episode ends (done=True) after the last patient is triaged.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, Optional

from .data    import TASK_REGISTRY
from .graders import grade_action
from .models  import (
    Action,
    EnvironmentState,
    Observation,
    PatientIntakeForm,
    PatientVitals,
    StepResult,
)


class ClinicalTriageEnv:
    """
    Clinical Patient Triage Environment.

    Each call to reset() starts a fresh triage episode for a chosen task.
    The agent is shown one patient at a time (via Observation) and must
    submit a TriageDecision (via Action). The environment grades the
    decision, gives a step reward, then presents the next patient.

    Episode structure (all 3 tasks have 3 patients each):
        reset()  →  Observation(patient 1)
        step(a1) →  StepResult(reward_1, Observation(patient 2), done=False)
        step(a2) →  StepResult(reward_2, Observation(patient 3), done=False)
        step(a3) →  StepResult(reward_3, final_observation,     done=True)
    """

    def __init__(self) -> None:
        self._session_id:  str  = ""
        self._task_id:     str  = ""
        self._cases:       list = []
        self._case_index:  int  = 0
        self._step_scores: list = []
        self._history:     list = []
        self._done:        bool = True
        self._task_meta:   dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str, session_id: Optional[str] = None) -> Observation:
        """
        Start a fresh episode for the given task.

        Args:
            task_id:    One of "task_1", "task_2", "task_3"
            session_id: Optional — provide to reuse an ID, omit for auto-generated UUID

        Returns:
            The first Observation (first patient's intake form + task instructions).

        Raises:
            ValueError: If task_id is not recognised.
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {sorted(TASK_REGISTRY.keys())}"
            )

        self._session_id  = session_id or str(uuid.uuid4())
        self._task_id     = task_id
        self._task_meta   = TASK_REGISTRY[task_id]
        self._cases       = deepcopy(self._task_meta["cases"])
        self._case_index  = 0
        self._step_scores = []
        self._history     = []
        self._done        = False

        return self._build_observation(self._cases[0])

    # ──────────────────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """
        Process the agent's triage decision for the current patient.

        Reward function:
          • Immediate feedback: the step reward = graded score for this case (0.0–1.0)
          • Partial progress: reward is non-zero even if only urgency is correct
          • Episode-end bonus: if average score > 0.90, info["bonus"] = 0.05
          • Penalty signal: if agent leaves clinical_notes blank on Task 3 and
            scores poorly on contraindications, info["notes_penalty"] = True

        Args:
            action: The agent's TriageDecision wrapped in an Action

        Returns:
            StepResult with reward, next observation, done flag, and info dict.

        Raises:
            RuntimeError: If called before reset() or after episode is done.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if not self._task_id:
            raise RuntimeError("No active episode. Call reset() first.")

        current_case = self._cases[self._case_index]

        # ── Grade the action ──────────────────────────────────────────────────
        reward_obj = grade_action(
            action=action,
            case=current_case,
            weights=self._task_meta["grader_weights"],
        )
        step_score = reward_obj.score

        # ── Record history ────────────────────────────────────────────────────
        self._step_scores.append(step_score)
        self._history.append({
            "step":           self._case_index + 1,
            "patient_id":     current_case["patient_form_data"]["patient_id"],
            "case_description": current_case.get("case_description", ""),
            "action": {
                "urgency_level":     action.triage.urgency_level.value,
                "specialist_referral": action.triage.specialist_referral.value,
                "medication_flags":  action.triage.medication_flags or [],
                "contraindications": action.triage.contraindications or [],
                "clinical_notes":    action.triage.clinical_notes or "",
            },
            "reward_breakdown": {
                "urgency_score":           reward_obj.urgency_score,
                "specialist_score":        reward_obj.specialist_score,
                "medication_score":        reward_obj.medication_score,
                "contraindication_score":  reward_obj.contraindication_score,
                "total":                   step_score,
            },
            "feedback": reward_obj.feedback,
        })

        # ── Advance to next case ──────────────────────────────────────────────
        self._case_index += 1
        done = self._case_index >= len(self._cases)
        self._done = done

        # ── Build info dict ───────────────────────────────────────────────────
        avg_score = sum(self._step_scores) / len(self._step_scores)
        info: Dict[str, Any] = {
            "step":          self._case_index,       # steps completed
            "step_score":    step_score,
            "average_score": round(avg_score, 4),
            "feedback":      reward_obj.feedback,
            "reward_breakdown": {
                "urgency_score":           reward_obj.urgency_score,
                "specialist_score":        reward_obj.specialist_score,
                "medication_score":        reward_obj.medication_score,
                "contraindication_score":  reward_obj.contraindication_score,
            },
        }

        if done:
            # ── Episode-end bonus for excellent performance ───────────────────
            bonus = 0.05 if avg_score >= 0.90 else 0.0
            info["episode_complete"]  = True
            info["final_score"]       = round(avg_score, 4)
            info["step_scores"]       = self._step_scores[:]
            info["bonus"]             = bonus
            info["summary"]           = (
                f"Episode finished. "
                f"Average score: {avg_score:.4f}. "
                + (f"Excellent performance bonus: +{bonus}" if bonus > 0 else "")
            )

        # ── Build next observation ────────────────────────────────────────────
        if not done:
            next_obs = self._build_observation(
                self._cases[self._case_index],
                previous_reward=step_score,
            )
        else:
            # Episode done — return the last observation again with done context
            next_obs = self._build_observation(
                current_case,
                previous_reward=step_score,
                is_final=True,
            )

        return StepResult(
            observation=next_obs,
            reward=step_score,
            done=done,
            info=info,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # state()
    # ──────────────────────────────────────────────────────────────────────────

    def state(self) -> EnvironmentState:
        """
        Return a snapshot of the current environment state.

        Can be called at any point during an episode.
        Useful for logging, debugging, and the /state API endpoint.
        """
        avg = (
            sum(self._step_scores) / len(self._step_scores)
            if self._step_scores else 0.0
        )

        return EnvironmentState(
            session_id=self._session_id or "not-started",
            task_id=self._task_id or "none",
            step_number=self._case_index,
            max_steps=len(self._cases) if self._cases else 0,
            done=self._done,
            current_score=round(avg, 4),
            step_scores=self._step_scores[:],
            episode_history=self._history[:],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        case: dict,
        previous_reward: Optional[float] = None,
        is_final: bool = False,
    ) -> Observation:
        """Construct an Observation from a case dict."""
        form_data = case["patient_form_data"]

        # Build PatientVitals if present
        vitals_data = form_data.get("vitals")
        vitals = PatientVitals(**vitals_data) if vitals_data else None

        patient_form = PatientIntakeForm(
            patient_id=form_data["patient_id"],
            age=form_data["age"],
            gender=form_data["gender"],
            chief_complaint=form_data["chief_complaint"],
            symptoms=form_data["symptoms"],
            symptom_duration_hours=form_data.get("symptom_duration_hours"),
            vitals=vitals,
            current_medications=form_data.get("current_medications"),
            proposed_medications=form_data.get("proposed_medications"),
            allergies=form_data.get("allergies"),
            medical_history=form_data.get("medical_history"),
        )

        step_num = self._case_index + (0 if not is_final else 0)

        return Observation(
            task_id=self._task_id,
            task_name=self._task_meta["name"],
            task_description=self._task_meta["description"],
            patient_form=patient_form,
            step_number=self._case_index + 1,
            max_steps=len(self._cases),
            previous_reward=previous_reward,
        )