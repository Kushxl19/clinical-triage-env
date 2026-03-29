"""
src/graders.py
──────────────
Deterministic scoring functions that evaluate an agent's triage decision
against the ground-truth expected answer.

All graders return floats in [0.0, 1.0].
All graders are deterministic — same input always produces same score.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from .models import Action, Reward


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT SCORERS
# ══════════════════════════════════════════════════════════════════════════════

# Ordinal mapping: lower index = less urgent
_URGENCY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Specialist groupings for partial-credit routing
# Emergency medicine is a valid "safe fallback" for any critical specialty
_SPECIALIST_GROUPS = {
    "cardiac":  {"cardiologist", "emergency_medicine"},
    "neuro":    {"neurologist",  "emergency_medicine"},
    "psych":    {"psychiatrist"},
    "ortho":    {"orthopedic"},
    "pulm":     {"pulmonologist", "emergency_medicine"},
    "gi":       {"gastroenterologist"},
    "endo":     {"endocrinologist"},
    "general":  {"general_practitioner"},
}


def score_urgency(predicted: str, expected: str) -> float:
    """
    Grade urgency level accuracy.

    Scoring (by ordinal distance):
      distance 0  → 1.00  (exact match)
      distance 1  → 0.60  (one level off — understandable clinical uncertainty)
      distance 2  → 0.20  (two levels off — significant error)
      distance 3  → 0.00  (completely wrong — low instead of critical)
    """
    pred_idx = _URGENCY_ORDER.get(predicted.lower(), -1)
    exp_idx  = _URGENCY_ORDER.get(expected.lower(),  -1)

    if pred_idx == -1 or exp_idx == -1:
        return 0.0  # unknown urgency string

    distance = abs(pred_idx - exp_idx)
    score_map = {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0}
    return score_map.get(distance, 0.0)


def score_specialist(predicted: str, expected: str) -> float:
    """
    Grade specialist routing accuracy.

    Scoring:
      exact match          → 1.00
      same specialist group → 0.50  (e.g., emergency_medicine when cardiologist expected)
      different group      → 0.00
    """
    if predicted == expected:
        return 1.0

    # Partial credit: emergency_medicine is a valid escalation for critical patients
    for group_members in _SPECIALIST_GROUPS.values():
        if predicted in group_members and expected in group_members:
            return 0.5

    return 0.0


def score_medication_flags(
    predicted_flags: Optional[List[str]],
    expected_flags:  List[str],
) -> float:
    """
    Grade medication flag detection using F1 score.

    If no medications should be flagged (expected_flags=[]):
      - agent correctly flags nothing     → 1.0
      - agent incorrectly raises a flag   → 0.7 (small false-positive penalty)

    Otherwise uses F1 = 2 * precision * recall / (precision + recall).
    Matching is case-insensitive substring matching so
    "ibuprofen 400mg" still matches expected "ibuprofen".
    """
    pred = predicted_flags or []

    if not expected_flags:
        return 1.0 if not pred else 0.7

    if not pred:
        return 0.0

    # Normalize
    pred_lower = [p.lower().strip() for p in pred]
    exp_lower  = [e.lower().strip() for e in expected_flags]

    # True positives: expected flag found somewhere in predicted flags (substring match)
    def flag_matched(exp_flag: str, pred_list: List[str]) -> bool:
        return any(exp_flag in p or p in exp_flag for p in pred_list)

    tp = sum(1 for ef in exp_lower if flag_matched(ef, pred_lower))
    fp = sum(1 for pf in pred_lower if not flag_matched(pf, exp_lower))
    fn = len(exp_lower) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def score_contraindications(
    predicted_list:   Optional[List[str]],
    concept_groups:   List[List[str]],
) -> float:
    """
    Grade contraindication detection using concept-group matching.

    `concept_groups` is a list of synonym groups.
    The agent gets credit for each group where AT LEAST ONE synonym
    appears anywhere in the combined predicted text.

    Example concept_groups:
      [["warfarin", "anticoagulant"], ["nsaid", "ibuprofen"], ["bleeding"]]
    If the agent writes "NSAIDs are contraindicated with warfarin due to bleeding risk":
      → "warfarin" matched ✓, "nsaid" matched ✓, "bleeding" matched ✓  →  3/3 = 1.0

    If no contraindications expected AND agent gives none → 1.0
    If no contraindications expected BUT agent describes some → 0.7
    """
    pred = predicted_list or []

    if not concept_groups:
        return 1.0 if not pred else 0.7

    if not pred:
        return 0.0

    combined_text = " ".join(pred).lower()

    matched_groups = 0
    for group in concept_groups:
        if any(keyword.lower() in combined_text for keyword in group):
            matched_groups += 1

    return matched_groups / len(concept_groups)


# ══════════════════════════════════════════════════════════════════════════════
# TASK-LEVEL GRADERS
# ══════════════════════════════════════════════════════════════════════════════

def grade_action(
    action:    Action,
    case:      Dict[str, Any],
    weights:   Dict[str, float],
) -> Reward:
    """
    Master grader — scores an action against a case's expected answer,
    using the task's component weights.

    Args:
        action:  The agent's TriageDecision
        case:    One patient case dict from data.py (with "expected" key)
        weights: Dict of {"urgency": w, "specialist": w, "medications": w, "contraindications": w}

    Returns:
        Reward object with overall score and per-component scores.
    """
    expected = case["expected"]
    triage   = action.triage

    # ── Component scores ──────────────────────────────────────────────────────
    urgency_sc = score_urgency(
        predicted=triage.urgency_level.value,
        expected=expected["urgency_level"],
    )

    specialist_sc = score_specialist(
        predicted=triage.specialist_referral.value,
        expected=expected["specialist_referral"],
    )

    medication_sc = score_medication_flags(
        predicted_flags=triage.medication_flags or [],
        expected_flags=expected.get("medication_flags", []),
    )

    # Contraindication groups differ per task:
    # Task 1 & 2: no groups expected  → expects no contraindications
    # Task 3: concept_groups defined
    concept_groups = expected.get("contraindication_concept_groups", [])
    contrain_sc = score_contraindications(
        predicted_list=triage.contraindications or [],
        concept_groups=concept_groups,
    )

    # ── Weighted total ────────────────────────────────────────────────────────
    total = (
        weights.get("urgency",          0.0) * urgency_sc +
        weights.get("specialist",       0.0) * specialist_sc +
        weights.get("medications",      0.0) * medication_sc +
        weights.get("contraindications",0.0) * contrain_sc
    )
    total = round(min(1.0, max(0.0, total)), 4)

    # ── Human-readable feedback ───────────────────────────────────────────────
    exp_urgency    = expected["urgency_level"]
    pred_urgency   = triage.urgency_level.value
    exp_specialist = expected["specialist_referral"]
    pred_specialist= triage.specialist_referral.value

    feedback_parts = []
    if weights.get("urgency", 0) > 0:
        mark = "✓" if urgency_sc == 1.0 else ("~" if urgency_sc > 0 else "✗")
        feedback_parts.append(f"Urgency {mark} ({pred_urgency}, expected {exp_urgency}, score={urgency_sc:.2f})")

    if weights.get("specialist", 0) > 0:
        mark = "✓" if specialist_sc == 1.0 else ("~" if specialist_sc > 0 else "✗")
        feedback_parts.append(f"Specialist {mark} ({pred_specialist}, expected {exp_specialist}, score={specialist_sc:.2f})")

    if weights.get("medications", 0) > 0:
        mark = "✓" if medication_sc >= 0.99 else ("~" if medication_sc > 0 else "✗")
        feedback_parts.append(f"Medication flags {mark} (score={medication_sc:.2f})")

    if weights.get("contraindications", 0) > 0:
        mark = "✓" if contrain_sc >= 0.99 else ("~" if contrain_sc > 0 else "✗")
        feedback_parts.append(f"Contraindications {mark} (score={contrain_sc:.2f})")

    feedback = " | ".join(feedback_parts) + f"  →  TOTAL: {total:.4f}"

    return Reward(
        score=total,
        urgency_score=urgency_sc,
        specialist_score=specialist_sc,
        medication_score=medication_sc,
        contraindication_score=contrain_sc,
        feedback=feedback,
    )