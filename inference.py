"""
inference.py
────────────
Baseline inference script for the Clinical Triage OpenEnv.

Runs an LLM agent through all 3 tasks and reports reproducible scores.
Uses the OpenAI Python client — compatible with OpenAI, Hugging Face
Inference, and any OpenAI-compatible API.

Environment variables required:
  API_BASE_URL   — API endpoint  (default: https://api.openai.com/v1)
  MODEL_NAME     — Model to use  (default: gpt-4o-mini)
  HF_TOKEN       — API key / Hugging Face token

Usage:
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o-mini"
  export HF_TOKEN="sk-..."
  python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ── Local imports ─────────────────────────────────────────────────────────────
# We import the environment directly (no HTTP server needed for the baseline).
sys.path.insert(0, os.path.dirname(__file__))
from src.environment import ClinicalTriageEnv
from src.models      import Action, TriageDecision, UrgencyLevel, SpecialistType

# ── Configuration from environment variables ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     os.getenv("OPENAI_API_KEY", ""))

TEMPERATURE  = 0.0    # Deterministic for reproducibility
MAX_TOKENS   = 800
MAX_STEPS    = 10     # Safety cap (episodes are 3 steps)

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ClinicalTriageAI, an expert clinical triage system.

You will receive patient intake forms. For each patient, you must respond ONLY
with a valid JSON object — no markdown, no code fences, no extra text.

JSON schema you must follow exactly:
{
  "urgency_level": "<low|medium|high|critical>",
  "specialist_referral": "<general_practitioner|cardiologist|neurologist|orthopedic|psychiatrist|pulmonologist|gastroenterologist|endocrinologist|emergency_medicine>",
  "medication_flags": ["<drug name if dangerous, else empty list>"],
  "contraindications": ["<description of each contraindication found, else empty list>"],
  "clinical_notes": "<one sentence of clinical reasoning>"
}

Clinical triage guidelines:
- CRITICAL: Immediate life threat (MI, stroke, anaphylaxis, septic shock)
- HIGH: Urgent — patient needs attention within the hour (active psychiatric crisis, significant injury)
- MEDIUM: Semi-urgent — needs attention within hours (moderate pain, ongoing infection)
- LOW: Non-urgent — can wait for routine appointment (minor ailments, mild symptoms)

For medication_flags: list ANY drug in proposed_medications that is dangerous
given the patient's current_medications or allergies.
For contraindications: explain WHY each flagged drug is dangerous.
"""


def build_user_prompt(observation_dict: Dict[str, Any], step: int, history: List[str]) -> str:
    """Build the user-turn prompt from the current observation."""
    form = observation_dict["patient_form"]
    vitals = form.get("vitals") or {}

    history_text = ""
    if history:
        history_text = "\n\nPrevious steps:\n" + "\n".join(history)

    prompt = f"""=== TRIAGE TASK: {observation_dict['task_name']} ===
{observation_dict['task_description']}

Step {step} of {observation_dict['max_steps']}
{history_text}

=== PATIENT INTAKE FORM ===
Patient ID:        {form['patient_id']}
Age / Gender:      {form['age']} / {form['gender']}
Chief Complaint:   {form['chief_complaint']}
Symptoms:          {', '.join(form.get('symptoms', []))}
Duration:          {form.get('symptom_duration_hours', 'unknown')} hours

Vitals:
  Blood Pressure:  {vitals.get('blood_pressure_systolic', 'N/A')}/{vitals.get('blood_pressure_diastolic', 'N/A')} mmHg
  Heart Rate:      {vitals.get('heart_rate', 'N/A')} bpm
  Temperature:     {vitals.get('temperature_celsius', 'N/A')} °C
  SpO2:            {vitals.get('oxygen_saturation', 'N/A')}%
  Resp. Rate:      {vitals.get('respiratory_rate', 'N/A')} breaths/min
  Pain Scale:      {vitals.get('pain_scale', 'N/A')} / 10

Current Medications:   {', '.join(form.get('current_medications') or []) or 'None'}
Proposed Medications:  {', '.join(form.get('proposed_medications') or []) or 'None'}
Known Allergies:       {', '.join(form.get('allergies') or []) or 'None'}
Medical History:       {', '.join(form.get('medical_history') or []) or 'None'}

Respond with ONLY a JSON object. No markdown, no explanation outside the JSON.
"""
    return prompt


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from the LLM's response.
    Handles markdown code fences and leading/trailing whitespace.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        # Grab content between first ``` and last ```
        parts = text.split("```")
        for part in parts[1::2]:  # odd-indexed parts are inside fences
            clean = part.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            if clean.startswith("{"):
                text = clean
                break

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response:\n{response_text[:300]}")


def dict_to_action(parsed: Dict[str, Any]) -> Action:
    """Convert the LLM's parsed JSON dict into a typed Action object."""
    # Validate and coerce urgency
    urgency_str = parsed.get("urgency_level", "medium").lower()
    try:
        urgency = UrgencyLevel(urgency_str)
    except ValueError:
        print(f"  [warn] Unknown urgency '{urgency_str}', defaulting to 'medium'")
        urgency = UrgencyLevel.MEDIUM

    # Validate and coerce specialist
    specialist_str = parsed.get("specialist_referral", "general_practitioner").lower()
    try:
        specialist = SpecialistType(specialist_str)
    except ValueError:
        print(f"  [warn] Unknown specialist '{specialist_str}', defaulting to 'general_practitioner'")
        specialist = SpecialistType.GENERAL

    return Action(
        action_type="triage_decision",
        triage=TriageDecision(
            urgency_level=urgency,
            specialist_referral=specialist,
            medication_flags=parsed.get("medication_flags") or [],
            contraindications=parsed.get("contraindications") or [],
            clinical_notes=parsed.get("clinical_notes") or "",
        ),
    )


def run_episode(task_id: str) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run a complete episode for one task.

    Returns:
        (final_average_score, list_of_step_results)
    """
    print(f"\n{'═'*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'═'*60}")

    env = ClinicalTriageEnv()
    observation = env.reset(task_id=task_id)
    obs_dict    = observation.model_dump()

    history: List[str] = []
    step_results: List[Dict[str, Any]] = []
    total_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        print(f"\n  ── Step {step_num}/{obs_dict['max_steps']} "
              f"| Patient: {obs_dict['patient_form']['patient_id']} ──")
        print(f"  Chief complaint: {obs_dict['patient_form']['chief_complaint']}")

        # ── Build messages ────────────────────────────────────────────────────
        user_prompt = build_user_prompt(obs_dict, step_num, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        # ── Call LLM ──────────────────────────────────────────────────────────
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [ERROR] LLM call failed: {exc}")
            print("  Using fallback action (medium urgency, general_practitioner).")
            response_text = json.dumps({
                "urgency_level":     "medium",
                "specialist_referral": "general_practitioner",
                "medication_flags":  [],
                "contraindications": [],
                "clinical_notes":    "Fallback — LLM unavailable.",
            })

        # ── Parse response ────────────────────────────────────────────────────
        try:
            parsed = parse_llm_response(response_text)
            action = dict_to_action(parsed)
        except (ValueError, KeyError) as e:
            print(f"  [ERROR] Could not parse response: {e}")
            print("  Using fallback action.")
            action = dict_to_action({
                "urgency_level":     "medium",
                "specialist_referral": "general_practitioner",
                "medication_flags":  [],
                "contraindications": [],
                "clinical_notes":    "Parse error fallback.",
            })

        print(f"  Agent → urgency={action.triage.urgency_level.value}, "
              f"specialist={action.triage.specialist_referral.value}, "
              f"flags={action.triage.medication_flags}")

        # ── Step environment ──────────────────────────────────────────────────
        result = env.step(action)
        obs_dict = result.observation.model_dump()

        reward      = result.reward
        total_reward += reward
        done        = result.done
        info        = result.info

        print(f"  Reward: {reward:.4f} | Feedback: {info.get('feedback','')}")

        # Track history for multi-step context
        history.append(
            f"Step {step_num}: urgency={action.triage.urgency_level.value}, "
            f"specialist={action.triage.specialist_referral.value} "
            f"→ reward {reward:+.4f}"
        )

        step_results.append({
            "step":   step_num,
            "reward": reward,
            "action": {
                "urgency":    action.triage.urgency_level.value,
                "specialist": action.triage.specialist_referral.value,
                "flags":      action.triage.medication_flags,
            },
            "feedback": info.get("feedback", ""),
        })

        if done:
            final_score = info.get("final_score", total_reward / step_num)
            print(f"\n  ✅ Episode complete.")
            print(f"  Final score: {final_score:.4f}")
            if info.get("bonus", 0) > 0:
                print(f"  🌟 Excellent performance bonus: +{info['bonus']}")
            return final_score, step_results

        # Small delay to respect rate limits
        time.sleep(0.3)

    # Should not reach here for 3-step episodes, but just in case
    avg = total_reward / step_num if step_num > 0 else 0.0
    return round(avg, 4), step_results


def main() -> None:
    """Run the full baseline evaluation across all 3 tasks."""
    print("\n" + "█" * 62)
    print("  CLINICAL TRIAGE OPENENV — BASELINE INFERENCE")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  API:     {API_BASE_URL}")
    print("█" * 62)

    # Verify API key is set
    if not HF_TOKEN:
        print("\n[ERROR] No API key found.")
        print("  Set HF_TOKEN (or OPENAI_API_KEY) before running:\n")
        print("  export HF_TOKEN='your-key-here'")
        sys.exit(1)

    tasks = ["task_1", "task_2", "task_3"]
    all_scores: Dict[str, float] = {}
    all_results: Dict[str, list] = {}

    start_time = time.time()

    for task_id in tasks:
        try:
            score, results = run_episode(task_id)
            all_scores[task_id]  = score
            all_results[task_id] = results
        except Exception as e:
            print(f"\n[ERROR] Task {task_id} failed: {e}")
            all_scores[task_id]  = 0.0
            all_results[task_id] = []

    elapsed = time.time() - start_time

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  FINAL SCORES")
    print("═" * 62)
    print(f"  {'Task':<35} {'Score':>8}  {'Difficulty'}")
    print("  " + "-" * 58)

    difficulties = {"task_1": "Easy", "task_2": "Medium", "task_3": "Hard"}
    for task_id in tasks:
        score = all_scores.get(task_id, 0.0)
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        diff  = difficulties.get(task_id, "")
        print(f"  {task_id} ({diff}){'':>15} {score:.4f}  {bar}")

    overall = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print("  " + "-" * 58)
    print(f"  {'Overall Average':<35} {overall:.4f}")
    print(f"\n  Time elapsed: {elapsed:.1f}s")
    print("═" * 62 + "\n")

    # ── Machine-readable output ───────────────────────────────────────────────
    output = {
        "model":    MODEL_NAME,
        "scores":   all_scores,
        "overall":  round(overall, 4),
        "elapsed_seconds": round(elapsed, 1),
        "results":  all_results,
    }
    print("JSON_RESULTS:", json.dumps(output))


if __name__ == "__main__":
    main()