---
title: Clinical Triage OpenEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
- openenv
---
# 🏥 Clinical Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/openenv)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An OpenEnv-compatible environment where AI agents practice real-world clinical patient triage.**

---

## 🎯 What Is This?

`clinical-triage-env` is a reinforcement learning environment built on the [OpenEnv](https://github.com/openenv) standard. AI agents are placed in the role of a clinical triage system at a hospital — they receive patient intake forms and must make evidence-based decisions.

This environment models a task that **real clinicians perform every day**: reading a patient's symptoms, vitals, medications, and history, then deciding:
1. **How urgently** does this patient need care?
2. **Which specialist** should they see?
3. **Are their medications safe** given what else they're taking?

Getting these decisions wrong in the real world has life-or-death consequences — making it a compelling and meaningful benchmark for evaluating AI agents.

---

## 🏗️ Environment Design

### Episode Structure

Each episode = 1 task × 3 patient cases = **3 steps**:

```
reset(task_id="task_1")        → Observation(Patient 1)
step(TriageDecision for P1)   → StepResult(reward=0.8, Observation(Patient 2), done=False)
step(TriageDecision for P2)   → StepResult(reward=1.0, Observation(Patient 3), done=False)
step(TriageDecision for P3)   → StepResult(reward=0.6, final_obs,             done=True)
                                            ↑ final_score = avg(0.8, 1.0, 0.6) = 0.80
```

### Reward Function

Reward is given **at every step** (partial progress signal, not just episode end).

Each step reward = weighted sum of component scores:

| Component | Scoring Logic |
|-----------|--------------|
| **Urgency** | Ordinal distance: exact=1.0, ±1 level=0.6, ±2=0.2, ±3=0.0 |
| **Specialist** | Exact match=1.0, same clinical group=0.5, wrong=0.0 |
| **Medication Flags** | F1 score between predicted and expected flagged drugs |
| **Contraindications** | Concept-group keyword matching (any synonym counts) |

**Episode bonus**: +0.05 if average score ≥ 0.90 (excellent performance)

---

## 📋 Tasks

### Task 1 — Urgency Classification `[Easy]`

**Objective**: Given a patient intake form, correctly classify urgency as `low`, `medium`, or `high` or `critical`.

**Grader**: 100% urgency accuracy (ordinal distance scoring)

**Cases**:
| Patient | Presentation | Expected Urgency |
|---------|-------------|-----------------|
| T1-001 | 24yo female, mild sore throat, low-grade fever | `low` |
| T1-002 | 56yo male, crushing chest pain radiating to left arm, sweating | `critical` |
| T1-003 | 68yo female, sudden right-sided weakness + slurred speech | `critical` |

**Expected baseline score**: ~0.78 (easy cases, but T1-001/T1-002 require careful reading)

---

### Task 2 — Specialist Routing `[Medium]`

**Objective**: Correctly identify both urgency level AND the right medical specialist.

**Grader**: 40% urgency + 60% specialist routing accuracy

**Cases**:
| Patient | Presentation | Expected Specialist |
|---------|-------------|---------------------|
| T2-001 | 38yo male, recurrent chest tightness + palpitations during exercise | `cardiologist` |
| T2-002 | 34yo female, 3-month depression with passive suicidal ideation | `psychiatrist` |
| T2-003 | 22yo male, knee swelling after soccer injury | `orthopedic` |

**Expected baseline score**: ~0.65 (requires clinical knowledge of specialties)

---

### Task 3 — Full Triage with Medication Safety `[Hard]`

**Objective**: Complete triage + identify dangerous proposed medications + describe contraindications.

**Grader**: 25% urgency + 25% specialist + 25% medication flags + 25% contraindication concepts

**Cases** (all with active pharmacological hazards):
| Patient | Hidden Danger | Why It's Hard |
|---------|--------------|---------------|
| T3-001 | Warfarin patient prescribed ibuprofen | Warfarin + NSAIDs → major bleeding risk |
| T3-002 | Penicillin-allergic patient prescribed amoxicillin | Amoxicillin IS a penicillin-class drug |
| T3-003 | MAOI patient (phenelzine) prescribed SSRI (sertraline) | MAOI + SSRI → potentially fatal serotonin syndrome |

**Expected baseline score**: ~0.45 (requires deep pharmacological knowledge)

---

## 🔌 API Reference

### `POST /reset`
Start a new episode.
```json
Request:  {"task_id": "task_1"}
Response: {"session_id": "uuid", "observation": {...}}
```

### `POST /step`
Submit a triage decision.
```json
Request:
{
  "session_id": "uuid",
  "action": {
    "action_type": "triage_decision",
    "triage": {
      "urgency_level": "critical",
      "specialist_referral": "cardiologist",
      "medication_flags": [],
      "contraindications": [],
      "clinical_notes": "Classic STEMI presentation."
    }
  }
}
Response: {"observation": {...}, "reward": 1.0, "done": false, "info": {...}}
```

### `GET /state?session_id=<id>`
Get current environment state (step number, scores, history).

### `GET /tasks`
List all tasks with metadata.

### `GET /health`
Liveness check — returns `{"status": "ok"}`.

---

## 📐 Observation & Action Spaces

### Observation Space
```
Observation
├── task_id          (str)
├── task_name        (str)
├── task_description (str)         ← tells the agent what to do
├── step_number      (int)
├── max_steps        (int)
├── previous_reward  (float | null)
└── patient_form
    ├── patient_id              (str)
    ├── age                     (int)
    ├── gender                  (str)
    ├── chief_complaint         (str)
    ├── symptoms                (list[str])
    ├── symptom_duration_hours  (int | null)
    ├── vitals
    │   ├── blood_pressure_systolic/diastolic  (int)
    │   ├── heart_rate                         (int)
    │   ├── temperature_celsius                (float)
    │   ├── oxygen_saturation                  (float)
    │   ├── respiratory_rate                   (int)
    │   └── pain_scale                         (int 0-10)
    ├── current_medications     (list[str] | null)
    ├── proposed_medications    (list[str] | null)   ← Task 3 key field
    ├── allergies               (list[str] | null)
    └── medical_history         (list[str] | null)
```

### Action Space
```
Action
├── action_type  = "triage_decision"  (always)
└── triage
    ├── urgency_level      (enum: low | medium | high | critical)
    ├── specialist_referral (enum: general_practitioner | cardiologist |
    │                              neurologist | orthopedic | psychiatrist |
    │                              pulmonologist | gastroenterologist |
    │                              endocrinologist | emergency_medicine)
    ├── medication_flags   (list[str])  ← dangerous proposed meds
    ├── contraindications  (list[str])  ← free-text explanation
    └── clinical_notes     (str | null) ← optional reasoning
```

---

## 🚀 Setup & Usage

### Option A: Docker (Recommended)

```bash
# Build
docker build -t clinical-triage-env .

# Run (starts FastAPI server on port 7860)
docker run -p 7860:7860 clinical-triage-env

# Test
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

### Option B: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Running the Baseline Inference Script

```bash
# Set credentials
export HF_TOKEN="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"

# Run baseline
python inference.py
```

The script imports the environment directly (no server needed) and produces:
```
═══════════════════════════════════════════════
  FINAL SCORES
═══════════════════════════════════════════════
  task_1 (Easy)                    0.7778  ████████████████░░░░
  task_2 (Medium)                  0.6267  ████████████░░░░░░░░
  task_3 (Hard)                    0.4583  █████████░░░░░░░░░░░
  ──────────────────────────────────────
  Overall Average                  0.6209
```

---

## 📊 Baseline Scores

Scores produced by `gpt-4o-mini` with `temperature=0` (deterministic):

| Task | Score | Notes |
|------|-------|-------|
| Task 1 — Urgency Classification | **~0.78** | High urgency cases scored well; mild cases occasionally over-triaged |
| Task 2 — Specialist Routing | **~0.63** | Psychiatric routing is the most challenging for the model |
| Task 3 — Medication Safety | **~0.46** | MAOI-SSRI interaction requires specialized pharmacological knowledge |
| **Overall** | **~0.62** | — |

---

## 🧩 Project Structure

```
clinical-triage-env/
├── src/
│   ├── __init__.py        # Package exports
│   ├── models.py          # All Pydantic models (Observation, Action, Reward, ...)
│   ├── data.py            # Patient cases + expected answers for all 3 tasks
│   ├── graders.py         # Deterministic scoring functions
│   └── environment.py     # ClinicalTriageEnv (reset/step/state)
│
├── main.py                # FastAPI server (HF Spaces entry point)
├── inference.py           # Baseline LLM agent (uses OpenAI client)
├── openenv.yaml           # OpenEnv metadata
├── Dockerfile             # Container for HF Spaces
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

---

## 🧠 Why Clinical Triage?

1. **Real-world impact** — Triage errors contribute to patient harm. Better AI triage tools could support overburdened clinical staff.
2. **Multi-dimensional reasoning** — The agent must integrate symptoms, vitals, drug interactions, and clinical guidelines simultaneously.
3. **Pharmacovigilance** — Task 3 tests a genuinely dangerous real-world capability: detecting drug interactions before prescribing.
4. **Novel for RL** — Most existing environments test coding, reasoning, or web browsing. Medical decision-making is underrepresented.
5. **Clear evaluation** — Urgency levels and specialist categories form a clean, deterministic grading space.

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

*Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon.*