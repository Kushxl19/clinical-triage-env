"""
src/data.py
───────────
Patient intake form cases used in each task, with ground-truth expected answers.

Each case is a dict with:
  - patient_form_data: fields for PatientIntakeForm
  - expected:          ground-truth for the grader to compare against
  - case_description:  human-readable summary (for logs/debugging)
"""

from typing import Any, Dict, List

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 CASES — Urgency Classification (Easy)
# Goal: correctly classify urgency level (low / medium / high / critical)
# The specialist field is NOT graded in Task 1 — urgency accuracy is all that matters.
# ══════════════════════════════════════════════════════════════════════════════

TASK_1_CASES: List[Dict[str, Any]] = [
    {
        "case_description": "Young adult with mild sore throat — non-urgent",
        "patient_form_data": {
            "patient_id":             "T1-001",
            "age":                    24,
            "gender":                 "female",
            "chief_complaint":        "Sore throat and mild fatigue for the past 3 days.",
            "symptoms":               ["mild sore throat", "low-grade fever", "slight fatigue", "no difficulty swallowing"],
            "symptom_duration_hours": 72,
            "vitals": {
                "blood_pressure_systolic":  115,
                "blood_pressure_diastolic": 75,
                "heart_rate":               78,
                "temperature_celsius":      37.8,
                "oxygen_saturation":        99.0,
                "respiratory_rate":         14,
                "pain_scale":               2,
            },
            "current_medications":  ["paracetamol as needed"],
            "proposed_medications": [],
            "allergies":            [],
            "medical_history":      [],
        },
        "expected": {
            "urgency_level":     "low",
            "specialist_referral": "general_practitioner",
            "medication_flags":  [],
            "contraindications": [],
        },
    },
    {
        "case_description": "Middle-aged man with chest pain — myocardial infarction presentation",
        "patient_form_data": {
            "patient_id":             "T1-002",
            "age":                    56,
            "gender":                 "male",
            "chief_complaint":        "Crushing chest pain radiating to the left arm, started 45 minutes ago.",
            "symptoms":               ["crushing chest pain", "pain radiating to left arm", "diaphoresis", "shortness of breath", "nausea", "light-headedness"],
            "symptom_duration_hours": 1,
            "vitals": {
                "blood_pressure_systolic":  168,
                "blood_pressure_diastolic": 102,
                "heart_rate":               118,
                "temperature_celsius":      37.0,
                "oxygen_saturation":        93.0,
                "respiratory_rate":         24,
                "pain_scale":               9,
            },
            "current_medications":  ["lisinopril 10mg", "atorvastatin 40mg"],
            "proposed_medications": [],
            "allergies":            [],
            "medical_history":      ["hypertension", "hyperlipidemia", "smoker 20 pack-years"],
        },
        "expected": {
            "urgency_level":     "critical",
            "specialist_referral": "cardiologist",
            "medication_flags":  [],
            "contraindications": [],
        },
    },
    {
        "case_description": "Elderly woman with sudden focal neurological deficit — stroke presentation",
        "patient_form_data": {
            "patient_id":             "T1-003",
            "age":                    68,
            "gender":                 "female",
            "chief_complaint":        "Sudden right-sided weakness and slurred speech, began 20 minutes ago.",
            "symptoms":               ["sudden right arm weakness", "right facial drooping", "slurred speech", "confusion", "no headache"],
            "symptom_duration_hours": 0,
            "vitals": {
                "blood_pressure_systolic":  196,
                "blood_pressure_diastolic": 108,
                "heart_rate":               90,
                "temperature_celsius":      37.2,
                "oxygen_saturation":        96.0,
                "respiratory_rate":         18,
                "pain_scale":               1,
            },
            "current_medications":  ["warfarin 3mg", "metoprolol 25mg"],
            "proposed_medications": [],
            "allergies":            ["sulfa drugs"],
            "medical_history":      ["atrial fibrillation", "hypertension", "type 2 diabetes"],
        },
        "expected": {
            "urgency_level":     "critical",
            "specialist_referral": "neurologist",
            "medication_flags":  [],
            "contraindications": [],
        },
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 CASES — Specialist Routing (Medium)
# Goal: both urgency AND the correct specialist must be identified.
# Grader weights: 40% urgency, 60% specialist routing accuracy.
# ══════════════════════════════════════════════════════════════════════════════

TASK_2_CASES: List[Dict[str, Any]] = [
    {
        "case_description": "Recurrent palpitations with exertional chest discomfort — needs cardiac workup",
        "patient_form_data": {
            "patient_id":             "T2-001",
            "age":                    38,
            "gender":                 "male",
            "chief_complaint":        "Recurring chest tightness and heart pounding episodes for the last 2 weeks, especially during exercise.",
            "symptoms":               ["recurrent chest tightness", "palpitations", "exertional dyspnea", "occasional dizziness", "no syncope"],
            "symptom_duration_hours": 336,
            "vitals": {
                "blood_pressure_systolic":  138,
                "blood_pressure_diastolic": 88,
                "heart_rate":               102,
                "temperature_celsius":      36.9,
                "oxygen_saturation":        97.0,
                "respiratory_rate":         17,
                "pain_scale":               4,
            },
            "current_medications":  [],
            "proposed_medications": [],
            "allergies":            [],
            "medical_history":      ["no significant history", "office worker, sedentary lifestyle"],
        },
        "expected": {
            "urgency_level":     "high",
            "specialist_referral": "cardiologist",
            "medication_flags":  [],
            "contraindications": [],
        },
    },
    {
        "case_description": "Prolonged depression with passive suicidal ideation — psychiatric referral",
        "patient_form_data": {
            "patient_id":             "T2-002",
            "age":                    34,
            "gender":                 "female",
            "chief_complaint":        "I've been feeling completely hopeless for 3 months. I sometimes think everyone would be better off without me.",
            "symptoms":               ["persistent low mood", "anhedonia", "insomnia", "loss of appetite", "fatigue", "passive suicidal ideation", "no active plan or intent"],
            "symptom_duration_hours": 2160,
            "vitals": {
                "blood_pressure_systolic":  118,
                "blood_pressure_diastolic": 76,
                "heart_rate":               68,
                "temperature_celsius":      36.7,
                "oxygen_saturation":        99.0,
                "respiratory_rate":         14,
                "pain_scale":               3,
            },
            "current_medications":  [],
            "proposed_medications": [],
            "allergies":            [],
            "medical_history":      ["anxiety disorder diagnosed 5 years ago"],
        },
        "expected": {
            "urgency_level":     "high",
            "specialist_referral": "psychiatrist",
            "medication_flags":  [],
            "contraindications": [],
        },
    },
    {
        "case_description": "Sports knee injury with swelling — orthopedic evaluation needed",
        "patient_form_data": {
            "patient_id":             "T2-003",
            "age":                    22,
            "gender":                 "male",
            "chief_complaint":        "Right knee pain and swelling after colliding with another player during soccer last night.",
            "symptoms":               ["right knee swelling", "pain on weight bearing", "limited range of motion", "no numbness", "no locking"],
            "symptom_duration_hours": 14,
            "vitals": {
                "blood_pressure_systolic":  122,
                "blood_pressure_diastolic": 80,
                "heart_rate":               84,
                "temperature_celsius":      36.8,
                "oxygen_saturation":        100.0,
                "respiratory_rate":         14,
                "pain_scale":               6,
            },
            "current_medications":  ["ibuprofen 400mg as needed"],
            "proposed_medications": [],
            "allergies":            [],
            "medical_history":      ["previous left ankle sprain 2 years ago"],
        },
        "expected": {
            "urgency_level":     "medium",
            "specialist_referral": "orthopedic",
            "medication_flags":  [],
            "contraindications": [],
        },
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 CASES — Full Triage with Medication Safety (Hard)
# Goal: urgency + specialist + flag dangerous proposed medications + describe contraindications.
# Grader weights: 25% each component.
#
# Each case also contains "contraindication_concept_groups":
#   A list of synonym groups. The grader checks that the agent's contraindication
#   text contains AT LEAST ONE keyword from EACH group.
#   Example: [["warfarin","anticoagulant"], ["nsaid","ibuprofen"], ["bleeding","hemorrhage"]]
#   → agent must mention all 3 concepts (in any synonymous wording) to get full credit.
# ══════════════════════════════════════════════════════════════════════════════

TASK_3_CASES: List[Dict[str, Any]] = [
    {
        "case_description": "Elderly patient on warfarin — NSAID interaction risk",
        "patient_form_data": {
            "patient_id":             "T3-001",
            "age":                    71,
            "gender":                 "male",
            "chief_complaint":        "Chronic osteoarthritis knee pain has flared badly. Requesting stronger pain relief. Physician considering adding ibuprofen.",
            "symptoms":               ["bilateral knee pain", "joint stiffness worse in morning", "reduced mobility", "no fever", "no swelling"],
            "symptom_duration_hours": 168,
            "vitals": {
                "blood_pressure_systolic":  142,
                "blood_pressure_diastolic": 88,
                "heart_rate":               76,
                "temperature_celsius":      36.9,
                "oxygen_saturation":        97.0,
                "respiratory_rate":         16,
                "pain_scale":               7,
            },
            "current_medications":  ["warfarin 5mg daily", "lisinopril 10mg daily", "omeprazole 20mg daily"],
            "proposed_medications": ["ibuprofen 400mg three times daily"],
            "allergies":            [],
            "medical_history":      ["atrial fibrillation", "hypertension", "osteoarthritis"],
        },
        "expected": {
            "urgency_level":     "medium",
            "specialist_referral": "orthopedic",
            "medication_flags":  ["ibuprofen"],
            # Concept groups: agent must mention each of these conceptual clusters
            "contraindication_concept_groups": [
                ["warfarin", "anticoagulant", "blood thinner", "coumadin"],
                ["ibuprofen", "nsaid", "non-steroidal", "anti-inflammatory"],
                ["bleeding", "hemorrhage", "haemorrhage", "bleed"],
            ],
        },
    },
    {
        "case_description": "Patient with penicillin allergy — amoxicillin is a penicillin-class antibiotic",
        "patient_form_data": {
            "patient_id":             "T3-002",
            "age":                    29,
            "gender":                 "female",
            "chief_complaint":        "Sinusitis symptoms for 10 days, not improving. Physician proposing a course of amoxicillin.",
            "symptoms":               ["facial pressure and pain", "nasal congestion", "yellow-green nasal discharge", "headache", "mild fever"],
            "symptom_duration_hours": 240,
            "vitals": {
                "blood_pressure_systolic":  118,
                "blood_pressure_diastolic": 74,
                "heart_rate":               82,
                "temperature_celsius":      38.1,
                "oxygen_saturation":        99.0,
                "respiratory_rate":         15,
                "pain_scale":               4,
            },
            "current_medications":  ["oral contraceptive pill", "cetirizine 10mg"],
            "proposed_medications": ["amoxicillin 500mg three times daily for 7 days"],
            "allergies":            ["penicillin — anaphylaxis documented in records"],
            "medical_history":      ["allergic rhinitis"],
        },
        "expected": {
            "urgency_level":     "low",
            "specialist_referral": "general_practitioner",
            "medication_flags":  ["amoxicillin"],
            "contraindication_concept_groups": [
                ["penicillin", "beta-lactam", "penam"],
                ["allergy", "allergic", "anaphylaxis", "hypersensitivity"],
                ["amoxicillin", "augmentin", "amox"],
            ],
        },
    },
    {
        "case_description": "Patient on MAOI antidepressant — SSRI co-prescription causes serotonin syndrome",
        "patient_form_data": {
            "patient_id":             "T3-003",
            "age":                    47,
            "gender":                 "male",
            "chief_complaint":        "Depression has worsened significantly. Outpatient psychiatry is full; ER team considering starting sertraline while awaiting psychiatric review.",
            "symptoms":               ["severe depressed mood", "hopelessness", "anhedonia", "poor concentration", "social withdrawal", "no active suicidal plan"],
            "symptom_duration_hours": 720,
            "vitals": {
                "blood_pressure_systolic":  124,
                "blood_pressure_diastolic": 80,
                "heart_rate":               72,
                "temperature_celsius":      37.0,
                "oxygen_saturation":        98.0,
                "respiratory_rate":         14,
                "pain_scale":               2,
            },
            "current_medications":  ["phenelzine 45mg daily"],
            "proposed_medications": ["sertraline 50mg daily"],
            "allergies":            [],
            "medical_history":      ["major depressive disorder — treatment resistant", "failed trials of fluoxetine and venlafaxine"],
        },
        "expected": {
            "urgency_level":     "high",
            "specialist_referral": "psychiatrist",
            "medication_flags":  ["sertraline"],
            "contraindication_concept_groups": [
                ["phenelzine", "maoi", "monoamine oxidase", "mao inhibitor"],
                ["sertraline", "ssri", "serotonin reuptake"],
                ["serotonin syndrome", "serotonin toxicity", "serotonergic"],
            ],
        },
    },
]

# ──────────────────────────────────────────────
# Task metadata registry
# ──────────────────────────────────────────────

TASK_REGISTRY = {
    "task_1": {
        "name":        "Urgency Classification",
        "difficulty":  "easy",
        "description": (
            "You are a clinical triage AI. Read the patient intake form carefully "
            "and classify the urgency level. Choose from: low, medium, high, critical. "
            "Also select the most appropriate specialist for referral. "
            "You are scored only on urgency accuracy in this task."
        ),
        "cases":         TASK_1_CASES,
        "max_steps":     len(TASK_1_CASES),
        "grader_weights": {"urgency": 1.0, "specialist": 0.0, "medications": 0.0, "contraindications": 0.0},
    },
    "task_2": {
        "name":        "Specialist Routing",
        "difficulty":  "medium",
        "description": (
            "You are a clinical triage AI. Read the patient intake form and determine: "
            "(1) urgency level — low, medium, high, or critical; "
            "(2) the correct medical specialist to refer the patient to. "
            "Both urgency and specialist routing are graded in this task."
        ),
        "cases":         TASK_2_CASES,
        "max_steps":     len(TASK_2_CASES),
        "grader_weights": {"urgency": 0.40, "specialist": 0.60, "medications": 0.0, "contraindications": 0.0},
    },
    "task_3": {
        "name":        "Full Triage with Medication Safety",
        "difficulty":  "hard",
        "description": (
            "You are a clinical triage AI with pharmacological knowledge. For each patient: "
            "(1) Assess urgency level — low, medium, high, or critical. "
            "(2) Route to the correct specialist. "
            "(3) Examine the proposed_medications list against current_medications and allergies; "
            "list any medications in proposed_medications that are dangerous or contraindicated. "
            "(4) Describe any contraindications or drug interactions found. "
            "All four components are equally weighted in your score."
        ),
        "cases":         TASK_3_CASES,
        "max_steps":     len(TASK_3_CASES),
        "grader_weights": {"urgency": 0.25, "specialist": 0.25, "medications": 0.25, "contraindications": 0.25},
    },
}