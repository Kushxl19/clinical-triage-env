"""
main.py
───────
FastAPI server that exposes the ClinicalTriageEnv as an HTTP API.

This is what runs on Hugging Face Spaces (port 7860).

Endpoints:
  GET  /              → HTML landing page (human-readable intro)
  GET  /health        → {"status": "ok"}  (ping test)
  GET  /tasks         → list of available tasks with metadata
  POST /reset         → start a new episode, returns first Observation
  POST /step          → submit an action, get StepResult
  GET  /state         → current EnvironmentState for a session
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import Request


from src.data        import TASK_REGISTRY
from src.environment import ClinicalTriageEnv
from src.models      import (
    Action,
    EnvironmentState,
    Observation,
    ResetRequest,
    StepRequest,
    StepResult,
)


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical Triage OpenEnv",
    description=(
        "An OpenEnv-compatible environment where AI agents practice "
        "clinical patient triage: urgency scoring, specialist routing, "
        "and medication safety checking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────────────────────────
# Each session_id maps to its own ClinicalTriageEnv instance.
# In production you'd use Redis or a database; for the hackathon this is fine.
SESSIONS: Dict[str, ClinicalTriageEnv] = {}


def get_session(session_id: str) -> ClinicalTriageEnv:
    """Retrieve a session or raise HTTP 404."""
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset first.",
        )
    return env


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Info"])
def landing_page() -> str:
    """HTML landing page shown on Hugging Face Spaces."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8"/>
      <title>Clinical Triage OpenEnv</title>
      <style>
        body{font-family:system-ui,sans-serif;max-width:820px;margin:60px auto;padding:0 20px;line-height:1.6;color:#1a1a1a;}
        h1{color:#c0392b;} h2{color:#2c3e50;border-bottom:2px solid #eee;padding-bottom:6px;}
        code{background:#f4f4f4;padding:2px 6px;border-radius:4px;font-size:0.9em;}
        pre{background:#f4f4f4;padding:16px;border-radius:8px;overflow-x:auto;}
        .badge{display:inline-block;background:#27ae60;color:white;padding:3px 10px;border-radius:12px;font-size:0.8em;margin-left:8px;}
        .badge.med{background:#e67e22;} .badge.hard{background:#c0392b;}
        table{width:100%;border-collapse:collapse;} th,td{padding:10px;text-align:left;border:1px solid #ddd;}
        th{background:#f8f8f8;}
      </style>
    </head>
    <body>
      <h1>🏥 Clinical Triage OpenEnv</h1>
      <p>An <strong>OpenEnv-compatible</strong> environment where AI agents practice real-world clinical patient triage.</p>

      <h2>Tasks</h2>
      <table>
        <tr><th>ID</th><th>Name</th><th>Difficulty</th><th>What's graded</th></tr>
        <tr><td><code>task_1</code></td><td>Urgency Classification</td><td><span class="badge">Easy</span></td><td>Urgency level accuracy</td></tr>
        <tr><td><code>task_2</code></td><td>Specialist Routing</td><td><span class="badge med">Medium</span></td><td>Urgency (40%) + Specialist (60%)</td></tr>
        <tr><td><code>task_3</code></td><td>Medication Safety Triage</td><td><span class="badge hard">Hard</span></td><td>Urgency + Specialist + Drug flags + Contraindications (25% each)</td></tr>
      </table>

      <h2>Quick Start</h2>
      <pre>
# 1. Reset (start episode)
POST /reset
{"task_id": "task_1"}

# 2. Step (submit triage decision)
POST /step
{
  "session_id": "&lt;from reset response&gt;",
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

# 3. Check state
GET /state?session_id=&lt;id&gt;
      </pre>

      <h2>API Docs</h2>
      <p>Interactive Swagger UI: <a href="/docs">/docs</a> &nbsp;|&nbsp; ReDoc: <a href="/redoc">/redoc</a></p>
    </body>
    </html>
    """


@app.get("/health", tags=["Info"])
def health_check() -> Dict[str, str]:
    """Liveness probe — returns 200 OK if the service is running."""
    return {"status": "ok", "service": "clinical-triage-openenv", "version": "1.0.0"}


@app.get("/tasks", tags=["Info"])
def list_tasks() -> Dict[str, Any]:
    """Return metadata for all available tasks."""
    tasks = []
    for task_id, meta in TASK_REGISTRY.items():
        tasks.append({
            "task_id":     task_id,
            "name":        meta["name"],
            "difficulty":  meta["difficulty"],
            "description": meta["description"],
            "max_steps":   meta["max_steps"],
            "num_cases":   len(meta["cases"]),
            "grader_weights": meta["grader_weights"],
        })
    return {"tasks": tasks, "total": len(tasks)}

@app.post("/reset", tags=["OpenEnv"])
async def reset(req: Request):
    try:
        body = await req.json()
    except:
        body = {}

    task_id = body.get("task_id", "task_1")
    session_id = body.get("session_id", str(uuid.uuid4()))

    env = ClinicalTriageEnv()
    observation = env.reset(task_id=task_id, session_id=session_id)

    SESSIONS[session_id] = env

    return {
        "session_id": session_id,
        "observation": observation.model_dump()
    }

@app.post("/step", response_model=Dict[str, Any], tags=["OpenEnv"])
def step(request: StepRequest) -> Dict[str, Any]:
    """
    **OpenEnv step(action)** — Submit a triage decision for the current patient.

    Returns:
    - `observation`: next patient's intake form (or final state if done)
    - `reward`:      score for this step (0.0–1.0)
    - `done`:        True when all patients have been triaged
    - `info`:        detailed score breakdown and feedback
    """
    env = get_session(request.session_id)

    try:
        result: StepResult = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    return result.model_dump()


@app.get("/state", response_model=Dict[str, Any], tags=["OpenEnv"])
def state(session_id: str) -> Dict[str, Any]:
    """
    **OpenEnv state()** — Get a snapshot of the current environment state.

    Shows step number, running average score, and full episode history.
    """
    env = get_session(session_id)
    return env.state().model_dump()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # HF Spaces expects the service on port 7860
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)