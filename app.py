"""FastAPI server — OpenEnv-compliant step() / reset() / state() endpoints."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import ICUEnv

app = FastAPI(
    title="ICU Resource Allocation — OpenEnv",
    description=(
        "Real-world OpenEnv: AI agent allocates ICU beds, staff and equipment "
        "in a 20-bed Indian tertiary-care ICU over a 24-hour duty cycle."
    ),
    version="1.0.0",
)

_env = ICUEnv(seed=42)


# ── Models ────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    action: int  # 0-6, see env.py for description

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "env": "ICU Resource Allocation OpenEnv v1.0",
        "actions": {
            0: "HOLD",
            1: "ADMIT_CRITICAL – admit highest-SOFA patient",
            2: "ADMIT_FIFO – admit longest-waiting patient",
            3: "TRANSFER_OUT – move stable patient to step-down",
            4: "CALL_EXTRA_NURSE – overtime nurse (₹1 200)",
            5: "SPECIALIST_CONSULT – reduce mortality risk (₹3 500)",
            6: "EXPEDITE_BED – faster bed turnover (₹600)",
        },
    }


@app.post("/reset", summary="Reset environment")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = ICUEnv(seed=req.seed if req.seed is not None else 42)
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResponse, summary="Take an action")
def step(req: StepRequest):
    if req.action not in range(7):
        raise HTTPException(status_code=400, detail="action must be 0–6")
    obs, reward, done, info = _env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", summary="Current state (no action)")
def state():
    return _env.state()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
