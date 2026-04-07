"""
inference.py — ICU Resource Allocation OpenEnv
================================================
Calls the running FastAPI server at localhost:7860 via HTTP.
Uses OpenAI client for LLM decisions, falls back to rule-based agent on any error.

Mandatory stdout format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
import sys
import time
import json
import requests
from typing import List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

# The running server (same container)
ENV_BASE_URL      = "http://localhost:7860"
BENCHMARK         = "icu-resource-allocation"
MAX_STEPS         = 48
SUCCESS_THRESHOLD = 0.40
REQUEST_TIMEOUT   = 30   # seconds

ACTION_NAMES = {
    0: "HOLD", 1: "ADMIT_CRITICAL", 2: "ADMIT_FIFO",
    3: "TRANSFER_OUT", 4: "CALL_EXTRA_NURSE",
    5: "SPECIALIST_CONSULT", 6: "EXPEDITE_BED",
}

SYSTEM_PROMPT = (
    "You are an ICU charge coordinator at a 500-bed Indian hospital. "
    "Every 30 minutes allocate scarce ICU resources. "
    "20 beds | 12 vents | 4 dialysis | budget Rs150000/day. "
    "NABH: max 2 patients/nurse. SOFA>=11 = CRITICAL, admit within 2 hours. "
    "ACTIONS - reply ONLY the digit: "
    "0=HOLD 1=ADMIT_CRITICAL 2=ADMIT_FIFO 3=TRANSFER_OUT "
    "4=CALL_EXTRA_NURSE 5=SPECIALIST_CONSULT 6=EXPEDITE_BED. "
    "Priority: admit critical fast, keep nurse ratio<=2.0, stay within budget. "
    "Reply with ONE digit 0-6 and nothing else."
)


# ── Stdout log helpers ────────────────────────────────────────────────────────

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── HTTP helpers (call the running server) ────────────────────────────────────

def _wait_for_server(max_wait: int = 60) -> bool:
    """Wait until the FastAPI server is ready."""
    for _ in range(max_wait):
        try:
            r = requests.get(f"{ENV_BASE_URL}/", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def _reset_env(seed: int = 42) -> Optional[dict]:
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"seed": seed},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[DEBUG] reset failed: {e}", flush=True)
        return None

def _step_env(action: int) -> Optional[tuple]:
    """Returns (obs, reward, done, info) or None on failure."""
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        obs    = data["observation"]
        reward = float(data["reward"])
        done   = bool(data["done"])
        info   = data.get("info", {})
        return obs, reward, done, info
    except Exception as e:
        print(f"[DEBUG] step failed: {e}", flush=True)
        return None


# ── LLM + fallback agent ──────────────────────────────────────────────────────

def _obs_to_prompt(obs: dict) -> str:
    shift = ["Day", "Evening", "Night"][int(obs.get("shift", 0))]
    return (
        f"Step {obs.get('step',0)}/48 {obs.get('time_of_day',8):.0f}:00 {shift}\n"
        f"Beds {obs.get('beds_occupied',0)}/20 | "
        f"free={obs.get('beds_available',0)} | "
        f"turnover={obs.get('beds_in_turnover',0)}\n"
        f"Queue {obs.get('queue_total',0)}: "
        f"CRITICAL={obs.get('queue_critical',0)} "
        f"SEVERE={obs.get('queue_severe',0)} "
        f"MOD={obs.get('queue_moderate',0)} | "
        f"longest_wait={obs.get('queue_max_wait_steps',0)} steps\n"
        f"ICU avg_SOFA={obs.get('avg_icu_sofa',0):.1f} "
        f"avg_mortality={obs.get('avg_icu_mortality_risk',0):.1%}\n"
        f"Nurses {obs.get('nurses_on_duty',10)} "
        f"ratio={obs.get('nurse_patient_ratio',1.2):.1f}/2.0\n"
        f"Budget Rs{obs.get('budget_remaining_inr',150000):,.0f} "
        f"({100-obs.get('budget_utilisation_pct',0):.0f}% left)\n"
        f"deaths={obs.get('deaths_in_queue',0)} "
        f"adverse={obs.get('adverse_events',0)}\n"
        "Action? (single digit 0-6)"
    )

def _fallback(obs: dict) -> int:
    """Rule-based fallback — always safe, never crashes."""
    try:
        if obs.get("queue_critical", 0) > 0 and obs.get("beds_available", 0) > 0:
            return 1  # ADMIT_CRITICAL
        if obs.get("nurse_patient_ratio", 1.0) > 2.2:
            return 4  # CALL_EXTRA_NURSE
        if obs.get("queue_total", 0) > 0 and obs.get("beds_available", 0) == 0:
            return 3  # TRANSFER_OUT
        if obs.get("queue_total", 0) > 0 and obs.get("beds_available", 0) > 0:
            return 2  # ADMIT_FIFO
    except Exception:
        pass
    return 0  # HOLD

def _get_action(client, obs: dict) -> tuple[int, Optional[str]]:
    """Try LLM, fall back to rule-based on any error."""
    if client is None:
        return _fallback(obs), "no_llm_client"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _obs_to_prompt(obs)},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw and raw[0].isdigit():
            a = int(raw[0])
            if 0 <= a <= 6:
                return a, None
        return _fallback(obs), f"bad_output:{raw[:20]}"
    except Exception as e:
        return _fallback(obs), f"llm_err:{type(e).__name__}"


# ── Inline scoring ────────────────────────────────────────────────────────────

def _score(task_id: str, m: dict) -> float:
    try:
        if task_id == "task_easy":
            s = (0.50 * max(0.0, 1.0 - m["deaths"] * 0.30)
               + 0.25 * max(0.0, 1.0 - m["ratio_breach_frac"] * 3.0)
               + 0.25 * min(1.0, m["admissions"] / 6.0))
        elif task_id == "task_medium":
            s = (0.35 * max(0.0, 1.0 - m["deaths"] * 0.35)
               + 0.25 * max(0.0, 1.0 - m["ratio_breach_frac"] * 4.0)
               + 0.25 * max(0.0, 1.0 - m["wait_violations"] * 0.12)
               + 0.15 * min(1.0, m["admissions"] / 8.0))
        elif task_id == "task_hard":
            bu = m["budget_used_pct"]
            s = (0.30 * max(0.0, 1.0 - m["deaths"] * 0.35)
               + 0.20 * max(0.0, 1.0 - m["adverse"] * 0.10)
               + 0.20 * max(0.0, 1.0 - m["wait_violations"] * 0.12)
               + 0.15 * (1.0 if bu <= 0.85 else max(0.0, 1.0-(bu-0.85)*4))
               + 0.15 * (1.0 if m["sofa_trend"] <= 0
                         else max(0.0, 1.0 - m["sofa_trend"] / 5.0)))
        else:
            s = 0.0
        return round(min(1.0, max(0.0, s)), 3)
    except Exception:
        return 0.0


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(task_id: str, client) -> None:
    rewards:        List[float] = []
    ratio_breaches: List[bool]  = []
    sofa_traj:      List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    obs         = {}

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment via HTTP
        obs = _reset_env(seed=42)
        if obs is None:
            print("[DEBUG] reset returned None, using empty obs", flush=True)
            obs = {}

        done = False

        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            action_int, llm_error = _get_action(client, obs)
            action_str = ACTION_NAMES.get(action_int, str(action_int))

            result = _step_env(action_int)
            if result is None:
                # Server call failed — end episode gracefully
                log_step(step=step_n, action=action_str,
                         reward=0.0, done=True,
                         error="step_http_failed")
                steps_taken = step_n
                break

            obs, reward, done, _ = result
            rewards.append(reward)
            ratio_breaches.append(
                float(obs.get("nurse_patient_ratio", 1.0)) > 2.0
            )
            sofa_traj.append(float(obs.get("avg_icu_sofa", 0.0)))
            steps_taken = step_n

            log_step(step=step_n, action=action_str,
                     reward=reward, done=done, error=llm_error)

        # Compute score
        if steps_taken > 0:
            n = max(1, len(ratio_breaches))
            metrics = {
                "deaths":            int(obs.get("deaths_in_queue", 0)),
                "adverse":           int(obs.get("adverse_events", 0)),
                "wait_violations":   int(obs.get("wait_violations", 0)),
                "admissions":        int(obs.get("admissions_today", 0)),
                "ratio_breach_frac": sum(ratio_breaches) / n,
                "budget_used_pct":   float(obs.get("budget_utilisation_pct", 0)) / 100.0,
                "sofa_trend":        (sofa_traj[-1] - sofa_traj[0])
                                     if len(sofa_traj) >= 2 else 0.0,
            }
            score   = _score(task_id, metrics)
            success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] task={task_id} unhandled exception: {e}", flush=True)
        # score and success remain 0/False — episode still gets [END] line

    finally:
        # [END] is ALWAYS emitted, even on exception
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Wait for server to be ready
    print("[DEBUG] Waiting for server...", flush=True)
    if not _wait_for_server(max_wait=60):
        print("[DEBUG] Server not ready after 60s, proceeding anyway", flush=True)

    # Build OpenAI client — safe even if token is missing
    client = None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN if HF_TOKEN else "dummy",
        )
    except Exception as e:
        print(f"[DEBUG] OpenAI client failed: {e}, using fallback only", flush=True)

    # Run all 3 tasks — each gets its own [START]..[END] block
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            run_task(task_id, client)
        except Exception as e:
            # Last-resort: emit [END] so parser doesn't hang
            print(f"[DEBUG] run_task crashed: {e}", flush=True)
            print(
                f"[END] success=false steps=0 score=0.000 rewards=",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEBUG] main crashed: {e}", flush=True)
    finally:
        sys.exit(0)   # ALWAYS exit 0 — never non-zero
