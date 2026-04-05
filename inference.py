"""
inference.py — ICU Resource Allocation OpenEnv
================================================
Mandatory stdout format (auto-parsed by judges):

  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
import sys
from typing import List, Optional

from openai import OpenAI
from env import ICUEnv

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

BENCHMARK         = "icu-resource-allocation"
MAX_STEPS         = 48
SUCCESS_THRESHOLD = 0.40

ACTION_NAMES = {
    0: "HOLD", 1: "ADMIT_CRITICAL", 2: "ADMIT_FIFO",
    3: "TRANSFER_OUT", 4: "CALL_EXTRA_NURSE",
    5: "SPECIALIST_CONSULT", 6: "EXPEDITE_BED",
}

SYSTEM_PROMPT = """You are an ICU charge coordinator at a 500-bed Indian hospital.
Every 30 minutes allocate scarce ICU resources. 20 beds | 12 vents | 4 dialysis | budget ₹1,50,000/day.
NABH: max 2 patients/nurse. SOFA≥11 = CRITICAL must admit within 2 hours.
ACTIONS (reply ONLY the digit):
0=HOLD 1=ADMIT_CRITICAL 2=ADMIT_FIFO 3=TRANSFER_OUT 4=CALL_EXTRA_NURSE 5=SPECIALIST_CONSULT 6=EXPEDITE_BED
Priority: admit critical fast > keep nurse ratio ≤2.0 > stay within budget.
Reply with ONE digit 0-6."""


# ── Stdout helpers (exact format required) ────────────────────────────────────

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── LLM + fallback ────────────────────────────────────────────────────────────

def _obs_prompt(obs: dict) -> str:
    shift = ["Day", "Evening", "Night"][obs["shift"]]
    return (
        f"Step {obs['step']}/48 {obs['time_of_day']:.0f}:00 {shift}\n"
        f"Beds {obs['beds_occupied']}/20 | free={obs['beds_available']} | turnover={obs['beds_in_turnover']}\n"
        f"Queue {obs['queue_total']}: CRITICAL={obs['queue_critical']} SEVERE={obs['queue_severe']} MOD={obs['queue_moderate']} | longest_wait={obs['queue_max_wait_steps']}steps\n"
        f"ICU avg_SOFA={obs['avg_icu_sofa']:.1f} avg_mortality={obs['avg_icu_mortality_risk']:.1%}\n"
        f"Vents {obs['ventilators_in_use']}/12 | Dialysis {4-obs['dialysis_available']}/4\n"
        f"Nurses {obs['nurses_on_duty']} ratio={obs['nurse_patient_ratio']:.1f}/2.0\n"
        f"Budget ₹{obs['budget_remaining_inr']:,.0f} ({100-obs['budget_utilisation_pct']:.0f}% left)\n"
        f"deaths={obs['deaths_in_queue']} adverse={obs['adverse_events']} wait_violations={obs['wait_violations']}\n"
        "Action?"
    )

def _fallback(obs: dict) -> int:
    if obs["queue_critical"] > 0 and obs["beds_available"] > 0: return 1
    if obs["nurse_patient_ratio"] > 2.2: return 4
    if obs["queue_total"] > 0 and obs["beds_available"] == 0: return 3
    if obs["queue_total"] > 0 and obs["beds_available"] > 0: return 2
    return 0

def _get_action(client: OpenAI, obs: dict) -> tuple[int, Optional[str]]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _obs_prompt(obs)},
            ],
            max_tokens=5, temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        a = int(raw[0])
        return (a, None) if 0 <= a <= 6 else (_fallback(obs), f"bad_output:{raw}")
    except Exception as e:
        return _fallback(obs), f"llm_err:{type(e).__name__}"


# ── Inline scoring (mirrors graders/task_graders.py exactly) ─────────────────

def _score(task_id: str, m: dict) -> float:
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
           + 0.15 * (1.0 if bu <= 0.85 else max(0.0, 1.0 - (bu - 0.85) * 4))
           + 0.15 * (1.0 if m["sofa_trend"] <= 0 else max(0.0, 1.0 - m["sofa_trend"] / 5.0)))
    else:
        s = 0.0
    return round(min(1.0, max(0.0, s)), 3)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI) -> None:
    env = ICUEnv(seed=42)
    rewards: List[float] = []
    ratio_breaches: List[bool] = []
    sofa_traj: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        done = False

        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            action_int, error = _get_action(client, obs)
            action_str = ACTION_NAMES.get(action_int, str(action_int))

            try:
                obs, reward, done, _ = env.step(action_int)
            except Exception as e:
                reward, done, error = 0.0, True, f"env_err:{e}"

            rewards.append(reward)
            ratio_breaches.append(obs["nurse_patient_ratio"] > 2.0)
            sofa_traj.append(obs["avg_icu_sofa"])
            steps_taken = step_n

            log_step(step=step_n, action=action_str, reward=reward, done=done, error=error)

        n = max(1, len(ratio_breaches))
        metrics = {
            "deaths":            obs["deaths_in_queue"],
            "adverse":           obs["adverse_events"],
            "wait_violations":   obs["wait_violations"],
            "admissions":        obs["admissions_today"],
            "ratio_breach_frac": sum(ratio_breaches) / n,
            "budget_used_pct":   obs["budget_utilisation_pct"] / 100.0,
            "sofa_trend":        (sofa_traj[-1] - sofa_traj[0]) if len(sofa_traj) >= 2 else 0.0,
        }
        score   = _score(task_id, metrics)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] task={task_id} exception: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        run_task(task_id, client)
    sys.exit(0)
