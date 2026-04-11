# -*- coding: utf-8 -*-
# inference.py - ICU Resource Allocation OpenEnv

import os
import sys
import time
import requests
from openai import OpenAI

MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
ENV_BASE_URL = "http://localhost:7860"
BENCHMARK    = "icu-resource-allocation"
MAX_STEPS    = 48
SUCCESS_THRESHOLD = 0.40
REQUEST_TIMEOUT   = 30

ACTION_NAMES = {
    0: "HOLD", 1: "ADMIT_CRITICAL", 2: "ADMIT_FIFO",
    3: "TRANSFER_OUT", 4: "CALL_EXTRA_NURSE",
    5: "SPECIALIST_CONSULT", 6: "EXPEDITE_BED",
}

SYSTEM_PROMPT = (
    "You are an ICU charge coordinator at a 500-bed Indian hospital. "
    "Every 30 minutes allocate scarce ICU resources. "
    "20 beds | 12 vents | 4 dialysis | budget Rs150000/day. "
    "NABH: max 2 patients/nurse. SOFA>=11 is CRITICAL, admit within 2 hours. "
    "ACTIONS - reply ONLY one digit: "
    "0=HOLD 1=ADMIT_CRITICAL 2=ADMIT_FIFO 3=TRANSFER_OUT "
    "4=CALL_EXTRA_NURSE 5=SPECIALIST_CONSULT 6=EXPEDITE_BED. "
    "Priority: admit critical first, keep nurse ratio<=2.0, stay within budget. "
    "Reply with ONE digit 0-6 and nothing else."
)


def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success, steps, score, rewards):
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


def _wait_for_server(max_wait=90):
    for _ in range(max_wait):
        try:
            r = requests.get(ENV_BASE_URL + "/", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def _reset_env(seed=42):
    try:
        r = requests.post(ENV_BASE_URL + "/reset", json={"seed": seed}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[DEBUG] reset failed: {e}", flush=True)
        return {}

def _step_env(action):
    try:
        r = requests.post(ENV_BASE_URL + "/step", json={"action": action}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["observation"], float(data["reward"]), bool(data["done"]), data.get("info", {})
    except Exception as e:
        print(f"[DEBUG] step failed: {e}", flush=True)
        return None

def _obs_to_prompt(obs):
    shift_names = ["Day", "Evening", "Night"]
    shift = shift_names[int(obs.get("shift", 0))]
    return "\n".join([
        f"Step {obs.get('step', 0)}/48  {obs.get('time_of_day', 8):.0f}:00  {shift}",
        f"Beds {obs.get('beds_occupied', 0)}/20 | free={obs.get('beds_available', 0)}",
        f"Queue {obs.get('queue_total', 0)}: CRITICAL={obs.get('queue_critical', 0)}",
        f"Nurses ratio={obs.get('nurse_patient_ratio', 1.2):.1f}/2.0",
        f"Budget Rs{int(obs.get('budget_remaining_inr', 150000))} remaining",
        f"deaths={obs.get('deaths_in_queue', 0)} adverse={obs.get('adverse_events', 0)}",
        "Action? Reply with ONE digit 0-6.",
    ])

def _fallback(obs):
    if obs.get("queue_critical", 0) > 0 and obs.get("beds_available", 0) > 0:
        return 1
    if obs.get("nurse_patient_ratio", 1.0) > 2.2:
        return 4
    if obs.get("queue_total", 0) > 0 and obs.get("beds_available", 0) == 0:
        return 3
    if obs.get("queue_total", 0) > 0 and obs.get("beds_available", 0) > 0:
        return 2
    return 0

def _get_action(client, obs):
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
    print(f"[DEBUG] LLM raw={raw}", flush=True)
    if raw and raw[0].isdigit():
        a = int(raw[0])
        if 0 <= a <= 6:
            return a
    return _fallback(obs)

def _score(task_id, m):
    try:
        if task_id == "task_easy":
            raw = 0.60 * max(0.0, 1.0 - m["deaths"] * 0.25) + 0.40 * (1.0 - m["ratio_breach_frac"])
        elif task_id == "task_medium":
            raw = (0.40 * max(0.0, 1.0 - m["deaths"] * 0.30) +
                   0.30 * (1.0 - m["ratio_breach_frac"]) +
                   0.30 * max(0.0, 1.0 - m["wait_violations"] * 0.15))
        elif task_id == "task_hard":
            bu = m["budget_used_pct"]
            budget_s = 1.0 if bu <= 0.85 else max(0.0, 1.0 - (bu - 0.85) * 4)
            sofa_s   = 1.0 if m["sofa_trend"] <= 0 else max(0.0, 1.0 - m["sofa_trend"] / 5.0)
            raw = (0.30 * max(0.0, 1.0 - m["deaths"] * 0.35) +
                   0.20 * max(0.0, 1.0 - m["adverse"] * 0.10) +
                   0.20 * max(0.0, 1.0 - m["wait_violations"] * 0.12) +
                   0.15 * budget_s + 0.15 * sofa_s)
        else:
            raw = 0.0
        return round(min(0.999, max(0.001, raw * 0.92 + 0.04)), 3)
    except Exception:
        return 0.001

def run_task(task_id, client):
    rewards, ratio_breaches, sofa_traj = [], [], []
    steps_taken = 0
    score = 0.001
    success = False
    obs = {}

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = _reset_env(seed=42)
        done = False

        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            try:
                action_int = _get_action(client, obs)
                llm_error  = None
            except Exception as e:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                action_int = _fallback(obs)
                llm_error  = f"llm_err:{type(e).__name__}"

            action_str = ACTION_NAMES.get(action_int, str(action_int))
            result = _step_env(action_int)
            if result is None:
                log_step(step_n, action_str, 0.0, True, "step_http_failed")
                steps_taken = step_n
                break

            obs, reward, done, _ = result
            rewards.append(reward)
            ratio_breaches.append(float(obs.get("nurse_patient_ratio", 1.0)) > 2.0)
            sofa_traj.append(float(obs.get("avg_icu_sofa", 0.0)))
            steps_taken = step_n
            log_step(step_n, action_str, reward, done, llm_error)

        if steps_taken > 0:
            n = max(1, len(ratio_breaches))
            metrics = {
                "deaths":            int(obs.get("deaths_in_queue", 0)),
                "adverse":           int(obs.get("adverse_events", 0)),
                "wait_violations":   int(obs.get("wait_violations", 0)),
                "ratio_breach_frac": sum(ratio_breaches) / n,
                "budget_used_pct":   float(obs.get("budget_utilisation_pct", 0)) / 100.0,
                "sofa_trend":        (sofa_traj[-1] - sofa_traj[0]) if len(sofa_traj) >= 2 else 0.0,
            }
            score   = _score(task_id, metrics)
            success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] task={task_id} CRASHED: {e}", flush=True)
        import traceback; traceback.print_exc()

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    # Read env vars here — guaranteed to be injected by now
    api_base_url = os.environ["API_BASE_URL"]
    api_key      = os.environ["API_KEY"]

    print(f"[DEBUG] API_BASE_URL={api_base_url}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    print("[DEBUG] Waiting for env server...", flush=True)
    if not _wait_for_server(max_wait=90):
        print("[DEBUG] Server not ready, continuing anyway", flush=True)

    # Create client inside main() — env vars are available here
    client = OpenAI(base_url=api_base_url, api_key=api_key)
    print("[DEBUG] OpenAI client ready", flush=True)

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        run_task(task_id, client)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEBUG] FATAL: {e}", flush=True)
        import traceback; traceback.print_exc()
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            log_start(task_id, BENCHMARK, MODEL_NAME)
            log_end(False, 0, 0.001, [])
    finally:
        sys.exit(0)
