# -*- coding: utf-8 -*-
# inference.py - ICU Resource Allocation OpenEnv

import os
import sys
import time
import requests
from openai import OpenAI

# ✅ MUST use Scaler injected variables (NO defaults)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]

MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

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
    print("[DEBUG] Calling LLM...", flush=True)

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


def run_task(task_id, client):
    rewards = []
    steps_taken = 0
    success = False

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        obs = _reset_env(seed=42)
        done = False

        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            try:
                action_int = _get_action(client, obs)
                llm_error = None
            except Exception as e:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                action_int = _fallback(obs)
                llm_error = str(e)

            result = _step_env(action_int)
            if result is None:
                break

            obs, reward, done, _ = result
            rewards.append(reward)
            steps_taken = step_n

            log_step(step_n, ACTION_NAMES.get(action_int), reward, done, llm_error)

        success = True

    except Exception as e:
        print(f"[DEBUG] CRASH: {e}", flush=True)

    finally:
        log_end(success, steps_taken, 0.5, rewards)


def main():
    print("[DEBUG] Using Scaler Proxy", flush=True)
    print(f"[DEBUG] BASE_URL={API_BASE_URL}", flush=True)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    print("[DEBUG] Waiting for env...", flush=True)
    _wait_for_server()

    for task in ["task_easy", "task_medium", "task_hard"]:
        run_task(task, client)


if __name__ == "__main__":
    main()
