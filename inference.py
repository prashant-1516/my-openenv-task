# -*- coding: utf-8 -*-
# inference.py - ICU Resource Allocation OpenEnv

import os
import sys
import time
import requests
from openai import OpenAI

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

ENV_BASE_URL      = "http://localhost:7860"
BENCHMARK         = "icu-resource-allocation"
MAX_STEPS         = 48
SUCCESS_THRESHOLD = 0.40
REQUEST_TIMEOUT   = 30

ACTION_NAMES = {
    0: "HOLD",
    1: "ADMIT_CRITICAL",
    2: "ADMIT_FIFO",
    3: "TRANSFER_OUT",
    4: "CALL_EXTRA_NURSE",
    5: "SPECIALIST_CONSULT",
    6: "EXPEDITE_BED",
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
    print("[START] task=" + task + " env=" + env_name + " model=" + model, flush=True)

def log_step(step, action, reward, done, error):
    print(
        "[STEP] step=" + str(step) +
        " action=" + str(action) +
        " reward=" + "{:.2f}".format(reward) +
        " done=" + ("true" if done else "false") +
        " error=" + (str(error) if error else "null"),
        flush=True,
    )

def log_end(success, steps, score, rewards):
    print(
        "[END] success=" + ("true" if success else "false") +
        " steps=" + str(steps) +
        " score=" + "{:.3f}".format(score) +
        " rewards=" + ",".join("{:.2f}".format(r) for r in rewards),
        flush=True,
    )


def _wait_for_server(max_wait=60):
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
        print("[DEBUG] reset failed: " + str(e), flush=True)
        return {}

def _step_env(action):
    try:
        r = requests.post(ENV_BASE_URL + "/step", json={"action": action}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["observation"], float(data["reward"]), bool(data["done"]), data.get("info", {})
    except Exception as e:
        print("[DEBUG] step failed: " + str(e), flush=True)
        return None


def _obs_to_prompt(obs):
    shift_names = ["Day", "Evening", "Night"]
    shift = shift_names[int(obs.get("shift", 0))]
    lines = [
        "Step " + str(obs.get("step", 0)) + "/48  " + str(obs.get("time_of_day", 8)) + ":00  " + shift,
        "Beds " + str(obs.get("beds_occupied", 0)) + "/20 | free=" + str(obs.get("beds_available", 0)) + " | turnover=" + str(obs.get("beds_in_turnover", 0)),
        "Queue " + str(obs.get("queue_total", 0)) + ": CRITICAL=" + str(obs.get("queue_critical", 0)) + " SEVERE=" + str(obs.get("queue_severe", 0)) + " MOD=" + str(obs.get("queue_moderate", 0)),
        "Longest wait: " + str(obs.get("queue_max_wait_steps", 0)) + " steps",
        "ICU avg_SOFA=" + str(obs.get("avg_icu_sofa", 0)) + " avg_mortality=" + str(obs.get("avg_icu_mortality_risk", 0)),
        "Nurses " + str(obs.get("nurses_on_duty", 10)) + " ratio=" + str(obs.get("nurse_patient_ratio", 1.2)) + "/2.0",
        "Budget Rs" + str(int(obs.get("budget_remaining_inr", 150000))) + " remaining",
        "deaths=" + str(obs.get("deaths_in_queue", 0)) + " adverse=" + str(obs.get("adverse_events", 0)) + " wait_violations=" + str(obs.get("wait_violations", 0)),
        "Action? Reply with ONE digit 0-6.",
    ]
    return "\n".join(lines)

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
    """Call LLM through the proxied client. Never silently swallow errors."""
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
    print("[DEBUG] LLM raw=" + raw, flush=True)
    if raw and raw[0].isdigit():
        a = int(raw[0])
        if 0 <= a <= 6:
            return a
    return _fallback(obs)


def _score(task_id, m):
    try:
        if task_id == "task_easy":
            death_penalty = min(0.999, m["deaths"] * 0.25)
            ratio_score   = 1.0 - m["ratio_breach_frac"]
            raw = 0.60 * (1.0 - death_penalty) + 0.40 * ratio_score
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
        scaled = raw * 0.92 + 0.04
        return round(min(0.999, max(0.001, scaled)), 3)
    except Exception:
        return 0.001


def run_task(task_id, client):
    rewards = []
    ratio_breaches = []
    sofa_traj = []
    steps_taken = 0
    score = 0.001
    success = False
    obs = {}

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        obs = _reset_env(seed=42)
        done = False

        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            action_int = _get_action(client, obs)
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

            log_step(step_n, action_str, reward, done, None)

        if steps_taken > 0:
            n = max(1, len(ratio_breaches))
            sofa_trend = (sofa_traj[-1] - sofa_traj[0]) if len(sofa_traj) >= 2 else 0.0
            metrics = {
                "deaths":            int(obs.get("deaths_in_queue", 0)),
                "adverse":           int(obs.get("adverse_events", 0)),
                "wait_violations":   int(obs.get("wait_violations", 0)),
                "ratio_breach_frac": sum(ratio_breaches) / n,
                "budget_used_pct":   float(obs.get("budget_utilisation_pct", 0)) / 100.0,
                "sofa_trend":        sofa_trend,
            }
            score   = _score(task_id, metrics)
            success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print("[DEBUG] task=" + task_id + " CRASHED: " + str(e), flush=True)
        import traceback
        traceback.print_exc()

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    # ------------------------------------------------------------------ #
    # Use the platform-injected env vars EXACTLY as provided.             #
    # API_BASE_URL must include /v1 so the OpenAI SDK routes correctly.   #
    # We only normalise the path — we never use a different host/key.     #
    # ------------------------------------------------------------------ #
    api_base_url = os.environ["API_BASE_URL"]
    api_key      = os.environ["API_KEY"]

    # The OpenAI SDK appends /chat/completions to whatever base_url you
    # give it.  LiteLLM expects requests at /v1/chat/completions.
    # So base_url MUST end with /v1 (the SDK adds a trailing slash itself).
    # If the platform already includes /v1 we leave it alone; otherwise we add it.
    stripped = api_base_url.rstrip("/")
    if not stripped.endswith("/v1"):
        stripped = stripped + "/v1"
    api_base_url = stripped   # SDK will normalise the trailing slash

    print("[DEBUG] API_BASE_URL (normalised)=" + api_base_url, flush=True)
    print("[DEBUG] API_KEY_LEN=" + str(len(api_key)), flush=True)
    print("[DEBUG] MODEL_NAME=" + MODEL_NAME, flush=True)

    print("[DEBUG] Waiting for env server...", flush=True)
    if not _wait_for_server(max_wait=60):
        print("[DEBUG] Server not ready, continuing anyway", flush=True)

    # Single client — every LLM call in this process goes through this one
    # client which is pointed at the platform's LiteLLM proxy.
    client = OpenAI(base_url=api_base_url, api_key=api_key)
    print("[DEBUG] OpenAI client ready, base_url=" + str(client.base_url), flush=True)

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        run_task(task_id, client)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[DEBUG] FATAL: " + str(e), flush=True)
        import traceback
        traceback.print_exc()
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            log_start(task_id, BENCHMARK, MODEL_NAME)
            log_end(success=False, steps=0, score=0.001, rewards=[])
    finally:
        sys.exit(0)
