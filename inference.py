# -*- coding: utf-8 -*-
# inference.py - ICU Resource Allocation OpenEnv
# Calls the running FastAPI server at localhost:7860 via HTTP.
#
# Required stdout format:
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

import os
import sys
import time
import requests
from typing import List, Optional

# Config
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

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


# Stdout log helpers

def log_start(task, env_name, model):
    print("[START] task=" + task + " env=" + env_name + " model=" + model, flush=True)


def log_step(step, action, reward, done, error):
    done_str  = "true" if done else "false"
    error_str = str(error) if error else "null"
    print(
        "[STEP] step=" + str(step) +
        " action=" + str(action) +
        " reward=" + "{:.2f}".format(reward) +
        " done=" + done_str +
        " error=" + error_str,
        flush=True,
    )


def log_end(success, steps, score, rewards):
    success_str  = "true" if success else "false"
    rewards_str  = ",".join("{:.2f}".format(r) for r in rewards)
    print(
        "[END] success=" + success_str +
        " steps=" + str(steps) +
        " score=" + "{:.3f}".format(score) +
        " rewards=" + rewards_str,
        flush=True,
    )


# HTTP helpers

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
        r = requests.post(
            ENV_BASE_URL + "/reset",
            json={"seed": seed},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[DEBUG] reset failed: " + str(e), flush=True)
        return {}


def _step_env(action):
    try:
        r = requests.post(
            ENV_BASE_URL + "/step",
            json={"action": action},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data   = r.json()
        obs    = data["observation"]
        reward = float(data["reward"])
        done   = bool(data["done"])
        info   = data.get("info", {})
        return obs, reward, done, info
    except Exception as e:
        print("[DEBUG] step failed: " + str(e), flush=True)
        return None


# LLM prompt builder

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


# Rule-based fallback agent

def _fallback(obs):
    try:
        if obs.get("queue_critical", 0) > 0 and obs.get("beds_available", 0) > 0:
            return 1
        if obs.get("nurse_patient_ratio", 1.0) > 2.2:
            return 4
        if obs.get("queue_total", 0) > 0 and obs.get("beds_available", 0) == 0:
            return 3
        if obs.get("queue_total", 0) > 0 and obs.get("beds_available", 0) > 0:
            return 2
    except Exception:
        pass
    return 0


def _get_action(client, obs):
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
        return _fallback(obs), "bad_output:" + raw[:10]
    except Exception as e:
        return _fallback(obs), "llm_err:" + type(e).__name__


# Inline scoring

def _score(task_id, m):
    try:
        if task_id == "task_easy":
            s = (
                0.50 * max(0.0, 1.0 - m["deaths"] * 0.30) +
                0.25 * max(0.0, 1.0 - m["ratio_breach_frac"] * 3.0) +
                0.25 * min(1.0, m["admissions"] / 6.0)
            )
        elif task_id == "task_medium":
            s = (
                0.35 * max(0.0, 1.0 - m["deaths"] * 0.35) +
                0.25 * max(0.0, 1.0 - m["ratio_breach_frac"] * 4.0) +
                0.25 * max(0.0, 1.0 - m["wait_violations"] * 0.12) +
                0.15 * min(1.0, m["admissions"] / 8.0)
            )
        elif task_id == "task_hard":
            bu = m["budget_used_pct"]
            budget_s = 1.0 if bu <= 0.85 else max(0.0, 1.0 - (bu - 0.85) * 4)
            sofa_s   = 1.0 if m["sofa_trend"] <= 0 else max(0.0, 1.0 - m["sofa_trend"] / 5.0)
            s = (
                0.30 * max(0.0, 1.0 - m["deaths"] * 0.35) +
                0.20 * max(0.0, 1.0 - m["adverse"] * 0.10) +
                0.20 * max(0.0, 1.0 - m["wait_violations"] * 0.12) +
                0.15 * budget_s +
                0.15 * sofa_s
            )
        else:
            s = 0.0
        return round(min(1.0, max(0.0, s)), 3)
    except Exception:
        return 0.0


# Episode runner

def run_task(task_id, client):
    rewards         = []
    ratio_breaches  = []
    sofa_traj       = []
    steps_taken     = 0
    score           = 0.0
    success         = False
    obs             = {}

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = _reset_env(seed=42)
        done = False

        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break

            action_int, llm_error = _get_action(client, obs)
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
            sofa_trend = (sofa_traj[-1] - sofa_traj[0]) if len(sofa_traj) >= 2 else 0.0
            metrics = {
                "deaths":            int(obs.get("deaths_in_queue",    0)),
                "adverse":           int(obs.get("adverse_events",     0)),
                "wait_violations":   int(obs.get("wait_violations",    0)),
                "admissions":        int(obs.get("admissions_today",   0)),
                "ratio_breach_frac": sum(ratio_breaches) / n,
                "budget_used_pct":   float(obs.get("budget_utilisation_pct", 0)) / 100.0,
                "sofa_trend":        sofa_trend,
            }
            score   = _score(task_id, metrics)
            success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print("[DEBUG] task=" + task_id + " exception: " + str(e), flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# Main

def main():
    print("[DEBUG] Waiting for server...", flush=True)
    ready = _wait_for_server(max_wait=60)
    if not ready:
        print("[DEBUG] Server not ready, continuing anyway", flush=True)

    client = None
    try:
        from openai import OpenAI
        token  = HF_TOKEN if HF_TOKEN else "dummy"
        client = OpenAI(base_url=API_BASE_URL, api_key=token)
        print("[DEBUG] OpenAI client ready", flush=True)
    except Exception as e:
        print("[DEBUG] OpenAI client failed: " + str(e) + " - using fallback", flush=True)

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            run_task(task_id, client)
        except Exception as e:
            print("[DEBUG] run_task crashed: " + str(e), flush=True)
            print("[END] success=false steps=0 score=0.000 rewards=", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[DEBUG] main crashed: " + str(e), flush=True)
    finally:
        sys.exit(0)
