# -*- coding: utf-8 -*-
# inference.py - ICU Resource Allocation OpenEnv

import os
import sys
import time
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
API_KEY      = os.getenv("API_KEY") 

ENV_BASE_URL      = "http://localhost:7860"
BENCHMARK         = "icu-resource-allocation"
MAX_STEPS         = 48
SUCCESS_THRESHOLD = 0.40
REQUEST_TIMEOUT   = 30

ACTION_NAMES = {
    0: "HOLD", 1: "ADMIT_CRITICAL", 2: "ADMIT_FIFO",
    3: "TRANSFER_OUT", 4: "CALL_EXTRA_NURSE",
    5: "SPECIALIST_CONSULT", 6: "EXPEDITE_BED",
}

SYSTEM_PROMPT = (
    "You are an ICU charge coordinator. Reply with ONE digit 0-6 only. "
    "0=HOLD 1=ADMIT_CRITICAL 2=ADMIT_FIFO 3=TRANSFER_OUT "
    "4=CALL_EXTRA_NURSE 5=SPECIALIST_CONSULT 6=EXPEDITE_BED."
)


def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
    r = requests.post(ENV_BASE_URL + "/reset", json={"seed": seed}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def _step_env(action):
    r = requests.post(ENV_BASE_URL + "/step", json={"action": action}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["observation"], float(data["reward"]), bool(data["done"]), data.get("info", {})

def _obs_to_prompt(obs):
    return (
        f"Beds {obs.get('beds_occupied',0)}/20 free={obs.get('beds_available',0)} "
        f"Queue CRITICAL={obs.get('queue_critical',0)} "
        f"ratio={obs.get('nurse_patient_ratio',1.2)} "
        f"Step {obs.get('step',0)}/48. Reply ONE digit 0-6."
    )

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
    # NO try/except — let failures be visible in logs
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
    print(f"[DEBUG] LLM raw={raw!r}", flush=True)
    if raw and raw[0].isdigit():
        a = int(raw[0])
        if 0 <= a <= 6:
            return a
    return _fallback(obs)

def _score(task_id, m):
    try:
        if task_id == "task_easy":
            raw = 0.60*(1.0-min(0.999,m["deaths"]*0.25)) + 0.40*(1.0-m["ratio_breach_frac"])
        elif task_id == "task_medium":
            raw = (0.40*max(0.0,1.0-m["deaths"]*0.30) +
                   0.30*(1.0-m["ratio_breach_frac"]) +
                   0.30*max(0.0,1.0-m["wait_violations"]*0.15))
        elif task_id == "task_hard":
            bu = m["budget_used_pct"]
            bs = 1.0 if bu<=0.85 else max(0.0,1.0-(bu-0.85)*4)
            ss = 1.0 if m["sofa_trend"]<=0 else max(0.0,1.0-m["sofa_trend"]/5.0)
            raw = (0.30*max(0.0,1.0-m["deaths"]*0.35)+0.20*max(0.0,1.0-m["adverse"]*0.10)+
                   0.20*max(0.0,1.0-m["wait_violations"]*0.12)+0.15*bs+0.15*ss)
        else:
            raw = 0.0
        return round(min(0.999, max(0.001, raw*0.92+0.04)), 3)
    except Exception:
        return 0.001

def run_task(task_id, client):
    rewards, ratio_breaches, sofa_traj = [], [], []
    steps_taken, score, success, obs = 0, 0.001, False, {}
    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)
    try:
        obs = _reset_env(seed=42)
        done = False
        for step_n in range(1, MAX_STEPS + 1):
            if done:
                break
            action_int = _get_action(client, obs)
            action_str = ACTION_NAMES.get(action_int, str(action_int))
            obs, reward, done, _ = _step_env(action_int)
            rewards.append(reward)
            ratio_breaches.append(float(obs.get("nurse_patient_ratio", 1.0)) > 2.0)
            sofa_traj.append(float(obs.get("avg_icu_sofa", 0.0)))
            steps_taken = step_n
            log_step(step_n, action_str, reward, done, None)
        if steps_taken > 0:
            n = max(1, len(ratio_breaches))
            metrics = {
                "deaths":            int(obs.get("deaths_in_queue", 0)),
                "adverse":           int(obs.get("adverse_events", 0)),
                "wait_violations":   int(obs.get("wait_violations", 0)),
                "ratio_breach_frac": sum(ratio_breaches) / n,
                "budget_used_pct":   float(obs.get("budget_utilisation_pct", 0)) / 100.0,
                "sofa_trend":        (sofa_traj[-1]-sofa_traj[0]) if len(sofa_traj)>=2 else 0.0,
            }
            score = _score(task_id, metrics)
            success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] CRASH in {task_id}: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    # Print ALL environment variables related to API so we can diagnose
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] API_KEY set={bool(API_KEY)}", flush=True)
    print(f"[DEBUG] All env keys with API: {[k for k in os.environ if 'API' in k.upper()]}", flush=True)
    print(f"[DEBUG] All env keys with TOKEN: {[k for k in os.environ if 'TOKEN' in k.upper()]}", flush=True)
    print(f"[DEBUG] All env keys with KEY: {[k for k in os.environ if 'KEY' in k.upper()]}", flush=True)

    if not API_KEY:
        print("ERROR: API_KEY not set", flush=True)
        sys.exit(1)

    print("[DEBUG] Waiting for env server...", flush=True)
    if not _wait_for_server(max_wait=60):
        print("[DEBUG] Server not ready, continuing anyway", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[DEBUG] client created base_url={client.base_url}", flush=True)

    # Test ONE LLM call before running tasks
    print("[DEBUG] Testing LLM call...", flush=True)
    test_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with digit 1 only."}],
        max_tokens=5,
    )
    print(f"[DEBUG] LLM test response: {test_resp.choices[0].message.content!r}", flush=True)

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
            log_end(success=False, steps=0, score=0.001, rewards=[])
    finally:
        sys.exit(0)
