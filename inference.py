# -*- coding: utf-8 -*-
# inference.py - ICU Resource Allocation OpenEnv

import os
import sys
import time
import requests
from openai import OpenAI

# ✅ SAFE env handling (no crash)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY      = os.getenv("API_KEY")

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

ENV_BASE_URL = "http://localhost:7860"
MAX_STEPS = 48

ACTION_NAMES = {
    0: "HOLD", 1: "ADMIT_CRITICAL", 2: "ADMIT_FIFO",
    3: "TRANSFER_OUT", 4: "CALL_EXTRA_NURSE",
    5: "SPECIALIST_CONSULT", 6: "EXPEDITE_BED",
}

SYSTEM_PROMPT = (
    "You are an ICU charge coordinator at a 500-bed Indian hospital. "
    "Reply ONLY with one digit (0-6)."
)

# ─────────────────────────────────────────

def _wait_for_server():
    for _ in range(60):
        try:
            if requests.get(ENV_BASE_URL).status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def _reset():
    try:
        return requests.post(ENV_BASE_URL + "/reset", json={"seed": 42}).json()
    except:
        return {}

def _step(a):
    try:
        r = requests.post(ENV_BASE_URL + "/step", json={"action": a})
        d = r.json()
        return d["observation"], d["reward"], d["done"]
    except:
        return None, 0, True

def _fallback(obs):
    if obs.get("queue_critical", 0) > 0 and obs.get("beds_available", 0) > 0:
        return 1
    if obs.get("queue_total", 0) > 0:
        return 2
    return 0

def _get_action(client, obs):
    try:
        print("[DEBUG] calling LLM...", flush=True)

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(obs)},
            ],
            max_tokens=5,
            temperature=0.0,
        )

        raw = (resp.choices[0].message.content or "").strip()
        print("[DEBUG] LLM:", raw, flush=True)

        if raw and raw[0].isdigit():
            return int(raw[0])

    except Exception as e:
        print("[DEBUG] LLM error:", e, flush=True)

    return _fallback(obs)

# ─────────────────────────────────────────

def main():
    print("[DEBUG] starting inference...", flush=True)

    # ✅ check env safely
    if not API_BASE_URL or not API_KEY:
        print("[ERROR] Missing API_BASE_URL or API_KEY", flush=True)
        sys.exit(1)

    # ✅ safe client creation
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
    except Exception as e:
        print("[ERROR] client init failed:", e, flush=True)
        sys.exit(1)

    print("[DEBUG] proxy:", API_BASE_URL, flush=True)

    _wait_for_server()
    obs = _reset()

    for i in range(MAX_STEPS):
        action = _get_action(client, obs)
        obs, reward, done = _step(action)

        print(f"[STEP] {i} action={action} reward={reward}", flush=True)

        if done:
            break

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
