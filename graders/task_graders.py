"""
Graders for ICU Resource Allocation OpenEnv.
Each grader runs a full 48-step episode and returns a score in (0.0, 1.0) EXCLUSIVE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import ICUEnv

_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _strict(score: float) -> float:
    return round(min(_SCORE_MAX, max(_SCORE_MIN, float(score))), 4)


def _make_llm_agent():
    """
    Build an LLM agent using the platform's proxy.
    Uses API_KEY (the proxy-tracked credential) with API_BASE_URL exactly as provided.
    Falls back to HF_TOKEN if API_KEY is not set.
    """
    try:
        from openai import OpenAI

        api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        api_key      = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
        model        = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

        if not api_key:
            print("[GRADER] No API_KEY or HF_TOKEN found, using rule-based agent", flush=True)
            return _make_rule_based_agent()

        client = OpenAI(base_url=api_base_url, api_key=api_key)

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

        def _obs_to_prompt(obs):
            shift_names = ["Day", "Evening", "Night"]
            shift = shift_names[int(obs.get("shift", 0))]
            return "\n".join([
                "Step " + str(obs.get("step", 0)) + "/48  shift=" + shift,
                "Beds " + str(obs.get("beds_occupied", 0)) + "/20 | free=" + str(obs.get("beds_available", 0)),
                "Queue " + str(obs.get("queue_total", 0)) + ": CRITICAL=" + str(obs.get("queue_critical", 0)),
                "Nurses ratio=" + str(obs.get("nurse_patient_ratio", 1.2)) + "/2.0",
                "Budget Rs" + str(int(obs.get("budget_remaining_inr", 150000))),
                "Action? Reply ONE digit 0-6.",
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

        def agent(obs):
            try:
                resp = client.chat.completions.create(
                    model=model,
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
                        return a
                return _fallback(obs)
            except Exception as e:
                print("[GRADER] LLM error: " + str(e), flush=True)
                return _fallback(obs)

        return agent

    except Exception as e:
        print("[GRADER] Could not build LLM agent: " + str(e), flush=True)
        return _make_rule_based_agent()


def _run_episode(agent_fn, seed: int = 42) -> dict:
    env = ICUEnv(seed=seed)
    obs = env.reset()

    nurse_ratio_steps = []
    sofa_trajectory   = []
    total_reward      = 0.0

    done = False
    while not done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        nurse_ratio_steps.append(obs["nurse_patient_ratio"])
        sofa_trajectory.append(obs["avg_icu_sofa"])

    ratio_breach_fraction = sum(1 for r in nurse_ratio_steps if r > 2.0) / len(nurse_ratio_steps)
    sofa_trend            = sofa_trajectory[-1] - sofa_trajectory[0]
    final_budget_used     = obs["budget_utilisation_pct"] / 100.0

    return {
        "deaths_in_queue":       obs["deaths_in_queue"],
        "adverse_events":        obs["adverse_events"],
        "wait_violations":       obs["wait_violations"],
        "ratio_breach_fraction": ratio_breach_fraction,
        "sofa_trend":            sofa_trend,
        "final_budget_used_pct": final_budget_used,
        "total_reward":          total_reward,
    }


def grade_task_easy(agent_fn=None, seed: int = 42) -> float:
    if agent_fn is None:
        agent_fn = _make_llm_agent()
    m = _run_episode(agent_fn, seed)
    death_penalty = min(0.999, m["deaths_in_queue"] * 0.25)
    ratio_score   = 1.0 - m["ratio_breach_fraction"]
    raw    = 0.60 * (1.0 - death_penalty) + 0.40 * ratio_score
    scaled = raw * 0.92 + 0.04
    return _strict(scaled)


def grade_task_medium(agent_fn=None, seed: int = 42) -> float:
    if agent_fn is None:
        agent_fn = _make_llm_agent()
    m = _run_episode(agent_fn, seed)
    death_score = max(0.0, 1.0 - m["deaths_in_queue"] * 0.30)
    ratio_score = 1.0 - m["ratio_breach_fraction"]
    wait_score  = max(0.0, 1.0 - m["wait_violations"] * 0.15)
    raw    = 0.40 * death_score + 0.30 * ratio_score + 0.30 * wait_score
    scaled = raw * 0.92 + 0.04
    return _strict(scaled)


def grade_task_hard(agent_fn=None, seed: int = 42) -> float:
    if agent_fn is None:
        agent_fn = _make_llm_agent()
    m = _run_episode(agent_fn, seed)
    death_score   = max(0.0, 1.0 - m["deaths_in_queue"] * 0.35)
    adverse_score = max(0.0, 1.0 - m["adverse_events"] * 0.10)
    wait_score    = max(0.0, 1.0 - m["wait_violations"] * 0.12)
    bu = m["final_budget_used_pct"]
    budget_score  = 1.0 if bu <= 0.85 else max(0.0, 1.0 - (bu - 0.85) * 4)
    sofa_score    = 1.0 if m["sofa_trend"] <= 0 else max(0.0, 1.0 - m["sofa_trend"] / 5.0)
    raw    = (0.30 * death_score + 0.20 * adverse_score + 0.20 * wait_score
              + 0.15 * budget_score + 0.15 * sofa_score)
    scaled = raw * 0.92 + 0.04
    return _strict(scaled)


def _make_random_agent(seed: int = 7):
    import random
    rng = random.Random(seed)
    def agent(obs): return rng.randint(0, 6)
    return agent


def _make_rule_based_agent():
    def agent(obs):
        if obs["queue_critical"] > 0 and obs["beds_available"] > 0:
            return 1
        if obs["nurse_patient_ratio"] > 2.2 and obs["beds_occupied"] > 12:
            return 4
        if obs["queue_total"] > 0 and obs["beds_available"] == 0:
            return 3
        if obs["queue_total"] > 0 and obs["beds_available"] > 0:
            return 2
        return 0
    return agent


if __name__ == "__main__":
    print("=" * 65)
    print("ICU Resource Allocation — Grader Evaluation")
    print("=" * 65)
    for label, agent in [
        ("Rule-based", _make_rule_based_agent()),
        ("Random",     _make_random_agent()),
    ]:
        e = grade_task_easy(agent)
        m = grade_task_medium(agent)
        h = grade_task_hard(agent)
        print(f"\n{label}: easy={e:.4f}  medium={m:.4f}  hard={h:.4f}")
        print(f"  All strictly in (0,1): {all(0 < s < 1 for s in [e,m,h])}")
