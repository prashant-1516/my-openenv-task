"""
Graders for ICU Resource Allocation OpenEnv.
Each grader runs a full 48-step episode and returns a score in (0.0, 1.0) EXCLUSIVE.

IMPORTANT: scores must be STRICTLY between 0 and 1 — not 0.0, not 1.0.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import ICUEnv


# Strictly clamp: ensures score is NEVER exactly 0.0 or 1.0
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _strict(score: float) -> float:
    """Clamp score to be strictly between 0 and 1 (exclusive), rounded to 4dp."""
    return round(min(_SCORE_MAX, max(_SCORE_MIN, float(score))), 4)


def _run_episode(agent_fn, seed: int = 42) -> dict:
    """Run a full episode and collect all outcome metrics."""
    env = ICUEnv(seed=seed)
    obs = env.reset()

    nurse_ratio_steps   = []
    sofa_trajectory     = []
    budget_used_pcts    = []
    total_reward        = 0.0
    step_infos          = []

    done = False
    while not done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        nurse_ratio_steps.append(obs["nurse_patient_ratio"])
        sofa_trajectory.append(obs["avg_icu_sofa"])
        budget_used_pcts.append(obs["budget_utilisation_pct"])
        step_infos.append(info)

    avg_nurse_ratio        = sum(nurse_ratio_steps) / len(nurse_ratio_steps)
    ratio_breach_fraction  = sum(1 for r in nurse_ratio_steps if r > 2.0) / len(nurse_ratio_steps)
    sofa_trend             = sofa_trajectory[-1] - sofa_trajectory[0]
    final_budget_used      = obs["budget_utilisation_pct"] / 100.0

    return {
        "deaths_in_queue":        obs["deaths_in_queue"],
        "adverse_events":         obs["adverse_events"],
        "wait_violations":        obs["wait_violations"],
        "avg_nurse_ratio":        avg_nurse_ratio,
        "ratio_breach_fraction":  ratio_breach_fraction,
        "sofa_trend":             sofa_trend,
        "final_budget_used_pct":  final_budget_used,
        "admissions_today":       obs["admissions_today"],
        "transfers_today":        obs["transfers_today"],
        "total_reward":           total_reward,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — EASY
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_easy(agent_fn, seed: int = 42) -> float:
    """
    Score strictly in (0, 1).
    Higher score for: zero deaths AND low nurse-ratio breach fraction.
    Uses linear scaling to guarantee strict (0, 1) exclusivity.
    """
    m = _run_episode(agent_fn, seed)

    death_penalty = min(0.999, m["deaths_in_queue"] * 0.25)
    ratio_score   = 1.0 - m["ratio_breach_fraction"]

    raw    = 0.60 * (1.0 - death_penalty) + 0.40 * ratio_score
    # Map [0,1] -> (0.04, 0.96) so perfect and worst scores are never 0 or 1
    scaled = raw * 0.92 + 0.04
    return _strict(scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — MEDIUM
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_medium(agent_fn, seed: int = 42) -> float:
    """
    Score strictly in (0, 1).
    Higher for: zero deaths + safe ratios + zero wait violations.
    """
    m = _run_episode(agent_fn, seed)

    death_score = max(0.0, 1.0 - m["deaths_in_queue"] * 0.30)
    ratio_score = 1.0 - m["ratio_breach_fraction"]
    wait_score  = max(0.0, 1.0 - m["wait_violations"] * 0.15)

    raw    = 0.40 * death_score + 0.30 * ratio_score + 0.30 * wait_score
    scaled = raw * 0.92 + 0.04
    return _strict(scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — HARD
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_hard(agent_fn, seed: int = 42) -> float:
    """
    Score strictly in (0, 1).
    Higher for: zero deaths, zero adverse events, zero wait violations,
    budget <= 85%, SOFA trend <= 0.
    """
    m = _run_episode(agent_fn, seed)

    death_score   = max(0.0, 1.0 - m["deaths_in_queue"] * 0.35)
    adverse_score = max(0.0, 1.0 - m["adverse_events"] * 0.10)
    wait_score    = max(0.0, 1.0 - m["wait_violations"] * 0.12)

    bu = m["final_budget_used_pct"]
    budget_score = 1.0 if bu <= 0.85 else max(0.0, 1.0 - (bu - 0.85) * 4)

    sofa_score = 1.0 if m["sofa_trend"] <= 0 else max(0.0, 1.0 - m["sofa_trend"] / 5.0)

    raw    = (0.30 * death_score + 0.20 * adverse_score + 0.20 * wait_score
              + 0.15 * budget_score + 0.15 * sofa_score)
    scaled = raw * 0.92 + 0.04
    return _strict(scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline agents for validation
# ─────────────────────────────────────────────────────────────────────────────

def _make_random_agent(seed: int = 7):
    import random
    rng = random.Random(seed)
    def agent(obs):
        return rng.randint(0, 6)
    return agent


def _make_rule_based_agent():
    """Clinically-grounded rule-based agent."""
    def agent(obs):
        beds_free  = obs["beds_available"]
        q_critical = obs["queue_critical"]
        q_total    = obs["queue_total"]
        ratio      = obs["nurse_patient_ratio"]
        occupied   = obs["beds_occupied"]

        if q_critical > 0 and beds_free > 0:
            return 1   # ADMIT_CRITICAL
        if ratio > 2.2 and occupied > 12:
            return 4   # CALL_EXTRA_NURSE
        if q_total > 0 and beds_free == 0:
            return 3   # TRANSFER_OUT
        if q_total > 0 and beds_free > 0:
            return 2   # ADMIT_FIFO
        return 0   # HOLD
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for hackathon validator
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("ICU Resource Allocation — Grader Evaluation")
    print("=" * 65)

    for label, agent in [
        ("Rule-based agent", _make_rule_based_agent()),
        ("Random agent",     _make_random_agent()),
    ]:
        e = grade_task_easy(agent)
        m = grade_task_medium(agent)
        h = grade_task_hard(agent)
        strictly_ok = all(0.0 < s < 1.0 for s in [e, m, h])
        print(f"\n{label}:")
        print(f"  Easy   (prevent deaths + safe ratio):     {e:.4f}")
        print(f"  Medium (+ critical response time):        {m:.4f}")
        print(f"  Hard   (+ budget + adverse events + SOFA):{h:.4f}")
        print(f"  All scores strictly in (0.0, 1.0): {strictly_ok}")
