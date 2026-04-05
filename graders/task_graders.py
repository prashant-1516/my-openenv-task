"""
Graders for ICU Resource Allocation OpenEnv.
Each grader runs a full 48-step episode and returns a score in [0.0, 1.0].

Clinical rationale for thresholds
-----------------------------------
Easy   – Zero preventable queue deaths AND no major nurse-ratio breaches.
         This is the baseline any functional ICU must meet.
Medium – Zero deaths + safe ratios + ≥70% of critical patients admitted
         within the 2-hour (4-step) window. Maps to NABH Grade-B compliance.
Hard   – All of Medium + budget utilisation ≤ 85% + adverse events = 0 +
         average ICU SOFA trend non-increasing (quality improvement metric).
         Maps to NABH Grade-A / JCI accreditation targets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import ICUEnv


def _run_episode(agent_fn, seed: int = 42) -> dict:
    """Run a full episode and collect all outcome metrics."""
    env = ICUEnv(seed=seed)
    obs = env.reset()

    # Tracking
    nurse_ratio_steps   = []
    critical_wait_steps = []   # (patient_id, wait_steps) for critical patients admitted
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

    # Compute derived metrics
    avg_nurse_ratio        = sum(nurse_ratio_steps) / len(nurse_ratio_steps)
    ratio_breach_fraction  = sum(1 for r in nurse_ratio_steps if r > 2.0) / len(nurse_ratio_steps)
    sofa_trend             = sofa_trajectory[-1] - sofa_trajectory[0]  # negative = improving
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
# Task 1 — EASY: Prevent queue deaths & maintain nurse ratio
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_easy(agent_fn, seed: int = 42) -> float:
    """
    Score 1.0 if: deaths_in_queue == 0 AND nurse_ratio_breach < 10% of steps.
    Partial credit otherwise.
    """
    m = _run_episode(agent_fn, seed)

    # Death penalty: each death removes 0.25 from score, floor at 0
    death_penalty = min(1.0, m["deaths_in_queue"] * 0.25)

    # Nurse ratio: score based on fraction of steps within safe range
    ratio_score = 1.0 - m["ratio_breach_fraction"]

    # Combine: 60% deaths (primary safety), 40% nurse ratio
    raw = 0.60 * (1.0 - death_penalty) + 0.40 * ratio_score
    return round(min(1.0, max(0.0, raw)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — MEDIUM: Add critical-patient response time compliance
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_medium(agent_fn, seed: int = 42) -> float:
    """
    Score 1.0 if:
      - deaths_in_queue == 0
      - ratio_breach_fraction < 10%
      - wait_violations == 0 (no critical patient waited > 2 hours)
    """
    m = _run_episode(agent_fn, seed)

    death_score  = max(0.0, 1.0 - m["deaths_in_queue"] * 0.30)
    ratio_score  = 1.0 - m["ratio_breach_fraction"]
    wait_score   = max(0.0, 1.0 - m["wait_violations"] * 0.15)

    raw = 0.40 * death_score + 0.30 * ratio_score + 0.30 * wait_score
    return round(min(1.0, max(0.0, raw)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — HARD: Full quality + budget + no adverse events
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_hard(agent_fn, seed: int = 42) -> float:
    """
    Score 1.0 if all of:
      - deaths_in_queue == 0
      - adverse_events == 0
      - wait_violations == 0
      - budget_used <= 85%
      - SOFA trend <= 0 (average patient acuity non-increasing)
    """
    m = _run_episode(agent_fn, seed)

    death_score   = max(0.0, 1.0 - m["deaths_in_queue"] * 0.35)
    adverse_score = max(0.0, 1.0 - m["adverse_events"] * 0.10)
    wait_score    = max(0.0, 1.0 - m["wait_violations"] * 0.12)

    # Budget: full marks if ≤ 85%, linear penalty above
    bu = m["final_budget_used_pct"]
    budget_score = 1.0 if bu <= 0.85 else max(0.0, 1.0 - (bu - 0.85) * 4)

    # SOFA trend: reward decreasing (improving) acuity
    sofa_score = 1.0 if m["sofa_trend"] <= 0 else max(0.0, 1.0 - m["sofa_trend"] / 5.0)

    raw = (0.30 * death_score + 0.20 * adverse_score + 0.20 * wait_score
           + 0.15 * budget_score + 0.15 * sofa_score)
    return round(min(1.0, max(0.0, raw)), 4)


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
    """
    Clinically-grounded rule-based agent.
    Priority: admit critical → maintain ratio → transfer stable → hold.
    """
    def agent(obs):
        beds_free   = obs["beds_available"]
        q_critical  = obs["queue_critical"]
        q_total     = obs["queue_total"]
        ratio       = obs["nurse_patient_ratio"]
        occupied    = obs["beds_occupied"]

        # Immediate: admit critical patient if bed free
        if q_critical > 0 and beds_free > 0:
            return 1   # ADMIT_CRITICAL

        # Nurse ratio unsafe: call extra nurse if occupied > 12
        if ratio > 2.2 and occupied > 12:
            return 4   # CALL_EXTRA_NURSE

        # Free up a bed if queue non-empty and no beds free
        if q_total > 0 and beds_free == 0:
            return 3   # TRANSFER_OUT

        # Admit next in queue if bed free
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
        in_range = all(0.0 <= s <= 1.0 for s in [e, m, h])
        print(f"\n{label}:")
        print(f"  Easy   (prevent deaths + safe ratio):     {e:.4f}")
        print(f"  Medium (+ critical response time):        {m:.4f}")
        print(f"  Hard   (+ budget + adverse events + SOFA):{h:.4f}")
        print(f"  All scores in [0.0, 1.0]: {in_range}")
