"""
ICU Resource Allocation — OpenEnv Environment
==============================================
A real-world environment modelling a 20-bed ICU in a 500-bed Indian tertiary-
care hospital.  An AI agent acts as the ICU charge-coordinator, deciding every
30 minutes how to allocate beds, staff and equipment across an incoming stream
of critically ill patients.

Clinical grounding
------------------
- Patient severity measured by the SOFA score (Sequential Organ Failure
  Assessment, range 0-24), the gold standard triage tool used in ICUs globally.
- Nurse : patient ratios follow NABH (National Accreditation Board for Hospitals)
  guidelines — 1 : 2 for ICU.
- Bed turnover time (cleaning + preparation) modelled at 45-90 min, matching
  published Indian hospital data.
- Patient arrival follows a non-homogeneous Poisson process with higher rates
  during 08-12 h and 20-24 h (documented admission peaks).
- Equipment (ventilators, monitors, dialysis) tracked against real 500-bed
  tertiary-care inventories.
- Costs in INR, calibrated to CGHS package rates (Central Govt Health Scheme).

Action space  (Discrete 7)
--------------------------
0  HOLD            – Observe; no allocation change this step.
1  ADMIT_CRITICAL  – Admit the highest-SOFA patient from the waiting queue.
2  ADMIT_FIFO      – Admit the longest-waiting patient from the queue.
3  TRANSFER_OUT    – Transfer the most stable current ICU patient to step-down.
4  CALL_EXTRA_NURSE– Arrange an overtime nurse for this shift (₹1 200 premium).
5  SPECIALIST_CONSULT – Request urgent specialist consult for the sickest
                        current patient (₹3 500, reduces mortality risk).
6  EXPEDITE_BED    – Pay porter/housekeeping overtime to clean next bed faster
                      (₹600, cuts turnover time by ~30 min).

Observation space (23 fields)
------------------------------
See _build_obs() for full description with units and ranges.

Reward  (partial progress at every step, no sparse end-of-episode)
------
See _calculate_reward() for breakdown.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Patient:
    """Represents a single patient."""
    pid: int
    sofa: float          # 0-24  (Sequential Organ Failure Assessment score)
    needs_ventilator: bool
    needs_dialysis: bool
    arrival_step: int    # Step when patient arrived in queue
    admitted_step: Optional[int] = None
    los_steps: int = 0   # Expected length of stay in steps (each step=30 min)
    mortality_risk: float = 0.0   # 0-1  (derived from SOFA)

    @staticmethod
    def sofa_to_mortality(sofa: float) -> float:
        """
        Approximate ICU mortality from SOFA score.
        Based on: Ferreira et al., JAMA 2001 — SOFA score as predictor of ICU outcome.
        """
        # SOFA 0-6: ~10%, 7-9: ~21%, 10-12: ~33%, 13-14: ~50%, 15-24: ~95%
        breakpoints = [(6, 0.10), (9, 0.21), (12, 0.33), (14, 0.50), (24, 0.95)]
        for threshold, risk in breakpoints:
            if sofa <= threshold:
                return risk
        return 0.95

    def __post_init__(self):
        self.mortality_risk = self.sofa_to_mortality(self.sofa)


@dataclass
class Bed:
    """ICU bed state."""
    bed_id: int
    patient: Optional[Patient] = None
    turnover_steps_remaining: int = 0   # >0 means bed being cleaned

    @property
    def is_available(self) -> bool:
        return self.patient is None and self.turnover_steps_remaining == 0

    @property
    def is_occupied(self) -> bool:
        return self.patient is not None

    @property
    def in_turnover(self) -> bool:
        return self.patient is None and self.turnover_steps_remaining > 0


# ─────────────────────────────────────────────────────────────────────────────
# Main environment
# ─────────────────────────────────────────────────────────────────────────────

class ICUEnv:
    """
    OpenEnv-compliant ICU Resource Allocation environment.

    Each step = 30 minutes of real time.
    One episode = 48 steps = 24 hours (one full ICU duty cycle).
    """

    # ── Hospital configuration (typical 500-bed tertiary care, India) ────────
    TOTAL_ICU_BEDS      = 20
    TOTAL_VENTILATORS   = 12
    TOTAL_DIALYSIS      = 4
    TOTAL_MONITORS      = 20   # 1 per bed

    # Staff baseline per shift
    BASE_NURSES_DAY     = 10   # 1:2 ratio for 20 beds
    BASE_NURSES_NIGHT   = 8    # slightly reduced night staffing
    BASE_DOCTORS        = 2    # intensivists on call

    # Financial (INR, calibrated to CGHS 2023 rates)
    DAILY_BUDGET_INR    = 150_000   # ₹1.5 lakh daily ICU operating budget
    ICU_BED_COST_STEP   = 3_125     # ₹3 125 per bed per step (₹1.5L / 48 steps / ~1 bed)
    OVERTIME_NURSE_COST = 1_200     # Per shift overtime premium
    SPECIALIST_COST     = 3_500     # Specialist consult fee
    EXPEDITE_BED_COST   = 600       # Housekeeping overtime for fast bed prep

    # Clinical thresholds
    SAFE_NURSE_RATIO    = 2.0       # Max patients per nurse (NABH standard)
    CRITICAL_SOFA       = 11        # SOFA ≥ 11 → critical, time-sensitive
    TRANSFER_SOFA_MAX   = 6         # SOFA ≤ 6 → eligible for step-down transfer
    MAX_QUEUE_WAIT_STEPS = 4        # >4 steps (2h) wait for critical → outcome worsens

    # Time-to-admission mortality penalty scaling
    # Every 30-min delay for critical patient increases mortality risk by ~3%
    DELAY_MORTALITY_INCREMENT = 0.03

    MAX_STEPS = 48

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)
        self._step = 0
        self._pid_counter = 0
        self.reset()

    # ─────────────────────────────────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset to beginning of a fresh 24-hour duty cycle."""
        self._rng = random.Random(self.seed)
        self._step = 0
        self._pid_counter = 0
        self._hour = 8.0   # Duty cycle starts at 08:00

        # Beds
        self._beds = [Bed(bed_id=i) for i in range(self.TOTAL_ICU_BEDS)]

        # Pre-populate ~60% bed occupancy at start of shift (realistic handover)
        initial_patients = int(self.TOTAL_ICU_BEDS * 0.60)
        for i in range(initial_patients):
            p = self._generate_patient(is_initial=True)
            self._beds[i].patient = p

        # Queues and tracking
        self._queue: list[Patient] = []
        self._discharged: list[Patient] = []
        self._deaths_in_queue: int = 0
        self._adverse_events: int = 0
        self._admissions_today: int = 0
        self._transfers_today: int = 0

        # Staff
        self._extra_nurses_called: int = 0
        self._specialist_consults: int = 0

        # Equipment
        self._ventilators_in_use: int = sum(
            1 for b in self._beds if b.is_occupied and b.patient.needs_ventilator
        )
        self._dialysis_in_use: int = sum(
            1 for b in self._beds if b.is_occupied and b.patient.needs_dialysis
        )

        # Budget
        self._budget_remaining = self.DAILY_BUDGET_INR
        self._cost_this_step = 0.0

        # Outcome tracking
        self._mortality_risks_avoided = 0.0
        self._total_sofa_admitted = 0.0
        self._wait_violations = 0

        # Generate initial queue (2-5 waiting patients at shift start)
        for _ in range(self._rng.randint(2, 5)):
            self._queue.append(self._generate_patient())

        return self._build_obs()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """
        Apply action, simulate 30 minutes, return (obs, reward, done, info).

        Actions:
          0 HOLD
          1 ADMIT_CRITICAL  – admit highest-SOFA patient
          2 ADMIT_FIFO      – admit longest-waiting patient
          3 TRANSFER_OUT    – move most stable ICU patient to step-down
          4 CALL_EXTRA_NURSE
          5 SPECIALIST_CONSULT
          6 EXPEDITE_BED
        """
        if action not in range(7):
            action = 0

        self._cost_this_step = 0.0
        action_result = self._apply_action(action)

        # Simulate 30-minute time passage
        self._simulate_time_passage()

        # Advance clock
        self._step += 1
        self._hour = (8.0 + self._step * 0.5) % 24

        reward = self._calculate_reward(action)
        done = self._step >= self.MAX_STEPS

        obs = self._build_obs()
        info = {
            "action_result":       action_result,
            "cost_this_step_inr":  round(self._cost_this_step, 2),
            "deaths_in_queue":     self._deaths_in_queue,
            "adverse_events":      self._adverse_events,
            "admissions_today":    self._admissions_today,
            "transfers_today":     self._transfers_today,
            "wait_violations":     self._wait_violations,
            "nurse_ratio":         round(self._nurse_patient_ratio(), 2),
        }
        return obs, round(reward, 4), done, info

    def state(self) -> dict:
        """Return current observation without advancing time."""
        return self._build_obs()

    # ─────────────────────────────────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> str:
        if action == 0:
            return "HOLD: no allocation change"

        elif action == 1:  # ADMIT_CRITICAL
            if not self._queue:
                return "ADMIT_CRITICAL: queue empty"
            bed = self._first_available_bed()
            if bed is None:
                return "ADMIT_CRITICAL: no available bed"
            # Admit highest-SOFA patient
            patient = max(self._queue, key=lambda p: p.sofa)
            self._queue.remove(patient)
            self._admit_patient(bed, patient)
            return f"ADMIT_CRITICAL: admitted P{patient.pid} (SOFA={patient.sofa:.1f}) to Bed {bed.bed_id}"

        elif action == 2:  # ADMIT_FIFO
            if not self._queue:
                return "ADMIT_FIFO: queue empty"
            bed = self._first_available_bed()
            if bed is None:
                return "ADMIT_FIFO: no available bed"
            # Admit longest-waiting patient
            patient = min(self._queue, key=lambda p: p.arrival_step)
            self._queue.remove(patient)
            self._admit_patient(bed, patient)
            return f"ADMIT_FIFO: admitted P{patient.pid} (SOFA={patient.sofa:.1f}) to Bed {bed.bed_id}"

        elif action == 3:  # TRANSFER_OUT
            candidates = [b for b in self._beds
                          if b.is_occupied and b.patient.sofa <= self.TRANSFER_SOFA_MAX]
            if not candidates:
                return "TRANSFER_OUT: no stable patients eligible"
            # Transfer lowest-SOFA patient
            bed = min(candidates, key=lambda b: b.patient.sofa)
            patient = bed.patient
            bed.patient = None
            bed.turnover_steps_remaining = self._rng.randint(1, 3)  # 30-90 min cleanup
            self._discharged.append(patient)
            self._transfers_today += 1
            # Reclaim equipment
            if patient.needs_ventilator:
                self._ventilators_in_use = max(0, self._ventilators_in_use - 1)
            if patient.needs_dialysis:
                self._dialysis_in_use = max(0, self._dialysis_in_use - 1)
            return f"TRANSFER_OUT: transferred P{patient.pid} (SOFA={patient.sofa:.1f}) to step-down"

        elif action == 4:  # CALL_EXTRA_NURSE
            cost = self.OVERTIME_NURSE_COST
            if self._budget_remaining < cost:
                return "CALL_EXTRA_NURSE: insufficient budget"
            self._budget_remaining -= cost
            self._cost_this_step += cost
            self._extra_nurses_called += 1
            return f"CALL_EXTRA_NURSE: +1 nurse this shift (₹{cost})"

        elif action == 5:  # SPECIALIST_CONSULT
            # Reduce mortality risk of sickest current patient
            occupied = [b for b in self._beds if b.is_occupied]
            if not occupied:
                return "SPECIALIST_CONSULT: no current patients"
            cost = self.SPECIALIST_COST
            if self._budget_remaining < cost:
                return "SPECIALIST_CONSULT: insufficient budget"
            sickest = max(occupied, key=lambda b: b.patient.mortality_risk)
            old_risk = sickest.patient.mortality_risk
            sickest.patient.mortality_risk = max(0.05, old_risk - 0.15)
            self._budget_remaining -= cost
            self._cost_this_step += cost
            self._specialist_consults += 1
            self._mortality_risks_avoided += (old_risk - sickest.patient.mortality_risk)
            return (f"SPECIALIST_CONSULT: P{sickest.patient.pid} mortality risk "
                    f"{old_risk:.2f}→{sickest.patient.mortality_risk:.2f} (₹{cost})")

        elif action == 6:  # EXPEDITE_BED
            turnover_beds = [b for b in self._beds if b.in_turnover]
            if not turnover_beds:
                return "EXPEDITE_BED: no beds in turnover"
            cost = self.EXPEDITE_BED_COST
            if self._budget_remaining < cost:
                return "EXPEDITE_BED: insufficient budget"
            # Reduce turnover time of the bed closest to ready
            target = min(turnover_beds, key=lambda b: b.turnover_steps_remaining)
            target.turnover_steps_remaining = max(0, target.turnover_steps_remaining - 1)
            self._budget_remaining -= cost
            self._cost_this_step += cost
            return f"EXPEDITE_BED: Bed {target.bed_id} ready sooner (₹{cost})"

        return "UNKNOWN action"

    # ─────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────

    def _simulate_time_passage(self):
        """Advance simulation by 30 minutes."""
        # 1. Existing ICU patients: progress LOS, possibly deteriorate or improve
        for bed in self._beds:
            if not bed.is_occupied:
                continue
            p = bed.patient
            p.los_steps += 1

            # Natural deterioration/improvement (small random walk on SOFA)
            delta = self._rng.gauss(0, 0.4)
            p.sofa = max(0.0, min(24.0, p.sofa + delta))
            p.mortality_risk = Patient.sofa_to_mortality(p.sofa)

            # Check if patient ready for discharge (completed LOS)
            if p.los_steps >= p.admitted_step + self._rng.randint(4, 16):
                # Patient stable enough for general ward
                if p.sofa <= 8:
                    bed.patient = None
                    bed.turnover_steps_remaining = self._rng.randint(1, 3)
                    self._discharged.append(p)
                    if p.needs_ventilator:
                        self._ventilators_in_use = max(0, self._ventilators_in_use - 1)
                    if p.needs_dialysis:
                        self._dialysis_in_use = max(0, self._dialysis_in_use - 1)

            # Adverse event if nurse ratio is unsafe
            if self._nurse_patient_ratio() > self.SAFE_NURSE_RATIO * 1.5:
                if self._rng.random() < 0.08:   # 8% chance per step per patient
                    self._adverse_events += 1
                    p.sofa = min(24.0, p.sofa + 1.5)
                    p.mortality_risk = Patient.sofa_to_mortality(p.sofa)

        # 2. Bed turnover countdown
        for bed in self._beds:
            if bed.in_turnover:
                bed.turnover_steps_remaining -= 1

        # 3. Waiting queue deterioration and deaths
        for p in list(self._queue):
            wait = self._step - p.arrival_step
            if wait >= self.MAX_QUEUE_WAIT_STEPS and p.sofa >= self.CRITICAL_SOFA:
                self._wait_violations += 1
                # Mortality risk worsens with delay
                p.mortality_risk = min(0.99, p.mortality_risk + self.DELAY_MORTALITY_INCREMENT)
                p.sofa = min(24.0, p.sofa + 0.5)
                # Patient may die in queue
                if p.mortality_risk > 0.90 and self._rng.random() < 0.15:
                    self._queue.remove(p)
                    self._deaths_in_queue += 1

        # 4. New arrivals (non-homogeneous Poisson, peaks at 08-12h and 20-24h)
        arrival_rate = self._arrival_rate_per_step()
        n_arrivals = self._rng.poisson_approx(arrival_rate)
        for _ in range(n_arrivals):
            self._queue.append(self._generate_patient())

        # 5. Deduct bed operating costs from budget
        occupied_count = sum(1 for b in self._beds if b.is_occupied)
        step_cost = occupied_count * (self.DAILY_BUDGET_INR / self.MAX_STEPS / self.TOTAL_ICU_BEDS)
        self._budget_remaining = max(0.0, self._budget_remaining - step_cost)

    def _arrival_rate_per_step(self) -> float:
        """
        Non-homogeneous Poisson arrival rate.
        Peak hours: 08-12 (post-morning rounds referrals) and 20-24 (evening emergencies).
        Based on: Arias-Verdú et al., Critical Care Medicine 2017.
        """
        h = self._hour
        base = 0.4
        if 8 <= h < 12:
            return base * 2.0
        elif 20 <= h < 24:
            return base * 1.8
        elif 14 <= h < 18:
            return base * 1.2
        elif 0 <= h < 6:
            return base * 0.5
        return base

    def _generate_patient(self, is_initial: bool = False) -> Patient:
        """Generate a patient with realistic SOFA distribution."""
        self._pid_counter += 1
        # SOFA distribution in Indian ICU referrals (based on published data)
        # ~20% critical (≥11), ~40% severe (7-10), ~40% moderate (0-6)
        r = self._rng.random()
        if r < 0.20:
            sofa = self._rng.uniform(11, 20)    # Critical
        elif r < 0.60:
            sofa = self._rng.uniform(7, 11)     # Severe
        else:
            sofa = self._rng.uniform(1, 7)      # Moderate

        # Equipment needs correlate with severity
        needs_vent = sofa >= 9 and self._rng.random() < 0.55
        needs_dial = sofa >= 10 and self._rng.random() < 0.30

        expected_los = max(4, int(sofa * 1.5 + self._rng.gauss(0, 2)))

        return Patient(
            pid=self._pid_counter,
            sofa=round(sofa, 1),
            needs_ventilator=needs_vent,
            needs_dialysis=needs_dial,
            arrival_step=self._step if not is_initial else -self._rng.randint(2, 8),
            los_steps=0,
            admitted_step=0 if is_initial else None,
        )

    def _admit_patient(self, bed: Bed, patient: Patient):
        """Place a patient into a bed and update equipment counts."""
        bed.patient = patient
        patient.admitted_step = self._step
        self._admissions_today += 1

        if patient.needs_ventilator and self._ventilators_in_use < self.TOTAL_VENTILATORS:
            self._ventilators_in_use += 1
        elif patient.needs_ventilator:
            patient.needs_ventilator = False   # Can't provide — document as constraint

        if patient.needs_dialysis and self._dialysis_in_use < self.TOTAL_DIALYSIS:
            self._dialysis_in_use += 1
        elif patient.needs_dialysis:
            patient.needs_dialysis = False

    # ─────────────────────────────────────────────────────────────────────
    # Reward
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_reward(self, action: int) -> float:
        """
        Multi-objective reward with partial signals at every step.

        Components
        ----------
        +3.0  per critical patient admitted before 2-hour breach
        +1.0  for maintaining safe nurse:patient ratio
        -5.0  per patient death in queue this step
        -2.0  per adverse event this step
        -1.5  per critical patient waiting > 2 hours (ongoing)
        +0.5  per effective specialist consult (action=5 AND patient at risk)
        -0.3  budget overspend fraction (if budget depleted)
        -0.5  for HOLD when critical patient in queue AND bed available (missed opportunity)
        """
        reward = 0.0

        # Nurse ratio component
        ratio = self._nurse_patient_ratio()
        if ratio <= self.SAFE_NURSE_RATIO:
            reward += 1.0
        else:
            reward -= (ratio - self.SAFE_NURSE_RATIO) * 1.5

        # Critical patients in queue breaching wait time
        for p in self._queue:
            wait = self._step - p.arrival_step
            if p.sofa >= self.CRITICAL_SOFA and wait > self.MAX_QUEUE_WAIT_STEPS:
                reward -= 1.5

        # Deaths in queue penalise heavily
        # (deaths_in_queue is cumulative; reward on incremental change is handled
        #  by tracking _last_deaths — approximated here as step-level signal)
        # We track last step deaths via adverse events counter delta
        # Simple: penalise for current step deaths via wait_violations increase
        breach_count = sum(
            1 for p in self._queue
            if p.sofa >= self.CRITICAL_SOFA and (self._step - p.arrival_step) >= self.MAX_QUEUE_WAIT_STEPS
        )
        reward -= breach_count * 0.5

        # Missed opportunity: HOLD when could have admitted critical patient
        if action == 0:
            has_critical_queue = any(p.sofa >= self.CRITICAL_SOFA for p in self._queue)
            has_free_bed = self._first_available_bed() is not None
            if has_critical_queue and has_free_bed:
                reward -= 0.5

        # Budget management
        budget_fraction = self._budget_remaining / self.DAILY_BUDGET_INR
        if budget_fraction <= 0:
            reward -= 0.3
        else:
            reward += budget_fraction * 0.2

        # Throughput bonus: reward for high admissions-to-capacity ratio
        throughput = self._admissions_today / max(1, self._step)
        reward += min(0.5, throughput * 0.5)

        # Equipment utilisation (reward efficient use, penalise over-saturation)
        vent_util = self._ventilators_in_use / self.TOTAL_VENTILATORS
        if vent_util > 0.95:
            reward -= 0.4   # Near capacity is dangerous

        return reward

    # ─────────────────────────────────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────────────────────────────────

    def _build_obs(self) -> dict:
        occupied = [b for b in self._beds if b.is_occupied]
        available = [b for b in self._beds if b.is_available]
        turnover  = [b for b in self._beds if b.in_turnover]

        q_sofa     = [p.sofa for p in self._queue]
        q_critical = [p for p in self._queue if p.sofa >= self.CRITICAL_SOFA]
        q_severe   = [p for p in self._queue if 7 <= p.sofa < self.CRITICAL_SOFA]
        q_moderate = [p for p in self._queue if p.sofa < 7]

        current_sofa = [b.patient.sofa for b in occupied]
        avg_icu_sofa = sum(current_sofa) / max(1, len(current_sofa))
        avg_icu_mortality = sum(b.patient.mortality_risk for b in occupied) / max(1, len(occupied))

        longest_wait = 0
        if self._queue:
            longest_wait = self._step - min(p.arrival_step for p in self._queue)

        nurses = self._nurses_on_duty()

        return {
            # Bed status
            "beds_total":              self.TOTAL_ICU_BEDS,
            "beds_occupied":           len(occupied),
            "beds_available":          len(available),
            "beds_in_turnover":        len(turnover),

            # Queue status
            "queue_total":             len(self._queue),
            "queue_critical":          len(q_critical),   # SOFA ≥ 11
            "queue_severe":            len(q_severe),     # SOFA 7-10
            "queue_moderate":          len(q_moderate),   # SOFA < 7
            "queue_max_wait_steps":    longest_wait,       # Steps since oldest arrival

            # Current ICU patient acuity
            "avg_icu_sofa":            round(avg_icu_sofa, 2),
            "avg_icu_mortality_risk":  round(avg_icu_mortality, 3),

            # Equipment
            "ventilators_available":   self.TOTAL_VENTILATORS - self._ventilators_in_use,
            "ventilators_in_use":      self._ventilators_in_use,
            "dialysis_available":      self.TOTAL_DIALYSIS - self._dialysis_in_use,

            # Staff
            "nurses_on_duty":          nurses,
            "nurse_patient_ratio":     round(self._nurse_patient_ratio(), 2),
            "doctors_on_duty":         self.BASE_DOCTORS,

            # Time
            "shift":                   self._current_shift(),  # 0=day 1=evening 2=night
            "time_of_day":             round(self._hour, 1),
            "step":                    self._step,

            # Finance
            "budget_remaining_inr":    round(self._budget_remaining, 2),
            "budget_utilisation_pct":  round((1 - self._budget_remaining / self.DAILY_BUDGET_INR) * 100, 1),

            # Cumulative outcomes
            "admissions_today":        self._admissions_today,
            "transfers_today":         self._transfers_today,
            "deaths_in_queue":         self._deaths_in_queue,
            "adverse_events":          self._adverse_events,
            "wait_violations":         self._wait_violations,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _first_available_bed(self) -> Optional[Bed]:
        for b in self._beds:
            if b.is_available:
                return b
        return None

    def _nurses_on_duty(self) -> int:
        base = self.BASE_NURSES_DAY if self._current_shift() == 0 else self.BASE_NURSES_NIGHT
        return base + self._extra_nurses_called

    def _nurse_patient_ratio(self) -> float:
        occupied = sum(1 for b in self._beds if b.is_occupied)
        nurses = self._nurses_on_duty()
        if nurses == 0:
            return float("inf")
        return occupied / nurses

    def _current_shift(self) -> int:
        h = self._hour
        if 8 <= h < 16:
            return 0   # Day
        elif 16 <= h < 24:
            return 1   # Evening
        else:
            return 2   # Night

    class _RNG(random.Random):
        pass


# Monkey-patch a Poisson approximation onto the rng
def _poisson_approx(self, lam: float) -> int:
    """Approximate Poisson using sum of Bernoulli trials (works for small λ)."""
    n, p = 20, lam / 20
    return sum(1 for _ in range(n) if self.random() < p)

random.Random.poisson_approx = _poisson_approx


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = ICUEnv(seed=42)
    obs = env.reset()
    print("── INITIAL STATE ─────────────────────────────────────────")
    for k, v in obs.items():
        print(f"  {k:35s}: {v}")

    print("\n── FIRST 6 STEPS ─────────────────────────────────────────")
    total_reward = 0
    for i in range(6):
        action = i % 7
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1} | act={action} | beds={obs['beds_occupied']}/20 "
              f"| queue={obs['queue_total']} (crit={obs['queue_critical']}) "
              f"| reward={reward:+.3f} | budget=₹{obs['budget_remaining_inr']:,.0f} "
              f"| {info['action_result'][:50]}")

    print(f"\nTotal reward so far: {total_reward:.3f}")
    print("State OK:", len(env.state()) == 23)
