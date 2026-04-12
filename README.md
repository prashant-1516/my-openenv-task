---
title: ICU Resource Allocation OpenEnv
emoji: 🏥
colorFrom: red
colorTo: Yellow
sdk: docker
pinned: false
---

# 🏥 ICU Resource Allocation — OpenEnv

A **real-world OpenEnv environment** modelling a 20-bed ICU in a 500-bed Indian tertiary-care hospital. An AI agent acts as the **ICU charge coordinator**, making resource allocation decisions every 30 minutes over a 24-hour duty cycle.

---

## 🌍 Real-World Grounding

This environment is not a simulation of a game — it models actual clinical practice:

| Element | Real-World Source |
|---------|-------------------|
| SOFA scoring (0–24) | Vincent et al., *JAMA* 1996 — gold-standard ICU triage |
| SOFA → mortality | Ferreira et al., *JAMA* 2001 |
| Nurse:patient ratio | NABH ICU Standard (India) — max 1:2 |
| Bed turnover times | Agnihotri et al., *Indian J Crit Care Med* 2019 |
| Arrival patterns | Arias-Verdú et al., *Critical Care Medicine* 2017 |
| Cost calibration | CGHS ICU package rates 2023 (Central Govt Health Scheme) |

---

## 🎮 Action Space (7 discrete actions)

| # | Action | Effect | Cost |
|---|--------|--------|------|
| 0 | **HOLD** | Observe, no change | ₹0 |
| 1 | **ADMIT_CRITICAL** | Admit highest-SOFA patient | — |
| 2 | **ADMIT_FIFO** | Admit longest-waiting patient | — |
| 3 | **TRANSFER_OUT** | Move stable patient to step-down | — |
| 4 | **CALL_EXTRA_NURSE** | +1 nurse, improves ratio | ₹1,200 |
| 5 | **SPECIALIST_CONSULT** | −15% mortality risk on sickest patient | ₹3,500 |
| 6 | **EXPEDITE_BED** | Faster bed turnover | ₹600 |

---

## 👁️ Observation Space (27 fields)

Rich, clinically meaningful state:
- **Bed status**: occupied / available / in turnover
- **Queue**: total waiting, broken down by SOFA severity (critical/severe/moderate), longest wait
- **Patient acuity**: avg SOFA score, avg mortality risk of current ICU patients
- **Equipment**: ventilators & dialysis machines available/in-use
- **Staff**: nurses on duty, nurse:patient ratio, doctors
- **Time**: hour, shift (Day/Evening/Night), step
- **Budget**: remaining INR, utilisation %
- **Outcomes**: admissions, transfers, deaths in queue, adverse events, wait violations

---

## 📋 Tasks

### 🟢 Easy — Prevent Preventable Deaths
Zero queue deaths + nurse:patient ratio ≤ 2.0 in ≥90% of steps.

### 🟡 Medium — NABH-Compliant Critical Care
All Easy criteria + all critical patients (SOFA ≥ 11) admitted within **2 hours** of arrival. Maps to NABH Grade-B ICU standard.

### 🔴 Hard — JCI-Grade ICU Excellence
All Medium criteria + zero adverse events + budget ≤ 85% + average SOFA non-increasing. Maps to JCI / NABH Grade-A accreditation benchmarks.

---

## 🔌 API

```bash
# Reset
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"seed": 42}'

# Step (action 1 = ADMIT_CRITICAL)
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": 1}'

# State
curl http://localhost:7860/state
```

---

## 🚀 Local Setup

```bash
pip install -r requirements.txt

# Run server
python app.py

# Run graders (validates all scores in [0,1])
python graders/task_graders.py

# Run LLM inference (needs env vars)
export API_BASE_URL=...
export MODEL_NAME=...
export HF_TOKEN=...
python inference.py
```

---

## 📁 Structure

```
hospital_env/
├── env.py                 # ICU environment (SOFA, NABH, Poisson arrivals)
├── app.py                 # FastAPI: /reset /step /state
├── inference.py           # LLM agent via OpenAI client
├── openenv.yaml           # OpenEnv spec
├── requirements.txt
├── Dockerfile
├── README.md
└── graders/
    └── task_graders.py    # Easy / Medium / Hard graders → scores [0,1]
```
