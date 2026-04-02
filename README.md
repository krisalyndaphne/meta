---
title: openenv-invoice-audit
emoji: "🧾"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# OpenEnv Procurement Invoice Audit

This benchmark simulates a real accounts-payable workflow where an agent reviews invoices against purchase orders, vendor risk, and policy constraints.  
The goal is to decide whether to:

- approve clean invoices,
- request correction for recoverable issues, or
- escalate suspicious patterns (including invoice splitting fraud detection).

The environment is deterministic, graded programmatically, and exposed through OpenEnv-style `reset()`, `step()`, and `state()` behavior.

## Why This Environment

Invoice audit is a real operational task with direct financial risk. This environment is designed to test:

- policy compliance behavior,
- safe decision making under ambiguity,
- calibration (confidence-aware actions),
- and anti-loop trajectory quality.

## Observation Space

`Observation` is a typed object containing:

- `task_id`
- `invoice_id`
- `vendor_name`
- `amount`
- `tax_code`
- `due_date`
- `po_number`
- `risk_level`
- `allowed_actions`
- `audit_trail` (running action history for the current episode)
- `context` (task metadata)

## Action Space

`Action` is a typed object:

- `action_type`: one of  
  `review_invoice`, `request_correction`, `approve_invoice`, `escalate_case`, `ask_vendor_question`
- `payload`: action-specific fields (dictionary)
- `confidence`: float in `[0.0, 1.0]`
- `reasoning`: short explanation string

## Tasks and Difficulty

- `easy_single_mismatch` (easy)  
  Single PO amount mismatch plus at least one clean invoice.
- `medium_policy_tangle` (medium)  
  Multi-constraint policy issues (tax + due date + threshold interactions).
- `hard_fraud_detection` (hard)  
  Invoice splitting fraud detection scenario with high-risk vendor patterns.

All tasks have deterministic graders with scores in `[0.0, 1.0]`.

## Reward Design

Rewards provide dense signal over the trajectory:

- positive for meaningful progress (`review`, correct routing, valid completion),
- penalties for looping, contradictions, and unsafe approvals,
- calibration shaping using `confidence` (better reward when confidence aligns with correctness).

## Local Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://127.0.0.1:7860/
```

## Run Tests

```bash
pytest -q
openenv validate
```

## Run Baseline Inference

Required environment variables:

- `HF_TOKEN` (or `OPENAI_API_KEY` fallback in this repo)
- `API_BASE_URL` (optional; default provided)
- `MODEL_NAME` (optional; default provided)
- `LOCAL_IMAGE_NAME` (optional, only if using local image mode)

Example:

```bash
set HF_TOKEN=hf_your_token_here
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

`inference.py` emits strict structured logs:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Docker

Build and run:

```bash
docker build -t openenv-invoice-audit .
docker run --rm -p 7860:7860 openenv-invoice-audit
```

## Hugging Face Space

This repository is configured for Docker Spaces (`sdk: docker`) and serves on port `7860`.

## Baseline Scores (placeholder)

- `easy_single_mismatch`: 0.00
- `medium_policy_tangle`: 0.00
- `hard_fraud_detection`: 0.00
- `average`: 0.00
