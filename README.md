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

# OpenEnv Invoice Audit Environment

This environment simulates procurement invoice auditing for accounts-payable workflows. Agents validate invoice integrity, enforce policy constraints, and decide whether to approve, request correction, or escalate suspected invoice splitting fraud.

## Observation Space

Observation is a typed object with:
- `task_id`, `invoice_id`, `vendor_name`, `amount`, `tax_code`, `due_date`, `po_number`
- `risk_level`
- `allowed_actions`
- `audit_trail` (running action history)
- `context` metadata

## Action Space

Actions are typed with:
- `action_type` in `{review_invoice, request_correction, approve_invoice, escalate_case, ask_vendor_question}`
- `payload` dictionary
- `confidence` float in `[0.0, 1.0]`
- `reasoning` string

## Tasks

- `easy_single_mismatch` (easy): obvious PO amount mismatch plus one clean invoice.
- `medium_policy_tangle` (medium): tax and due-date policy conflict plus one clean invoice.
- `hard_fraud_detection` (hard): invoice splitting fraud detection pattern plus one clean invoice.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Run Tests

```bash
pytest -q
```

## Run Inference

Set environment variables:
- `HF_TOKEN`
- `API_BASE_URL` (optional default provided)
- `MODEL_NAME` (optional default provided)

```bash
python inference.py
```

## Placeholder Baseline Scores

- `easy_single_mismatch`: 0.00
- `medium_policy_tangle`: 0.00
- `hard_fraud_detection`: 0.00
- average: 0.00
