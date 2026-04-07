---
title: openenv-invoice-audit
sdk: docker
pinned: false
tags:
 - openenv
---

# OpenEnv Procurement Invoice Audit

This benchmark simulates a real accounts-payable workflow where an agent reviews invoices against purchase orders, vendor risk, and policy constraints. The goal is to decide whether to approve clean invoices, request correction for recoverable issues, or escalate suspicious patterns (including invoice splitting fraud detection).

The environment is deterministic, graded programmatically, and exposed through OpenEnv-style `reset()`, `step()`, and `state()` behavior.

## Table of Contents

- [Why This Environment](#why-this-environment)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Tasks and Difficulty](#tasks-and-difficulty)
- [Reward Design](#reward-design)
- [Local Setup](#local-setup)
- [Run the API](#run-the-api)
- [Run Tests](#run-tests)
- [Run Baseline Inference](#run-baseline-inference)
- [Docker](#docker)
- [Hugging Face Space](#hugging-face-space)
- [Baseline Scores](#baseline-scores)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Why This Environment

Invoice audit is a real operational task with direct financial risk. This environment is designed to test:

- **Policy compliance behavior** - Adherence to organizational policies and constraints
- **Safe decision making under ambiguity** - Handling incomplete or conflicting information
- **Calibration** - Confidence-aware actions that reflect true uncertainty
- **Anti-loop trajectory quality** - Avoiding repetitive or circular decision patterns

## Observation Space

`Observation` is a typed object containing:

- `task_id` - Unique identifier for the task
- `invoice_id` - Unique invoice identifier
- `vendor_name` - Name of the vendor submitting the invoice
- `amount` - Invoice amount
- `tax_code` - Tax classification code
- `due_date` - Payment due date
- `po_number` - Associated purchase order number
- `risk_level` - Risk classification (low, medium, high)
- `allowed_actions` - List of valid actions for current state
- `audit_trail` - Running action history for the current episode
- `context` - Task metadata and additional context

### Example Observation

```json
{
  "task_id": "task_001",
  "invoice_id": "INV-20260402-001",
  "vendor_name": "ABC Supplies Inc.",
  "amount": 4500.00,
  "tax_code": "TAX_10",
  "due_date": "2026-04-30",
  "po_number": "PO-2026-0123",
  "risk_level": "medium",
  "allowed_actions": ["review_invoice", "request_correction", "approve_invoice"],
  "audit_trail": [
    {"action": "review_invoice", "timestamp": "2026-04-02T10:00:00Z", "notes": "Initial review"}
  ],
  "context": {"priority": "high", "department": "operations"}
}
```

## Action Space

`Action` is a typed object:

- `action_type` - One of: `review_invoice`, `request_correction`, `approve_invoice`, `escalate_case`, `ask_vendor_question`
- `payload` - Action-specific fields (dictionary)
- `confidence` - Float in `[0.0, 1.0]` indicating decision confidence
- `reasoning` - Short explanation string for the action

### Example Action

```json
{
  "action_type": "request_correction",
  "payload": {
    "issue": "PO amount mismatch",
    "expected_amount": 4000.00,
    "invoice_amount": 4500.00,
    "required_docs": ["revised_invoice", "vendor_explanation"]
  },
  "confidence": 0.95,
  "reasoning": "Invoice amount exceeds PO by 12.5%. Requesting clarification before approval."
}
```

## Tasks and Difficulty

- **`easy_single_mismatch`** (Easy)  
  Single PO amount mismatch plus at least one clean invoice. Tests basic mismatch detection.

- **`medium_policy_tangle`** (Medium)  
  Multi-constraint policy issues (tax + due date + threshold interactions). Tests ability to navigate complex policy constraints.

- **`hard_fraud_detection`** (Hard)  
  Invoice splitting fraud detection scenario with high-risk vendor patterns. Tests pattern recognition and risk assessment.

All tasks have deterministic graders with scores in `[0.0, 1.0]`.

## Reward Design

Rewards provide dense signal over the trajectory:

- **Positive rewards:**
  - `+1.0` for correct invoice approval (clean invoice)
  - `+0.8` for appropriate correction request
  - `+0.5` for valid initial review action
  - Calibration bonus when `confidence` aligns with correctness

- **Penalties:**
  - `-0.5` for incorrect approvals (unsafe decisions)
  - `-0.3` for unnecessary escalations
  - `-0.2` for looping (repeating the same action)
  - `-0.1` for contradictory actions (e.g., approving then requesting correction)

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

Expected response:

```json
{"status": "ok", "version": "1.0"}
```

## Run Tests

```bash
pytest -q
openenv validate
```

## Run Baseline Inference

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token (or `OPENAI_API_KEY` as fallback) | `hf_your_token_here` |
| `API_BASE_URL` | API endpoint (optional) | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier (optional) | `Qwen/Qwen2.5-72B-Instruct` |
| `LOCAL_IMAGE_NAME` | Local image name (optional, local mode only) | `my-model:latest` |

### Running Inference

```bash
set HF_TOKEN=hf_your_token_here
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

`inference.py` emits structured logs with the following format:

- `[START] ...` - Marks the beginning of an inference run
- `[STEP] ...` - Logs each step in the episode trajectory
- `[END] ...` - Marks completion with final scores

## Docker

Build and run the Docker container:

```bash
docker build -t openenv-invoice-audit .
docker run --rm -p 7860:7860 openenv-invoice-audit
```

Verify the container is running:

```bash
curl http://localhost:7860/
```

## Hugging Face Space

This repository is configured for Docker Spaces (`sdk: docker`) and serves on port `7860`. Push to the Hugging Face Hub to auto-deploy.
For metadata options, see the Spaces config reference: https://huggingface.co/docs/hub/spaces-config-reference

## Baseline Scores

| Task | Score | Status |
|------|-------|--------|
| `easy_single_mismatch` | 0.50 | Baseline pending |
| `medium_policy_tangle` | 0.50 | Baseline pending |
| `hard_fraud_detection` | 0.50 | Baseline pending |
| **Average** | **0.50** | - |

*Note: Baseline scores will be updated once reference implementations are validated.*

## Troubleshooting

### Common Issues

**Import Errors or Missing Dependencies**
- Ensure all packages in `requirements.txt` are installed
- Verify Python version compatibility (3.8+)
- Try: `pip install --upgrade -r requirements.txt`

**Port Already in Use (7860)**
- Use a different port: `uvicorn app:app --port 8000`
- Or kill the existing process: `lsof -ti:7860 | xargs kill -9`

**Inference Fails with API Errors**
- Verify `HF_TOKEN` is set correctly: `echo $HF_TOKEN`
- Check API connectivity: `curl https://router.huggingface.co/v1/status`
- Review detailed logs in `inference.py` output

**Tests Fail Unexpectedly**
- Clear cache: `rm -rf __pycache__ .pytest_cache`
- Run with verbose output: `pytest -v`
- Check for conflicting environment variables

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository** on GitHub
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit with clear messages:
   ```bash
   git commit -m "feat: add new invoice validation rule"
   ```
4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** with a description of your changes

Please ensure:
- Code follows existing style conventions
- Tests pass: `pytest -q`
- Documentation is updated as needed

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.