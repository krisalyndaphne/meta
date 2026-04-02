from __future__ import annotations

from typing import Dict

from invoice_audit_env.models import Action, Reward, TaskFixture


def compute_reward(
    fixture: TaskFixture,
    action: Action,
    step_num: int,
    action_history: list[str],
    invoice_id: str,
) -> Reward:
    breakdown: Dict[str, float] = {
        "progress": 0.0,
        "calibration": 0.0,
        "loop_penalty": 0.0,
        "contradiction_penalty": 0.0,
        "unsafe_penalty": 0.0,
    }

    if action.action_type == "review_invoice":
        breakdown["progress"] += 0.25
    elif action.action_type in {"request_correction", "escalate_case"}:
        breakdown["progress"] += 0.5
    elif action.action_type == "approve_invoice" and invoice_id in fixture.clean_invoice_ids:
        breakdown["progress"] += 0.6

    if action.action_type in fixture.perfect_actions:
        breakdown["calibration"] += 0.2 * action.confidence
        if action.confidence < 0.4:
            breakdown["calibration"] -= 0.1
    else:
        if action.confidence > 0.8:
            breakdown["calibration"] -= 0.3

    if len(action_history) >= 2 and action_history[-1] == action.action_type == action_history[-2]:
        breakdown["loop_penalty"] -= 0.2
    if step_num >= fixture.max_steps:
        breakdown["loop_penalty"] -= 0.3

    if action.action_type in action_history:
        for pair in fixture.contradiction_pairs:
            if action.action_type in pair and any(p in action_history for p in pair if p != action.action_type):
                breakdown["contradiction_penalty"] -= 0.25
                break

    if action.action_type == "approve_invoice" and invoice_id in fixture.escalation_required_invoice_ids:
        breakdown["unsafe_penalty"] -= 0.8

    total = round(sum(breakdown.values()), 4)
    return Reward(value=total, breakdown=breakdown)
