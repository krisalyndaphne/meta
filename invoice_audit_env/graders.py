from __future__ import annotations

from invoice_audit_env.tasks import TASK_FIXTURES

MIN_SCORE = 0.01
MAX_SCORE = 0.99

def _clamp(value: float) -> float:
    # Phase-2 validator requires strict bounds: 0.0 < score < 1.0.
    return max(MIN_SCORE, min(MAX_SCORE, round(value, 4)))


def grade_episode(task_id: str, action_history: list[str]) -> float:
    fixture = TASK_FIXTURES[task_id]
    if action_history == fixture.perfect_actions:
        return _clamp(1.0)

    if action_history == fixture.worst_actions:
        return _clamp(0.0)

    expected = set(fixture.perfect_actions)
    got = set(action_history)
    overlap = len(expected.intersection(got))
    base = overlap / max(1, len(expected))
    penalty = 0.0
    if "approve_invoice" in action_history and fixture.escalation_required_invoice_ids:
        penalty += 0.4
    if len(action_history) > fixture.max_steps:
        penalty += 0.2
    return _clamp(base - penalty)
