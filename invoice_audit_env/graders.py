from __future__ import annotations
from invoice_audit_env.tasks import TASK_FIXTURES

EPS = 1e-6  # small buffer to stay strictly inside (0,1)

def _clamp(value: float) -> float:
    # Force strictly inside (0,1)
    if value <= 0.0:
        return EPS
    if value >= 1.0:
        return 1.0 - EPS
    return value


def grade_episode(task_id: str, action_history: list[str]) -> float:
    fixture = TASK_FIXTURES[task_id]

    # NEVER return constants directly
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

    score = base - penalty

    return _clamp(score)
