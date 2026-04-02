from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from invoice_audit_env.models import TaskFixture
from invoice_audit_env.tasks import TASK_FIXTURES


@dataclass
class EpisodeState:
    task_id: str
    seed: Optional[int]
    step_num: int = 0
    done: bool = False
    current_invoice_idx: int = 0
    action_history: List[str] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)
    last_action_error: Optional[str] = None
    grader_score: Optional[float] = None

    def snapshot(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "step_num": self.step_num,
            "done": self.done,
            "current_invoice_idx": self.current_invoice_idx,
            "action_history": list(self.action_history),
            "audit_trail": list(self.audit_trail),
            "last_action_error": self.last_action_error,
            "grader_score": self.grader_score,
        }


class StateStore:
    def __init__(self) -> None:
        self._rng = random.Random(0)
        self._episode: Optional[EpisodeState] = None
        self._fixture: Optional[TaskFixture] = None

    @property
    def fixture(self) -> TaskFixture:
        if self._fixture is None:
            raise RuntimeError("Environment not reset.")
        return self._fixture

    @property
    def episode(self) -> EpisodeState:
        if self._episode is None:
            raise RuntimeError("Environment not reset.")
        return self._episode

    def reset(self, task_id: str, seed: Optional[int]) -> EpisodeState:
        if task_id not in TASK_FIXTURES:
            raise ValueError(f"Unknown task_id: {task_id}")
        self._fixture = TASK_FIXTURES[task_id]
        self._rng = random.Random(seed if seed is not None else 0)
        start_index = 0 if len(self._fixture.invoices) == 1 else self._rng.randint(0, len(self._fixture.invoices) - 1)
        self._episode = EpisodeState(task_id=task_id, seed=seed, current_invoice_idx=start_index)
        return self._episode

    def rotate_invoice(self) -> None:
        ep = self.episode
        ep.current_invoice_idx = (ep.current_invoice_idx + 1) % len(self.fixture.invoices)
