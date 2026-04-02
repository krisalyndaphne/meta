from __future__ import annotations

from typing import Dict, Optional, Tuple

from invoice_audit_env.graders import grade_episode
from invoice_audit_env.models import Action, Observation, Reward
from invoice_audit_env.reward import compute_reward
from invoice_audit_env.state import StateStore


class InvoiceAuditEnv:
    def __init__(self) -> None:
        self.store = StateStore()

    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        episode = self.store.reset(task_id=task_id, seed=seed)
        invoice = self.store.fixture.invoices[episode.current_invoice_idx]
        return Observation(
            task_id=task_id,
            invoice_id=invoice.invoice_id,
            vendor_name=invoice.vendor_name,
            amount=invoice.amount,
            tax_code=invoice.tax_code,
            due_date=invoice.due_date,
            po_number=invoice.po_number,
            risk_level=self.store.fixture.vendors[0].risk_level,
            allowed_actions=["review_invoice", "request_correction", "approve_invoice", "escalate_case", "ask_vendor_question"],
            audit_trail=list(episode.audit_trail),
            context={"description": self.store.fixture.description},
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, object]]:
        episode = self.store.episode
        fixture = self.store.fixture
        episode.step_num += 1
        episode.last_action_error = None

        if episode.done:
            episode.last_action_error = "episode_already_done"

        invoice = fixture.invoices[episode.current_invoice_idx]

        reward = compute_reward(
            fixture=fixture,
            action=action,
            step_num=episode.step_num,
            action_history=episode.action_history,
            invoice_id=invoice.invoice_id,
        )

        episode.action_history.append(action.action_type)
        episode.audit_trail.append(f"step={episode.step_num} action={action.action_type} confidence={action.confidence:.2f}")

        done = False
        if action.action_type in {"approve_invoice", "request_correction", "escalate_case"}:
            done = True
        if episode.step_num >= fixture.max_steps:
            done = True
        episode.done = done

        grader_score = None
        if done:
            grader_score = grade_episode(fixture.task_id, episode.action_history)
            episode.grader_score = grader_score
        else:
            self.store.rotate_invoice()

        invoice = fixture.invoices[episode.current_invoice_idx]
        observation = Observation(
            task_id=fixture.task_id,
            invoice_id=invoice.invoice_id,
            vendor_name=invoice.vendor_name,
            amount=invoice.amount,
            tax_code=invoice.tax_code,
            due_date=invoice.due_date,
            po_number=invoice.po_number,
            risk_level=fixture.vendors[0].risk_level,
            allowed_actions=["review_invoice", "request_correction", "approve_invoice", "escalate_case", "ask_vendor_question"],
            audit_trail=list(episode.audit_trail),
            context={"description": fixture.description},
        )
        info: Dict[str, object] = {
            "last_action_error": episode.last_action_error,
            "step_num": episode.step_num,
            "task_id": fixture.task_id,
            "grader_score": grader_score,
        }
        return observation, reward, done, info

    def state(self) -> Dict[str, object]:
        episode = self.store.episode
        return episode.snapshot()
